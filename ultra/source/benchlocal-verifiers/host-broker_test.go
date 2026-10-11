//go:build linux

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

type roundTrip func(*http.Request) (*http.Response, error)

func (f roundTrip) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }
func result(status int, s string) *http.Response {
	return &http.Response{StatusCode: status, Body: io.NopCloser(strings.NewReader(s)), Header: http.Header{}}
}
func sampleConfig() Config {
	return Config{Controller: "test-ultra", Image: "sha256:" + strings.Repeat("a", 64), Root: runtimeRoot("test-ultra")}
}
func TestConfigRejectsUntrustedTargets(t *testing.T) {
	valid := sampleConfig()
	if e := validateConfig(valid); e != nil {
		t.Fatal(e)
	}
	for _, change := range []func(*Config){func(c *Config) { c.Image = "example:latest" }, func(c *Config) { c.Controller = "../other" }, func(c *Config) { c.Root = "/" }, func(c *Config) { c.Root = "/run/sm75-benchlocal/test/../escape" }} {
		c := valid
		change(&c)
		if validateConfig(c) == nil {
			t.Fatal("unsafe configuration accepted")
		}
	}
}
func TestContainerSpecCannotExposeProduction(t *testing.T) {
	b := newBroker(sampleConfig(), nil)
	j := &Job{ID: randomID(), RunID: randomID(), Role: "cli-40", Token: "job-only", Dir: filepath.Join(b.cfg.Root, "jobs", "fixture")}
	s := b.spec(j)
	h := s["HostConfig"].(map[string]any)
	if s["Image"] != b.cfg.Image || h["NetworkMode"] != "none" || h["Runtime"] != "runc" || h["ReadonlyRootfs"] != true || h["AutoRemove"] != true {
		t.Fatal("unsafe isolation flags")
	}
	for _, name := range []string{"Privileged", "PidMode", "Devices", "DeviceRequests", "VolumesFrom", "MaskedPaths", "ReadonlyPaths"} {
		if _, ok := h[name]; ok {
			t.Fatalf("unexpected override: %s", name)
		}
	}
	binds := h["Binds"].([]string)
	if len(binds) != 1 || !strings.HasSuffix(binds[0], "/worker.sock:/run/sm75-job.sock:ro") {
		t.Fatal(binds)
	}
	caps := h["CapAdd"].([]string)
	if strings.Join(caps, ",") != "SETUID,SETGID,KILL" {
		t.Fatal(caps)
	}
	if strings.Join(h["SecurityOpt"].([]string), ",") != "no-new-privileges" {
		t.Fatal("default security changed")
	}
	if h["Memory"] != int64(4)<<30 || h["PidsLimit"] != 512 {
		t.Fatal("limits missing")
	}
}
func TestProtocolRejectsArbitraryFieldsAndRoles(t *testing.T) {
	b := newBroker(sampleConfig(), nil)
	for _, f := range []Frame{frame(map[string]any{"type": "start", "id": "r:1", "runId": randomID(), "role": "cli-40", "mount": "/:/host"}), frame(map[string]any{"type": "start", "id": "r:2", "runId": randomID(), "role": "shell"}), frame(map[string]any{"type": "call", "id": "r:3", "runId": randomID(), "route": "/exec", "body": map[string]string{}})} {
		a, c := net.Pipe()
		w := wire(a)
		done := make(chan error, 1)
		go func() { done <- b.command(w, f) }()
		r := bufio.NewReader(c)
		raw, e := r.ReadBytes('\n')
		if e != nil {
			t.Fatal(e)
		}
		var got map[string]any
		if json.Unmarshal(raw, &got) != nil || got["error"] == nil {
			t.Fatalf("request was not rejected: %s", raw)
		}
		if e = <-done; e != nil {
			t.Fatal(e)
		}
		w.close()
		c.Close()
	}
}
func TestCleanupCannotDeleteAnUnrelatedContainer(t *testing.T) {
	count := 0
	d := &Docker{&http.Client{Transport: roundTrip(func(r *http.Request) (*http.Response, error) {
		count++
		if r.Method != "GET" {
			t.Fatal("delete was attempted")
		}
		return result(200, `{"Id":"unrelated","Config":{"Labels":{}}}`), nil
	})}}
	b := newBroker(sampleConfig(), d)
	if b.removeOwned() == nil || count != 1 {
		t.Fatal("unrelated container was not protected")
	}
}
func TestCleanupDeletesOnlyOwnedContainer(t *testing.T) {
	b := newBroker(sampleConfig(), nil)
	var methods []string
	b.d = &Docker{&http.Client{Transport: roundTrip(func(r *http.Request) (*http.Response, error) {
		methods = append(methods, r.Method+" "+r.URL.Path)
		if r.Method == "GET" {
			raw, _ := json.Marshal(map[string]any{"Id": "owned-id", "Config": map[string]any{"Labels": map[string]string{labelOwner: b.owner, "io.sm75.benchlocal.kind": "temporary-job"}}})
			return result(200, string(raw)), nil
		}
		if r.URL.Path != "/containers/owned-id" || r.URL.Query().Get("force") != "true" {
			t.Fatal("unexpected delete target")
		}
		return result(204, ""), nil
	})}}
	if e := b.removeOwned(); e != nil {
		t.Fatal(e)
	}
	if len(methods) != 2 {
		t.Fatal(methods)
	}
}
func TestStopInterruptsPendingCreate(t *testing.T) {
	cfg := sampleConfig()
	cfg.Root = t.TempDir()
	if e := os.Mkdir(filepath.Join(cfg.Root, "jobs"), 0700); e != nil {
		t.Fatal(e)
	}
	started := make(chan struct{})
	var once sync.Once
	d := &Docker{&http.Client{Transport: roundTrip(func(r *http.Request) (*http.Response, error) {
		switch {
		case strings.HasPrefix(r.URL.Path, "/images/"):
			return result(200, `{"Config":{"Labels":{"io.sm75.benchlocal.execution":"ondemand-v1"}}}`), nil
		case r.Method == "GET":
			return result(404, ""), nil
		case r.URL.Path == "/containers/create":
			once.Do(func() { close(started) })
			<-r.Context().Done()
			return nil, context.Canceled
		default:
			t.Fatal("unexpected Docker operation")
			return nil, context.Canceled
		}
	})}}
	b := newBroker(cfg, d)
	a, c := net.Pipe()
	defer c.Close()
	w := wire(a)
	defer w.close()
	b.control = w
	run := randomID()
	done := make(chan error, 1)
	go func() { _, e := b.start(w, run, "cli-40"); done <- e }()
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("creation did not start")
	}
	stopped := make(chan error, 1)
	go func() { stopped <- b.stop(w, run, true) }()
	select {
	case e := <-done:
		if e == nil {
			t.Fatal("cancelled creation succeeded")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("cancel failed to interrupt creation")
	}
	if e := <-stopped; e != nil {
		t.Fatal(e)
	}
	if b.active != nil {
		t.Fatal("active task survived cancellation")
	}
}
func TestNamespaceIdentityNeedsFullContainerID(t *testing.T) {
	if belongs(os.Getpid(), "") || belongs(os.Getpid(), "docker") || belongs(os.Getpid(), strings.Repeat("f", 64)) {
		t.Fatal("unrelated process accepted")
	}
}

func TestInvalidFramesAreRejectedBeforeQueueing(t *testing.T) {
	for _, v := range []map[string]any{{"type": "hello", "id": "r:1", "data": strings.Repeat("x", 8192)}, {"type": "call", "id": "r:2", "runId": randomID(), "route": "/verify-answer", "body": map[string]string{"answer": strings.Repeat("x", 3<<20)}}, {"type": "start", "id": "r:3", "runId": randomID(), "role": "cli-40", "image": "untrusted"}} {
		if validateControl(frame(v)) {
			t.Fatal("unsafe frame accepted")
		}
	}
	if !validateControl(frame(map[string]any{"type": "start", "id": "r:4", "runId": randomID(), "role": "cli-40"})) {
		t.Fatal("valid start rejected")
	}
}

func TestCrashCleanupOnlyRemovesOwnedTaskDirectories(t *testing.T) {
	c := sampleConfig()
	c.Root = t.TempDir()
	root := filepath.Join(c.Root, "jobs")
	if e := os.Mkdir(root, 0700); e != nil {
		t.Fatal(e)
	}
	stale := filepath.Join(root, randomID())
	os.Mkdir(stale, 0700)
	os.WriteFile(filepath.Join(stale, "socket-marker"), []byte("fixture"), 0600)
	keep := filepath.Join(root, "unrelated")
	os.Mkdir(keep, 0700)
	b := newBroker(c, nil)
	if e := b.cleanJobDirs(); e != nil {
		t.Fatal(e)
	}
	if _, e := os.Stat(stale); !os.IsNotExist(e) {
		t.Fatal("stale task survived")
	}
	if _, e := os.Stat(keep); e != nil {
		t.Fatal("unrelated data removed")
	}
	outside := t.TempDir()
	marker := filepath.Join(outside, "keep")
	os.WriteFile(marker, []byte("keep"), 0600)
	os.Symlink(outside, filepath.Join(root, randomID()))
	if b.cleanJobDirs() == nil {
		t.Fatal("symlink task accepted")
	}
	if _, e := os.Stat(marker); e != nil {
		t.Fatal("cleanup followed symlink")
	}
}

func TestLongControllerNamesFitUnixSocketLimit(t *testing.T) {
	c := sampleConfig()
	c.Controller = "IP53_" + strings.Repeat("model-", 15)
	c.Root = runtimeRoot(c.Controller)
	if e := validateConfig(c); e != nil {
		t.Fatal(e)
	}
	p := filepath.Join(c.Root, "jobs", randomID(), "worker.sock")
	if len(p) >= 108 {
		t.Fatalf("Unix socket path too long: %d", len(p))
	}
}
