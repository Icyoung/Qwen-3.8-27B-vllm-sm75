//go:build linux

// Fixed-purpose BenchLocal launcher. Only this host process has Docker access.
package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"syscall"
	"time"
)

const maxFrame = 24 << 20
const labelOwner = "io.sm75.benchlocal.owner"
const workerPath = "/opt/sm75-workbench/benchlocal-verifiers/ondemand-gateway.mjs"

var roles = map[string]bool{"bugfind-15": true, "cli-40": true, "hermesagent-20": true}
var routes = map[string]bool{"/verify-answer": true, "/verify-canonical": true, "/verify-oneshot": true, "/verify-multiround": true, "/run-scenario": true}
var uuidRE = regexp.MustCompile(`^[a-f0-9]{8}-[a-f0-9]{4}-[1-5][a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$`)
var imageRE = regexp.MustCompile(`^sha256:[a-f0-9]{64}$`)
var idRE = regexp.MustCompile(`^[a-zA-Z0-9:_-]{1,96}$`)

type Config struct {
	Controller string `json:"controller"`
	Image      string `json:"image"`
	Root       string `json:"root"`
	Seccomp    string `json:"seccomp,omitempty"`
}
type Frame map[string]json.RawMessage

func str(f Frame, k string) string { var s string; _ = json.Unmarshal(f[k], &s); return s }
func frame(v any) Frame            { b, _ := json.Marshal(v); var f Frame; _ = json.Unmarshal(b, &f); return f }
func parse(raw []byte) (Frame, error) {
	if len(raw) > maxFrame {
		return nil, errors.New("frame too large")
	}
	var f Frame
	if json.Unmarshal(raw, &f) != nil || f == nil {
		return nil, errors.New("invalid frame")
	}
	return f, nil
}
func keys(f Frame, names ...string) bool {
	allowed := map[string]bool{}
	for _, n := range names {
		allowed[n] = true
	}
	for n := range f {
		if !allowed[n] {
			return false
		}
	}
	return true
}
func validateControl(f Frame) bool {
	t, id := str(f, "type"), str(f, "id")
	if !idRE.MatchString(id) {
		return false
	}
	switch t {
	case "hello", "ping":
		return keys(f, "type", "id")
	case "start":
		return keys(f, "type", "id", "runId", "role") && uuidRE.MatchString(str(f, "runId")) && roles[str(f, "role")]
	case "stop":
		return keys(f, "type", "id", "runId") && uuidRE.MatchString(str(f, "runId"))
	case "call":
		return keys(f, "type", "id", "runId", "route", "body") && uuidRE.MatchString(str(f, "runId")) && routes[str(f, "route")] && len(f["body"]) > 0 && len(f["body"]) <= 2<<20 && f["body"][0] == '{'
	case "model-head", "model-data", "model-end", "model-error":
		return keys(f, "type", "id", "jobId", "status", "headers", "data", "error") && uuidRE.MatchString(str(f, "jobId")) && len(f["data"]) <= 256<<10 && len(f["headers"]) <= 2048 && len(f["error"]) <= 1024
	}
	return false
}

type Wire struct {
	net.Conn
	mu   sync.Mutex
	once sync.Once
	done chan struct{}
}

func wire(c net.Conn) *Wire { return &Wire{Conn: c, done: make(chan struct{})} }
func (w *Wire) close()      { w.once.Do(func() { close(w.done); _ = w.Conn.Close() }) }
func (w *Wire) send(v any) error {
	b, e := json.Marshal(v)
	if e != nil || len(b) > maxFrame {
		return errors.New("frame too large")
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	_ = w.SetWriteDeadline(time.Now().Add(10 * time.Second))
	_, e = w.Write(append(b, '\n'))
	if e != nil {
		w.close()
	}
	return e
}
func (w *Wire) read(fn func(Frame) error) error {
	r := bufio.NewScanner(w)
	r.Buffer(make([]byte, 4096), maxFrame)
	for r.Scan() {
		f, e := parse(r.Bytes())
		if e != nil {
			return e
		}
		if e = fn(f); e != nil {
			return e
		}
	}
	return r.Err()
}
func response(w *Wire, id string, value any, e error) {
	if e != nil {
		_ = w.send(map[string]any{"type": "reply", "id": id, "error": e.Error()})
	} else {
		_ = w.send(map[string]any{"type": "reply", "id": id, "value": value})
	}
}

type Docker struct{ client *http.Client }

func dockerClient() *Docker {
	tr := &http.Transport{DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
		return (&net.Dialer{Timeout: 5 * time.Second}).DialContext(ctx, "unix", "/var/run/docker.sock")
	}}
	return &Docker{&http.Client{Transport: tr, Timeout: 30 * time.Second}}
}
func (d *Docker) request(ctx context.Context, method, p string, value any, out any) error {
	var body io.Reader
	if value != nil {
		b, e := json.Marshal(value)
		if e != nil {
			return e
		}
		body = bytes.NewReader(b)
	}
	req, e := http.NewRequestWithContext(ctx, method, "http://docker"+p, body)
	if e != nil {
		return e
	}
	req.Header.Set("Content-Type", "application/json")
	res, e := d.client.Do(req)
	if e != nil {
		return fmt.Errorf("Docker connection failed: %w", e)
	}
	defer res.Body.Close()
	b, e := io.ReadAll(io.LimitReader(res.Body, 4<<20))
	if e != nil {
		return e
	}
	if res.StatusCode == 404 {
		return os.ErrNotExist
	}
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		var detail struct{ Message string }
		_ = json.Unmarshal(b, &detail)
		message := regexp.MustCompile(`[a-fA-F0-9]{64}`).ReplaceAllString(detail.Message, "[id]")
		if len(message) > 600 {
			message = message[:600]
		}
		return fmt.Errorf("Docker %s failed (%d): %s", method, res.StatusCode, message)
	}
	if out != nil && len(b) > 0 {
		return json.Unmarshal(b, out)
	}
	return nil
}

type Inspection struct {
	ID     string `json:"Id"`
	Image  string
	Name   string
	Config struct{ Labels map[string]string }
	State  struct {
		Running bool
		Pid     int
	}
}

func (d *Docker) inspect(ctx context.Context, name string) (Inspection, error) {
	var c Inspection
	e := d.request(ctx, "GET", "/containers/"+url.PathEscape(name)+"/json", nil, &c)
	return c, e
}
func peer(c net.Conn) (*syscall.Ucred, error) {
	u, ok := c.(*net.UnixConn)
	if !ok {
		return nil, errors.New("Unix connection required")
	}
	raw, e := u.SyscallConn()
	if e != nil {
		return nil, e
	}
	var cred *syscall.Ucred
	var inner error
	e = raw.Control(func(fd uintptr) {
		cred, inner = syscall.GetsockoptUcred(int(fd), syscall.SOL_SOCKET, syscall.SO_PEERCRED)
	})
	if e != nil {
		return nil, e
	}
	return cred, inner
}
func belongs(pid int, containerID string) bool {
	if len(containerID) != 64 {
		return false
	}
	b, e := os.ReadFile(fmt.Sprintf("/proc/%d/cgroup", pid))
	return e == nil && strings.Contains(string(b), containerID)
}
func runtimeRoot(controller string) string {
	sum := sha256.Sum256([]byte(controller))
	return "/run/sm75-benchlocal/" + hex.EncodeToString(sum[:12])
}
func validateConfig(c Config) error {
	if !regexp.MustCompile(`^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,100}$`).MatchString(c.Controller) || !imageRE.MatchString(c.Image) {
		return errors.New("invalid controller or pinned image")
	}
	if c.Root != runtimeRoot(c.Controller) {
		return errors.New("invalid runtime directory")
	}
	return nil
}
func strictFile(p string) ([]byte, error) {
	s, e := os.Lstat(p)
	if e != nil {
		return nil, e
	}
	st, ok := s.Sys().(*syscall.Stat_t)
	if !ok || st.Uid != 0 || !s.Mode().IsRegular() || s.Mode().Perm()&0022 != 0 {
		return nil, errors.New("configuration must be root-owned and not writable by other users")
	}
	return os.ReadFile(p)
}

type Job struct {
	ID, RunID, Role, Token, Dir, Container string
	owner                                  *Wire
	worker                                 *Wire
	listener                               net.Listener
	ready                                  chan struct{}
	readyOnce                              sync.Once
	pending                                string
	models                                 map[string]bool
	cancel                                 context.CancelFunc
	started                                time.Time
	stopping                               bool
}
type Broker struct {
	cfg                  Config
	d                    *Docker
	owner, name, seccomp string
	mu, ops, statusMu    sync.Mutex
	active               *Job
	control              *Wire
	lastPing             time.Time
	closing              bool
	probeOK              bool
}

func newBroker(c Config, d *Docker) *Broker {
	sum := sha256.Sum256([]byte(c.Controller))
	owner := hex.EncodeToString(sum[:16])
	return &Broker{cfg: c, d: d, owner: owner, name: "sm75-bench-task-" + owner[:24]}
}
func randomID() string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	b[6] = (b[6] & 15) | 64
	b[8] = (b[8] & 63) | 128
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[:4], b[4:6], b[6:8], b[8:10], b[10:])
}
func (b *Broker) writeStatus() {
	b.statusMu.Lock()
	defer b.statusMu.Unlock()
	b.mu.Lock()
	value := map[string]any{"controller": b.cfg.Controller, "image": b.cfg.Image, "connected": b.control != nil, "ready": b.control != nil && b.probeOK && !b.closing, "time": time.Now().Unix(), "pid": os.Getpid(), "active": false}
	if j := b.active; j != nil {
		value["active"] = true
		value["role"] = j.Role
		value["runId"] = j.RunID
		value["jobId"] = j.ID
	}
	b.mu.Unlock()
	raw, _ := json.Marshal(value)
	p := filepath.Join(b.cfg.Root, "status.json")
	if os.WriteFile(p+".tmp", raw, 0600) == nil {
		_ = os.Rename(p+".tmp", p)
	}
}
func (b *Broker) checkImage(ctx context.Context) error {
	var image struct {
		Config struct{ Labels map[string]string }
	}
	if e := b.d.request(ctx, "GET", "/images/"+url.PathEscape(b.cfg.Image)+"/json", nil, &image); e != nil {
		return e
	}
	if image.Config.Labels["io.sm75.benchlocal.execution"] != "ondemand-v1" {
		return errors.New("Ultra image lacks the on-demand execution runtime")
	}
	return nil
}
func (b *Broker) authenticate(c net.Conn) error {
	u, e := peer(c)
	if e != nil || u.Uid != 0 {
		return errors.New("controller identity rejected")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	v, e := b.d.inspect(ctx, b.cfg.Controller)
	if e != nil || !v.State.Running || v.Image != b.cfg.Image || !belongs(int(u.Pid), v.ID) {
		return errors.New("controller container or image does not match")
	}
	return nil
}
func (b *Broker) spec(j *Job) map[string]any {
	opts := []string{"no-new-privileges"}
	if b.seccomp != "" {
		opts = append(opts, "seccomp="+b.seccomp)
	}
	return map[string]any{
		"Image": b.cfg.Image, "Hostname": "benchlocal", "User": "0:0", "WorkingDir": "/", "Entrypoint": []string{"/usr/local/bin/node", "--disable-sigusr1", workerPath}, "Cmd": []string{}, "StopTimeout": 3,
		"Env":        []string{"SM75_BENCHLOCAL_ROLE=" + j.Role, "SM75_JOB_TOKEN=" + j.Token, "SM75_JOB_ID=" + j.ID, "NVIDIA_VISIBLE_DEVICES=void", "CUDA_VISIBLE_DEVICES=", "HOME=/home/verifier", "LANG=C.UTF-8", "NODE_OPTIONS=--disable-sigusr1 --max-old-space-size=768"},
		"Labels":     map[string]string{labelOwner: b.owner, "io.sm75.benchlocal.kind": "temporary-job", "io.sm75.benchlocal.controller": b.cfg.Controller, "io.sm75.benchlocal.run": j.RunID, "io.sm75.benchlocal.role": j.Role},
		"HostConfig": map[string]any{"AutoRemove": true, "Init": true, "Runtime": "runc", "NetworkMode": "none", "CgroupnsMode": "private", "IpcMode": "private", "ReadonlyRootfs": true, "CapDrop": []string{"ALL"}, "CapAdd": []string{"SETUID", "SETGID", "KILL"}, "SecurityOpt": opts, "Memory": int64(4) << 30, "MemorySwap": int64(4) << 30, "NanoCpus": int64(2e9), "PidsLimit": 512, "ShmSize": int64(256) << 20, "Binds": []string{filepath.Join(j.Dir, "worker.sock") + ":/run/sm75-job.sock:ro"}, "Tmpfs": map[string]string{"/tmp": "rw,exec,nosuid,nodev,size=2g,mode=1777", "/workspace": "rw,exec,nosuid,nodev,size=512m,uid=10001,gid=10001,mode=700", "/home/verifier": "rw,nosuid,nodev,size=256m,uid=10001,gid=10001,mode=700"}, "LogConfig": map[string]any{"Type": "json-file", "Config": map[string]string{"max-size": "2m", "max-file": "1"}}, "Ulimits": []map[string]any{{"Name": "nofile", "Soft": 4096, "Hard": 4096}, {"Name": "core", "Soft": 0, "Hard": 0}}}}
}
func (b *Broker) removeOwned() error {
	ctx, cancel := context.WithTimeout(context.Background(), 12*time.Second)
	defer cancel()
	v, e := b.d.inspect(ctx, b.name)
	if errors.Is(e, os.ErrNotExist) {
		return nil
	}
	if e != nil {
		return e
	}
	if v.Config.Labels[labelOwner] != b.owner || v.Config.Labels["io.sm75.benchlocal.kind"] != "temporary-job" {
		return errors.New("temporary container name is occupied by an unrelated container")
	}
	e = b.d.request(ctx, "DELETE", "/containers/"+v.ID+"?force=true&v=true", nil, nil)
	if errors.Is(e, os.ErrNotExist) {
		return nil
	}
	return e
}
func (b *Broker) cleanJobDirs() error {
	root := filepath.Join(b.cfg.Root, "jobs")
	entries, e := os.ReadDir(root)
	if e != nil {
		return e
	}
	for _, entry := range entries {
		if !uuidRE.MatchString(entry.Name()) {
			continue
		}
		p := filepath.Join(root, entry.Name())
		info, e := os.Lstat(p)
		if e != nil {
			return e
		}
		st, ok := info.Sys().(*syscall.Stat_t)
		if !ok || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 || st.Uid != uint32(os.Geteuid()) {
			return errors.New("unexpected stale task directory")
		}
		if e = os.RemoveAll(p); e != nil {
			return e
		}
	}
	return nil
}
func (b *Broker) stop(owner *Wire, run string, notify bool) error {
	b.mu.Lock()
	j := b.active
	if j != nil && (owner != nil && j.owner != owner || run != "" && j.RunID != run) {
		b.mu.Unlock()
		return errors.New("task ownership mismatch")
	}
	if j != nil {
		j.cancel()
	}
	b.mu.Unlock()
	b.ops.Lock()
	defer b.ops.Unlock()
	return b.stopLocked(owner, run, notify)
}
func (b *Broker) stopLocked(owner *Wire, run string, notify bool) error {
	b.mu.Lock()
	j := b.active
	if j != nil && (owner != nil && j.owner != owner || run != "" && j.RunID != run) {
		b.mu.Unlock()
		return errors.New("task ownership mismatch")
	}
	if j != nil {
		j.stopping = true
		j.cancel()
		if j.listener != nil {
			_ = j.listener.Close()
		}
		if j.worker != nil {
			j.worker.close()
		}
	}
	b.mu.Unlock()
	if e := b.removeOwned(); e != nil {
		return e
	}
	if j != nil {
		if filepath.Dir(j.Dir) != filepath.Join(b.cfg.Root, "jobs") {
			return errors.New("invalid task directory")
		}
		_ = os.RemoveAll(j.Dir)
		b.mu.Lock()
		if b.active == j {
			b.active = nil
		}
		pending := j.pending
		j.pending = ""
		b.mu.Unlock()
		if notify && pending != "" {
			response(j.owner, pending, nil, errors.New("execution cancelled"))
		}
	}
	b.writeStatus()
	return nil
}
func (b *Broker) start(owner *Wire, run, role string) (any, error) {
	b.ops.Lock()
	defer b.ops.Unlock()
	select {
	case <-owner.done:
		return nil, errors.New("controller disconnected")
	default:
	}
	b.mu.Lock()
	occupied := b.active != nil
	current := b.control == owner && !b.closing
	b.mu.Unlock()
	if occupied {
		return nil, errors.New("only one temporary executor is allowed")
	}
	if !current {
		return nil, errors.New("controller disconnected")
	}
	if e := b.removeOwned(); e != nil {
		return nil, e
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	token := make([]byte, 32)
	_, _ = rand.Read(token)
	j := &Job{ID: randomID(), RunID: run, Role: role, Token: hex.EncodeToString(token), owner: owner, ready: make(chan struct{}), models: map[string]bool{}, cancel: cancel, started: time.Now()}
	j.Dir = filepath.Join(b.cfg.Root, "jobs", j.ID)
	if e := os.Mkdir(j.Dir, 0700); e != nil {
		cancel()
		return nil, e
	}
	listener, e := net.Listen("unix", filepath.Join(j.Dir, "worker.sock"))
	if e != nil {
		cancel()
		_ = os.RemoveAll(j.Dir)
		return nil, e
	}
	_ = os.Chmod(filepath.Join(j.Dir, "worker.sock"), 0600)
	j.listener = listener
	b.mu.Lock()
	b.active = j
	b.mu.Unlock()
	go b.acceptWorker(j)
	failed := true
	defer func() {
		if failed {
			_ = b.stopLocked(owner, run, true)
		}
	}()
	if e = b.checkImage(ctx); e != nil {
		return nil, e
	}
	var created struct {
		ID string `json:"Id"`
	}
	e = b.d.request(ctx, "POST", "/containers/create?name="+b.name, b.spec(j), &created)
	if e != nil {
		return nil, e
	}
	b.mu.Lock()
	j.Container = created.ID
	b.mu.Unlock()
	if e = b.d.request(ctx, "POST", "/containers/"+created.ID+"/start", nil, nil); e != nil {
		return nil, e
	}
	select {
	case <-j.ready:
	case <-ctx.Done():
		return nil, errors.New("temporary executor startup timed out")
	case <-owner.done:
		return nil, errors.New("controller disconnected")
	}
	b.mu.Lock()
	healthy := b.active == j && j.worker != nil
	b.mu.Unlock()
	if !healthy {
		return nil, errors.New("temporary executor exited during startup")
	}
	failed = false
	b.writeStatus()
	return map[string]any{"jobId": j.ID, "role": j.Role, "mode": "ondemand"}, nil
}
func (b *Broker) acceptWorker(j *Job) {
	c, e := j.listener.Accept()
	if e != nil {
		return
	}
	w := wire(c)
	defer w.close()
	_ = c.SetReadDeadline(time.Now().Add(15 * time.Second))
	first := true
	e = w.read(func(f Frame) error {
		if first {
			first = false
			if str(f, "type") != "ready" || subtle.ConstantTimeCompare([]byte(str(f, "token")), []byte(j.Token)) != 1 {
				return errors.New("worker identity rejected")
			}
			u, err := peer(c)
			b.mu.Lock()
			cid := j.Container
			b.mu.Unlock()
			if err != nil || u.Uid != 0 || !belongs(int(u.Pid), cid) {
				return errors.New("worker container rejected")
			}
			_ = c.SetReadDeadline(time.Time{})
			b.mu.Lock()
			if b.active != j {
				b.mu.Unlock()
				return errors.New("obsolete job")
			}
			j.worker = w
			b.mu.Unlock()
			_ = j.listener.Close()
			j.readyOnce.Do(func() { close(j.ready) })
			return nil
		}
		typ, id := str(f, "type"), str(f, "id")
		if !idRE.MatchString(id) {
			return errors.New("invalid worker message")
		}
		b.mu.Lock()
		if b.active != j {
			b.mu.Unlock()
			return errors.New("obsolete job")
		}
		switch typ {
		case "result":
			if j.pending != id {
				b.mu.Unlock()
				return errors.New("unexpected result")
			}
			j.pending = ""
			b.mu.Unlock()
			return j.owner.send(map[string]any{"type": "reply", "id": id, "value": json.RawMessage(f["value"]), "error": str(f, "error")})
		case "model-open":
			if len(j.models) >= 8 || j.models[id] || !keys(f, "type", "id", "method", "path", "headers", "data") {
				b.mu.Unlock()
				return errors.New("model request limit")
			}
			if !(str(f, "path") == "/benchlocal-inference/v1/chat/completions" && str(f, "method") == "POST" || str(f, "path") == "/benchlocal-inference/v1/models" && str(f, "method") == "GET") || len(f["data"]) > 12<<20 {
				b.mu.Unlock()
				return errors.New("invalid model route")
			}
			j.models[id] = true
		case "model-abort":
			if !j.models[id] {
				b.mu.Unlock()
				return nil
			}
			delete(j.models, id)
		default:
			b.mu.Unlock()
			return errors.New("unknown worker message")
		}
		b.mu.Unlock()
		f["jobId"], _ = json.Marshal(j.ID)
		return j.owner.send(f)
	})
	_ = e
	b.mu.Lock()
	active := b.active == j && !j.stopping
	wasReady := j.worker == w
	b.mu.Unlock()
	if active {
		if wasReady {
			_ = j.owner.send(map[string]any{"type": "fatal", "error": "temporary executor disconnected"})
		}
		go func() { _ = b.stop(j.owner, j.RunID, true) }()
	}
}
func (b *Broker) model(owner *Wire, f Frame) error {
	if !keys(f, "type", "id", "jobId", "status", "headers", "data", "error") || !idRE.MatchString(str(f, "id")) {
		return errors.New("invalid model response")
	}
	typ, id := str(f, "type"), str(f, "id")
	if len(f["data"]) > 256<<10 {
		return errors.New("model chunk too large")
	}
	b.mu.Lock()
	j := b.active
	if j == nil || j.owner != owner || j.ID != str(f, "jobId") || !j.models[id] || j.worker == nil {
		b.mu.Unlock()
		return nil
	}
	w := j.worker
	if typ == "model-end" || typ == "model-error" {
		delete(j.models, id)
	}
	b.mu.Unlock()
	return w.send(f)
}
func (b *Broker) command(w *Wire, f Frame) error {
	typ, id := str(f, "type"), str(f, "id")
	if strings.HasPrefix(typ, "model-") {
		if typ != "model-head" && typ != "model-data" && typ != "model-end" && typ != "model-error" {
			return errors.New("invalid response type")
		}
		return b.model(w, f)
	}
	if !idRE.MatchString(id) {
		return errors.New("invalid request id")
	}
	b.mu.Lock()
	b.lastPing = time.Now()
	b.mu.Unlock()
	switch typ {
	case "hello", "ping":
		if !keys(f, "type", "id") {
			return errors.New("unknown request field")
		}
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		var e error
		if typ == "hello" {
			e = b.checkImage(ctx)
			b.mu.Lock()
			if b.active != nil && b.active.stopping {
				e = errors.New("previous task cleanup is pending")
			}
			b.mu.Unlock()
		}
		if typ == "hello" {
			b.mu.Lock()
			b.probeOK = e == nil
			b.mu.Unlock()
		}
		response(w, id, map[string]any{"mode": "ondemand", "image": b.cfg.Image}, e)
		b.writeStatus()
	case "start":
		if !keys(f, "type", "id", "runId", "role") || !uuidRE.MatchString(str(f, "runId")) || !roles[str(f, "role")] {
			response(w, id, nil, errors.New("invalid run or role"))
			return nil
		}
		v, e := b.start(w, str(f, "runId"), str(f, "role"))
		response(w, id, v, e)
	case "stop":
		if !keys(f, "type", "id", "runId") || !uuidRE.MatchString(str(f, "runId")) {
			response(w, id, nil, errors.New("invalid run"))
			return nil
		}
		e := b.stop(w, str(f, "runId"), true)
		response(w, id, map[string]bool{"stopped": e == nil}, e)
	case "call":
		if !keys(f, "type", "id", "runId", "route", "body") || len(f["body"]) > 2<<20 || !routes[str(f, "route")] {
			response(w, id, nil, errors.New("invalid execution request"))
			return nil
		}
		b.mu.Lock()
		j := b.active
		if j == nil || j.owner != w || j.RunID != str(f, "runId") || j.worker == nil || j.pending != "" {
			b.mu.Unlock()
			response(w, id, nil, errors.New("executor is not ready or is busy"))
			return nil
		}
		j.pending = id
		worker := j.worker
		b.mu.Unlock()
		if e := worker.send(map[string]any{"type": "call", "id": id, "route": str(f, "route"), "body": json.RawMessage(f["body"])}); e != nil {
			response(w, id, nil, e)
			go func() { _ = b.stop(w, j.RunID, true) }()
		}
	default:
		response(w, id, nil, errors.New("unsupported operation"))
	}
	return nil
}
func (b *Broker) handle(c net.Conn) {
	if e := b.authenticate(c); e != nil {
		_ = c.Close()
		return
	}
	w := wire(c)
	b.mu.Lock()
	if b.control != nil || b.closing {
		b.mu.Unlock()
		w.close()
		return
	}
	b.control = w
	b.lastPing = time.Now()
	b.mu.Unlock()
	commands := make(chan Frame, 16)
	done := make(chan struct{})
	go func() {
		defer close(done)
		for f := range commands {
			select {
			case <-w.done:
				return
			default:
			}
			if b.command(w, f) != nil {
				w.close()
				return
			}
		}
	}()
	defer func() {
		w.close()
		close(commands)
		_ = b.stop(w, "", true)
		<-done
		b.mu.Lock()
		if b.control == w {
			b.control = nil
		}
		b.mu.Unlock()
		b.writeStatus()
	}()
	_ = w.read(func(f Frame) error {
		if !validateControl(f) {
			id := str(f, "id")
			if !idRE.MatchString(id) {
				return errors.New("invalid request id")
			}
			response(w, id, nil, errors.New("invalid or oversized operation"))
			return nil
		}
		if strings.HasPrefix(str(f, "type"), "model-") {
			return b.command(w, f)
		}
		if str(f, "type") == "stop" {
			b.mu.Lock()
			if j := b.active; j != nil && j.owner == w && j.RunID == str(f, "runId") {
				j.cancel()
			}
			b.mu.Unlock()
		}
		select {
		case commands <- f:
			return nil
		default:
			return errors.New("too many pending commands")
		}
	})
}
func (b *Broker) watch(ctx context.Context) {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			b.writeStatus()
			b.mu.Lock()
			w, j, last := b.control, b.active, b.lastPing
			b.mu.Unlock()
			if w != nil && time.Since(last) > 45*time.Second {
				w.close()
			}
			if j != nil {
				b.mu.Lock()
				stopping := j.stopping
				b.mu.Unlock()
				if stopping || time.Since(j.started) > 6*time.Hour {
					_ = b.stop(j.owner, j.RunID, true)
				}
			}
			if j == nil {
				b.ops.Lock()
				b.mu.Lock()
				idle := b.active == nil
				b.mu.Unlock()
				if idle {
					if b.removeOwned() == nil {
						_ = b.cleanJobDirs()
					}
				}
				b.ops.Unlock()
			}
		}
	}
}
func main() {
	cfgPath := flag.String("config", "", "root-owned configuration JSON")
	flag.Parse()
	if os.Geteuid() != 0 {
		log.Fatal("host launcher requires root")
	}
	raw, e := strictFile(*cfgPath)
	if e != nil {
		log.Fatal(e)
	}
	var cfg Config
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if e = dec.Decode(&cfg); e != nil {
		log.Fatal(e)
	}
	if e = validateConfig(cfg); e != nil {
		log.Fatal(e)
	}
	if e = os.MkdirAll(cfg.Root, 0700); e != nil {
		log.Fatal(e)
	}
	real, e := filepath.EvalSymlinks(cfg.Root)
	if e != nil || real != cfg.Root {
		log.Fatal("runtime root cannot contain symlinks")
	}
	if e = os.Chmod(cfg.Root, 0700); e != nil {
		log.Fatal(e)
	}
	lock, e := os.OpenFile(filepath.Join(cfg.Root, "broker.lock"), os.O_CREATE|os.O_RDWR, 0600)
	if e != nil {
		log.Fatal(e)
	}
	defer lock.Close()
	if syscall.Flock(int(lock.Fd()), syscall.LOCK_EX|syscall.LOCK_NB) != nil {
		log.Fatal("launcher already running")
	}
	for _, d := range []string{"control", "jobs"} {
		if e = os.MkdirAll(filepath.Join(cfg.Root, d), 0700); e != nil {
			log.Fatal(e)
		}
	}
	b := newBroker(cfg, dockerClient())
	if cfg.Seccomp != "" {
		v, e := strictFile(cfg.Seccomp)
		if e != nil || !json.Valid(v) {
			log.Fatal("invalid execution seccomp profile")
		}
		b.seccomp = string(v)
	}
	if e = b.removeOwned(); e != nil {
		log.Fatal(e)
	}
	if e = b.cleanJobDirs(); e != nil {
		log.Fatal(e)
	}
	sock := filepath.Join(cfg.Root, "control", "control.sock")
	if s, e := os.Lstat(sock); e == nil {
		if s.Mode()&os.ModeSocket == 0 {
			log.Fatal("control endpoint is not a socket")
		}
		_ = os.Remove(sock)
	}
	l, e := net.Listen("unix", sock)
	if e != nil {
		log.Fatal(e)
	}
	defer l.Close()
	_ = os.Chmod(sock, 0600)
	_ = os.WriteFile(filepath.Join(cfg.Root, "broker.pid"), []byte(fmt.Sprint(os.Getpid())), 0600)
	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
	defer cancel()
	go b.watch(ctx)
	go func() {
		<-ctx.Done()
		b.mu.Lock()
		b.closing = true
		w := b.control
		b.mu.Unlock()
		if w != nil {
			w.close()
		}
		_ = l.Close()
	}()
	log.Printf("BenchLocal launcher ready for %s", cfg.Controller)
	for {
		c, e := l.Accept()
		if e != nil {
			break
		}
		go b.handle(c)
	}
	_ = b.stop(nil, "", true)
	_ = os.Remove(sock)
	_ = os.Remove(filepath.Join(cfg.Root, "broker.pid"))
}
