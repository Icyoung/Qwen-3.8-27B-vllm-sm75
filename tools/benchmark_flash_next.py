"""Reproduce the 2026-10-03 Flash Next workload; requires an idle server.

Uses the existing token-ID timer. Model/tokenizer and endpoint are supplied by
the caller; this script never starts, stops, or reconfigures the model service.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import re
import statistics
import time
import uuid

import benchmark_v017 as b


def make_inputs(model_path, sizes):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    # Same template, text, and token slicing as the final integration retest.
    prompt = (
        'You are given performance records from a model serving system.\n<reference>PLACEHOLDER</reference>\n'
        'Write a detailed engineering design for a reproducible performance benchmark based on these records. '
        'Cover workload design, cache controls, Prefill and Decode measurement, concurrency, warmup, data validation, '
        'and error handling. Include complete Python code, then a detailed explanation with examples and test cases. '
        'Produce at least 1500 words. Begin directly with your design and continue until the implementation and explanation are complete.'
    )
    template = tok.apply_chat_template(
        [{'role': 'user', 'content': prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    a, z = template.split('PLACEHOLDER')
    pa = tok.encode(a, add_special_tokens=False)
    pz = tok.encode(z, add_special_tokens=False)
    corpus = tok.encode(''.join(
        f'Measurement {i}: {30+i%17} requests completed; median latency {120+i%31} ms; cache reuse {i%70} percent. The run measured memory, response latency, and token throughput.\n'
        for i in range(max(sizes)//12+100)
    ), add_special_tokens=False)
    if min(sizes) <= len(pa) + len(pz):
        raise ValueError('Input budget is shorter than the task template')
    values = {str(n): pa + corpus[:n-len(pa)-len(pz)] + pz for n in sizes}
    assert all(len(v) == int(k) for k, v in values.items())
    special = {tok.convert_tokens_to_ids(t) for t in
               ['<|im_end|>', '<|im_start|>', '<|endoftext|>']}
    return values, special


def check_row(row, special, expect_mtp=True):
    assert row['cached_tokens'] == 0, 'Expected cold prefix cache'
    assert (row['drafted'] > 0 if expect_mtp else row['drafted'] == 0), 'Speculation mode did not match the test'
    assert not any(t in special for e in row['events'] for t in e['ids'])
    row['client_prefill_tok_s'] = row['input_tokens'] / row['ttft_s']
    row['client_decode_tok_s'] = row['output_tokens'] / (row['elapsed_s'] - row['ttft_s'])
    return {k: v for k, v in row.items() if k not in ('events', 'metrics_delta')}


def mixed_stream(ids, limit, special):
    body = dict(model=b.MODEL, prompt=ids, max_tokens=limit, temperature=0, seed=42,
                cache_salt=str(uuid.uuid4()), ignore_eos=False, stream=True,
                return_token_ids=True, stream_options={'include_usage': True})
    events, texts, usage, finish = [], [], None, None
    start = time.perf_counter()
    with b.call('/v1/completions', body) as response:
        for line in response:
            if not line.startswith(b'data: ') or line.strip() == b'data: [DONE]':
                continue
            obj = json.loads(line[6:])
            assert not obj.get('error'), obj.get('error')
            usage = obj.get('usage') or usage
            for c in obj.get('choices', []):
                texts.append(c.get('text', ''))
                finish = c.get('finish_reason') or finish
                if c.get('token_ids'):
                    events.append({'seconds': time.perf_counter()-start, 'ids': c['token_ids']})
    elapsed = time.perf_counter()-start
    token_ids = [t for e in events for t in e['ids']]
    assert usage and usage['prompt_tokens'] == len(ids)
    assert usage['completion_tokens'] == len(token_ids) == limit
    assert usage.get('prompt_tokens_details', {}).get('cached_tokens') == 0
    assert '\ufffd' not in ''.join(texts) and not (set(token_ids) & special)
    assert len(events) > 1
    return dict(input_tokens=len(ids), output_tokens=limit, usage=usage, finish=finish,
                elapsed_s=elapsed, ttft_s=events[0]['seconds'],
                client_prefill_tok_s=len(ids)/events[0]['seconds'],
                decode_tok_s=(limit-len(events[0]['ids']))/(events[-1]['seconds']-events[0]['seconds']),
                events=events)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', required=True, help='Local tokenizer/model directory')
    parser.add_argument('--expect-mtp', choices=['on','off'], default='on', help='Validate MTP3 or true no-MTP startup')
    parser.add_argument('--label', required=True)
    parser.add_argument('--sizes', default='8192,130560,261632')
    parser.add_argument('--repeats', type=int, default=7, help='Formal repetitions for <=8K')
    parser.add_argument('--long-repeats', type=int, default=1)
    parser.add_argument('--output', type=int, default=512)
    parser.add_argument('--warmups', type=int, default=2, help='8K requests with the full output budget')
    parser.add_argument('--mixed', action='store_true', help='Also run cold C4 1K/8K/16K/32K, output256 each')
    args = parser.parse_args()
    if not b.KEY:
        parser.error('Set VLLM_API_KEY')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.label):
        parser.error('Use a label without path separators')
    try:
        sizes = list(dict.fromkeys(int(v) for v in args.sizes.split(',')))
    except ValueError:
        parser.error('--sizes requires comma-separated integers')
    if not sizes or min(sizes) < 1 or min(args.repeats, args.long_repeats, args.output) < 1 or args.warmups < 0:
        parser.error('Invalid length, repetition count, or output budget')
    if max(sizes+[8192]) + args.output > 262144:
        parser.error('Input plus output must fit 262144 tokens')
    result_dir = b.R / 'results'
    result_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ('.jsonl', '-job.json', '-inputs.json'):
        if (result_dir / (args.label+suffix)).exists():
            parser.error('Label already exists; choose a new label')
    result = dict(label=args.label, started=time.time(), stage='inputs', passed=False, cases=[])
    job = result_dir / (args.label+'-job.json')

    def save():
        job.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')

    try:
        save()
        if args.expect_mtp == 'off':
            with b.call('/monitor/spec_decode') as response: state = json.load(response)
            assert state.get('spec_configured') is False, 'Start without speculative-config; hot disabling MTP is a different mode'
            result['speculation_state'] = state
        all_sizes = sorted(set(sizes+[8192]+([1024,16384,32768] if args.mixed else [])))
        values, special = make_inputs(args.model_path, all_sizes)
        hashes = {k: hashlib.sha256(json.dumps(v).encode()).hexdigest() for k, v in values.items()}
        (result_dir/(args.label+'-inputs.json')).write_text(
            json.dumps({'hashes': hashes, 'token_ids': values})+'\n', encoding='utf-8')
        result.update(stage='warmup', input_hashes=hashes)
        save()
        for i in range(args.warmups):
            check_row(b.run(args.label, f'warm8k-{i}', values['8192'], args.output,
                            True, ignore_eos=False), special, args.expect_mtp == 'on')
        for n in sizes:
            result['stage'] = str(n)
            save()
            for _ in range(args.repeats if n <= 8192 else args.long_repeats):
                result['cases'].append(check_row(
                    b.run(args.label, str(n), values[str(n)], args.output, ignore_eos=False), special, args.expect_mtp == 'on'))
                save()
        fields = ('effective_prefill_tok_s','decode_tok_s','ttft_s','mean_acceptance_length',
                  'client_prefill_tok_s','client_decode_tok_s')
        result['medians'] = {str(n): {k: (statistics.median(v) if
            (v := [r[k] for r in result['cases'] if r['input_tokens'] == n and r[k] is not None]) else None)
            for k in fields} for n in sizes}
        if args.mixed:
            result['stage'] = 'mixed4'
            save()
            before = b.metrics()
            assert before['vllm:num_requests_running'] == before['vllm:num_requests_waiting'] == 0
            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=4) as pool:
                rows = list(pool.map(lambda n: mixed_stream(values[str(n)], 256, special),
                                     [1024,8192,16384,32768]))
            elapsed = time.perf_counter()-start
            after = b.metrics()
            assert after['vllm:request_prompt_tokens_count']-before['vllm:request_prompt_tokens_count'] == 4
            assert after['vllm:num_preemptions_total'] == before['vllm:num_preemptions_total']
            drafted = after.get('vllm:spec_decode_num_draft_tokens_total', 0)-before.get('vllm:spec_decode_num_draft_tokens_total', 0)
            assert (drafted > 0 if args.expect_mtp == 'on' else drafted == 0)
            result['mixed4'] = dict(elapsed_s=elapsed, aggregate_e2e_tok_s=1024/elapsed,
                                    requests=rows, preemptions=0)
        result.update(stage='complete', passed=True)
    except Exception as exc:
        result.update(stage='failed', error=str(exc))
        raise
    finally:
        result['finished'] = time.time()
        save()
    print(json.dumps({'passed': True, 'medians': result['medians'],
                      'report': str(job)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
