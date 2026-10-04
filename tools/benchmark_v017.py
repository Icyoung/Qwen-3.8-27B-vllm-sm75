"""Cold-KV Prefill and token-ID-checked speculative Decode benchmark."""
import argparse,hashlib,json,pathlib,statistics,time,urllib.request,uuid,os,re
R=pathlib.Path(os.getenv('VLLM_BENCH_ROOT','./benchmark-results'))
KEY=os.getenv('VLLM_API_KEY','')
BASE=os.getenv('VLLM_BASE_URL','http://127.0.0.1:8000').rstrip('/')
MODEL=os.getenv('VLLM_MODEL','Flash-Next-TP8')

def call(route,body=None,timeout=600):
    return urllib.request.urlopen(urllib.request.Request(BASE+route,data=None if body is None else json.dumps(body).encode(),
        headers={'Content-Type':'application/json','Authorization':'Bearer '+KEY,'x-api-key-hash':hashlib.sha256(KEY.encode()).hexdigest()}),timeout=timeout)
def metrics():
    with call('/metrics',timeout=10) as response:lines=response.read().decode().splitlines()
    out={}
    for line in lines:
        if not line.startswith('vllm:'):continue
        name=line.split('{',1)[0].split(' ',1)[0]
        if name.endswith(('_sum','_count','_total')) or name in ('vllm:num_requests_running','vllm:num_requests_waiting'):
            out[name]=out.get(name,0)+float(line.rsplit(' ',1)[1])
    return out
def run(label,case,ids,output,warmup=False,salt=None,ignore_eos=True):
    before=metrics();assert before['vllm:num_requests_running']==before['vllm:num_requests_waiting']==0,'Other request active'
    body={'model':MODEL,'prompt':ids,'max_tokens':output,'temperature':0,'seed':42,'ignore_eos':ignore_eos,
          'cache_salt':salt or str(uuid.uuid4()),'stream':True,'return_token_ids':True,'stream_options':{'include_usage':True}}
    events=[];texts=[];usage=None;finish=None;start=time.perf_counter()
    with call('/v1/completions',body) as response:
        for line in response:
            if not line.startswith(b'data: ') or line.strip()==b'data: [DONE]':continue
            obj=json.loads(line[6:]);assert not obj.get('error'),obj.get('error')
            usage=obj.get('usage') or usage
            for choice in obj.get('choices',[]):
                if choice.get('token_ids'):events.append({'seconds':time.perf_counter()-start,'ids':choice['token_ids']})
                texts.append(choice.get('text',''));finish=choice.get('finish_reason') or finish
    elapsed=time.perf_counter()-start;after=metrics();delta={k:after[k]-before.get(k,0) for k in after}
    tokens=[t for e in events for t in e['ids']]
    assert usage and len(tokens)==usage['completion_tokens']==output and usage['prompt_tokens']==len(ids),(case,usage,output,finish)
    assert delta['vllm:request_prompt_tokens_count']==1 and delta['vllm:request_prefill_time_seconds_count']==1
    prefill=delta['vllm:request_prefill_time_seconds_sum'];first_count=len(events[0]['ids'])
    assert prefill>0 and len(events)>1
    drafted=delta.get('vllm:spec_decode_num_draft_tokens_total',0);accepted=delta.get('vllm:spec_decode_num_accepted_tokens_total',0)
    drafts=delta.get('vllm:spec_decode_num_drafts_total',0)
    row={'variant':label,'case':case,'warmup':warmup,'input_tokens':len(ids),'output_tokens':output,
         'ignore_eos':ignore_eos,'input_hash':hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
         'cached_tokens':usage.get('prompt_tokens_details',{}).get('cached_tokens'),
         'ttft_s':events[0]['seconds'],'prefill_s':prefill,'effective_prefill_tok_s':len(ids)/prefill,
         'decode_tok_s':(output-first_count)/(events[-1]['seconds']-events[0]['seconds']),
         'first_event_tokens':first_count,'max_tokens_per_event':max(len(e['ids']) for e in events),
         'elapsed_s':elapsed,'finish':finish,'replacement_count':''.join(texts).count('\ufffd'),
         'token_hash':hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
         'accepted':accepted,'drafted':drafted,'draft_steps':drafts,'acceptance_rate':accepted/drafted if drafted else None,
         'mean_acceptance_length':1+accepted/drafts if drafts else None,
         'preemptions':delta.get('vllm:num_preemptions_total',0),'events':events,'metrics_delta':delta}
    with (R/'results'/f'{label}.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in row.items() if k not in ('events','metrics_delta')},ensure_ascii=False),flush=True)
    assert row['replacement_count']==0 and row['preemptions']==0
    return row
def inputs():
    path=R/'results/input-token-ids.json'
    if path.exists():return json.loads(path.read_text())
    text=''.join(f'Record {i}: The laboratory compares reproducible measurements of memory, latency, and throughput.\n' for i in range(1800))
    with call('/tokenize',{'model':MODEL,'prompt':text,'add_special_tokens':False}) as response:long=json.load(response)['tokens']
    values={'4k':long[:4096],'8k':long[:8192]}
    for name,prompt in [('chinese','请详细设计一个图书馆借阅管理系统的数据结构和处理流程。需要包含借阅、归还、超期处理、图书预订和库存查询，解释并发操作如何保持一致性，并给出详细的Python伪代码和测试用例。回答至少1000字。'),
                        ('code','Write a detailed Python implementation of an LRU cache with O(1) get and put. Include the complete code, tests for eviction and updates, and explain each design decision. Continue with thread safety and memory management considerations. Write at least 700 words.')]:
        with call('/tokenize',{'model':MODEL,'messages':[{'role':'user','content':prompt}],'add_generation_prompt':True,'chat_template_kwargs':{'enable_thinking':False}}) as response:values[name]=json.load(response)['tokens']
    path.write_text(json.dumps(values)+'\n');return values
def main():
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--repeats',type=int,default=2);p.add_argument('--output',type=int,default=512);p.add_argument('--cases',default='4k,8k,chinese,code');args=p.parse_args()
    if not KEY:raise ValueError('Set VLLM_API_KEY')
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.label):raise ValueError('Use a simple label without path separators')
    (R/'results').mkdir(parents=True,exist_ok=True)
    assert not (R/'results'/f'{args.label}.jsonl').exists(),'Do not overwrite or append a completed variant'
    data=inputs();run(args.label,'warmup',data['4k'],32,True)
    rows=[]
    for case in args.cases.split(','):
        if case!='4k':run(args.label,'warmup-'+case,data[case],32,True)
        for i in range(args.repeats):rows.append(run(args.label,case,data[case],args.output))
    summary={case:{k:statistics.median(v) if (v:=[r[k] for r in rows if r['case']==case and r[k] is not None]) else None
        for k in ('effective_prefill_tok_s','decode_tok_s','ttft_s','acceptance_rate','mean_acceptance_length','cached_tokens')}
        for case in args.cases.split(',')}
    (R/'results'/f'{args.label}-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('SUMMARY',json.dumps(summary),flush=True)
if __name__=='__main__':main()
