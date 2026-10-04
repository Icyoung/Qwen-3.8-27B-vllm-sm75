export async function copyApiText(value, host) {
  const document=host.ownerDocument, navigator=document.defaultView.navigator;
  try {if (navigator.clipboard?.writeText) {await navigator.clipboard.writeText(value);return;}} catch {}
  // LAN HTTP pages may not have Clipboard API access. Keep the fallback inside
  // the active dialog so its focus trap does not prevent the copy operation.
  const previous=document.activeElement, input=document.createElement('textarea');
  input.value=value;input.style.cssText='position:fixed;left:-10000px;top:0';host.append(input);
  let copied=false;
  try {input.focus();input.select();copied=document.execCommand('copy');}
  finally {input.value='';input.remove();previous?.focus();}
  if(!copied)throw Error('浏览器未允许复制，请点击显示后手动复制');
}

export function mountApiAccess(host,{api}) {
  const document=host.ownerDocument, element=(tag,text)=>{const node=document.createElement(tag);if(text)node.textContent=text;return node;};
  host.classList.add('api-access-panel');
  let generation=0, info;
  const status=element('p');status.className='help api-access-status';status.setAttribute('role','status');
  const field=(caption,source)=>{
    const label=element('label'),title=element('span',caption),row=element('div'),input=element('input');
    label.className='api-access-field';row.className='api-access-row';input.readOnly=true;input.autocomplete='off';input.spellcheck=false;
    input.dataset.apiValue=source;input.type=source==='url'?'text':'password';
    if(source!=='url')input.placeholder='点击显示或复制';
    row.append(input);label.append(title,row);host.append(label);
    const button=(text,action)=>{const b=element('button',text);b.type='button';b.dataset.apiAction=action;row.append(b);return b;};
    return {label,title,input,show:source==='url'?null:button('显示','show-'+source),copy:button(source==='url'?'复制地址':'复制 Key','copy-'+source)};
  };
  const url=field('API 地址（Base URL）','url'),current=field('API Key','current'),saved=field('待生效 API Key','saved');saved.label.hidden=true;
  const note=element('p','用于 OpenAI 兼容客户端，与管理页面的登录凭据分开。');note.className='help';host.append(note,status);
  const replace=element('details'),summary=element('summary','更换 API Key'),replacement=element('input'),save=element('button','保存新 Key');
  replacement.type='password';replacement.autocomplete='new-password';replacement.placeholder='填写新的 API Key';replacement.setAttribute('aria-label','新的 API Key');save.type='button';save.dataset.apiAction='save';
  const replaceRow=element('div');replaceRow.className='api-access-row';replaceRow.append(replacement,save);replace.append(summary,replaceRow);host.append(replace);
  const clear=()=>{generation++;for(const item of [current,saved]){item.input.value='';item.input.type='password';item.show.textContent='显示';}replacement.value='';};
  const load=async()=>{
    clear();const token=generation;status.textContent='正在读取接入配置…';
    try {
      const data=await api('/console-api/api-access');if(token!==generation)return;
      info=data;
      const address=new URL(document.defaultView.location.href);address.port=String(data.port || 8000);address.pathname='/v1';address.search='';address.hash='';url.input.value=address.href;
      current.title.textContent=data.running?'当前生效 API Key':'API Key';saved.label.hidden=!data.pending;
      status.textContent=data.pending?'新 Key 尚未生效，当前客户端请复制上方的 Key；重启模型后使用待生效 Key。':'';
    } catch(error) {if(token===generation)status.textContent=error.message;throw error;}
  };
  for(const [source,item] of [['current',current],['saved',saved]]) {
    const get=async()=>{const token=generation;const data=await api('/console-api/api-access/reveal',{source});if(token!==generation)return null;if(typeof data.key!=='string'||!data.key)throw Error('未读取到 API Key');return data.key;};
    item.show.onclick=async()=>{
      if(item.input.type==='text'){item.input.value='';item.input.type='password';item.show.textContent='显示';return;}
      item.show.disabled=true;
      try{const value=await get();if(value===null)return;item.input.value=value;item.input.type='text';item.show.textContent='隐藏';}
      catch(error){status.textContent=error.message;}finally{item.show.disabled=false;}
    };
    item.copy.onclick=async()=>{
      item.copy.disabled=true;
      try {const value=await get();if(value===null)return;await copyApiText(value,host);status.textContent=source==='saved'?'待生效 Key 已复制':'API Key 已复制';}
      catch(error){status.textContent=error.message;}finally{item.copy.disabled=false;}
    };
  }
  url.copy.onclick=async()=>{try{await copyApiText(url.input.value,host);status.textContent='API 地址已复制';}catch(error){status.textContent=error.message;}};
  save.onclick=async()=>{
    save.disabled=true;
    try{const result=await api('/console-api/api-access',{key:replacement.value});await load();status.textContent=result.restartRequired?'新 Key 已保存，重启模型后生效；当前 Key 仍可复制使用。':'API Key 已保存。';replace.open=false;}
    catch(error){status.textContent=error.message;}finally{save.disabled=false;}
  };
  return {load,clear};
}
