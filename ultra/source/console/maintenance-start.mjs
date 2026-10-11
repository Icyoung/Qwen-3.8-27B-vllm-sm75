import fs from 'node:fs';
import path from 'node:path';
// A bounded, one-use guard for an image update that preserves a stopped model.
export function consumeStoppedModelGuard(root,now=Date.now()){
 const file=path.join(root,'.preserve-stopped-model-once.json');
 let value;
 try{const stat=fs.lstatSync(file);if(!stat.isFile()||stat.isSymbolicLink())throw Error('Invalid maintenance guard');value=JSON.parse(fs.readFileSync(file,'utf8'));fs.unlinkSync(file);}
 catch(error){if(error.code==='ENOENT')return false;console.error('维护启动标志无效，本次保持模型停止。');return true;}
 return Number.isSafeInteger(value.expires)&&value.expires>now&&value.expires-now<=15*60*1000;
}
