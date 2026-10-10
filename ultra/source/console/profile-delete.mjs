import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

export function persistProfiles(file, profiles) {
  const temporary=file+'.'+crypto.randomUUID()+'.tmp';
  try {
    fs.writeFileSync(temporary,JSON.stringify(profiles,null,2),{mode:0o600,flag:'wx'});
    fs.renameSync(temporary,file);
  } finally {if(fs.existsSync(temporary))fs.unlinkSync(temporary);}
}

export function deletedBootProfile(profiles,store,boot,env={}) {
  const explicit=boot?.id||env.SM75_DEFAULT_PROFILE;
  const id=explicit||store.get('startup','environment')?.profile;
  if(!id||profiles.some(p=>p.id===id))return null;
  const deleted=store.get('profile-deletion',id);
  return deleted&&(explicit||env.MODEL===deleted.modelPath)?id:null;
}

export async function deleteRuntimeProfile({id,profiles,profilesPath,store,status,locks,write=persistProfiles}) {
  if(!/^[a-z0-9][a-z0-9-]{0,47}$/.test(id||''))throw Error('配置ID无效');
  const p=profiles.find(p=>p.id===id);if(!p)throw Error('配置不存在或已删除');
  if(locks.has(id)||locks.has('model-switch')||locks.has('model-delete'))throw Error('配置操作进行中，请稍后重试');
  locks.add(id);locks.add('model-switch');
  try {
    const current=await status(p);
    if(current?.running||['starting','stopping'].includes(current?.phase))throw Error('该配置正在运行，请先停止模型再删除');
    if(!current||typeof current.running!=='boolean')throw Error('无法确认模型状态，未删除配置');
    const settings=store.settings(),nextSettings={...settings};
    if(settings.defaultProfile===id){nextSettings.defaultProfile='';nextSettings.autoStart=false;}
    if(settings.defaultChatProfile===id)nextSettings.defaultChatProfile='';
    const remaining=profiles.filter(p=>p.id!==id),deletedAt=Date.now();
    const directory=path.join(path.dirname(profilesPath),'deleted-profiles');
    fs.mkdirSync(directory,{recursive:true,mode:0o700});
    const backup=id+'-'+crypto.randomUUID()+'.json';
    fs.writeFileSync(path.join(directory,backup),JSON.stringify({schema:'sm75-profile-v1',profile:p,deletedAt},null,2),{mode:0o600,flag:'wx'});
    let wrote=false;
    store.db.exec('BEGIN IMMEDIATE');
    try {
      store.put('profile-deletion',id,{id,name:p.name,modelPath:p.args[0],backup,deletedAt});
      store.put('settings','main',nextSettings);
      write(profilesPath,remaining);wrote=true;
      store.db.exec('COMMIT');
    } catch(error) {
      try {store.db.exec('ROLLBACK');} catch {}
      if(wrote)persistProfiles(profilesPath,profiles);
      throw error;
    }
    profiles.splice(0,profiles.length,...remaining);
    return {profiles:remaining,settings:nextSettings,deleted:id};
  } finally {locks.delete(id);locks.delete('model-switch');}
}
