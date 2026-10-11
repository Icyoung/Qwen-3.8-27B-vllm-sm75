import net from 'node:net';
import {EventEmitter} from 'node:events';
export const MAX_FRAME=24*1024*1024;
export class JsonChannel extends EventEmitter {
 constructor(socket){super();this.socket=socket;this.pending=0;this.buffer=Buffer.alloc(0);this.reason='连接已关闭';this.closed=false;
  socket.on('data',part=>{this.buffer=Buffer.concat([this.buffer,part]);if(this.buffer.length>MAX_FRAME){this.close('消息过大');return;}let at;while((at=this.buffer.indexOf(10))>=0){const line=this.buffer.subarray(0,at);this.buffer=this.buffer.subarray(at+1);try{const value=JSON.parse(line);if(!value||Array.isArray(value)||typeof value!=='object')throw Error();this.emit('message',value);}catch{this.close('无效消息');return;}}});
  socket.on('error',e=>{this.reason=e.message;});socket.on('close',()=>{this.closed=true;this.emit('close',this.reason);});
 }
 send(value){let data;try{data=Buffer.from(JSON.stringify(value)+'\n');}catch(e){return Promise.reject(e);}if(this.closed||this.socket.destroyed)return Promise.reject(Error(this.reason));
  if(data.length>MAX_FRAME||this.pending+data.length>MAX_FRAME*2){this.close('发送队列超限');return Promise.reject(Error(this.reason));}
  this.pending+=data.length;return new Promise((resolve,reject)=>{this.socket.write(data,e=>{this.pending-=data.length;e?reject(e):resolve();});});
 }
 close(reason){if(reason)this.reason=reason;this.socket.destroy();}
}
export async function connectChannel(path,{timeout=5000}={}){
 const socket=net.createConnection(path),channel=new JsonChannel(socket);
 await new Promise((resolve,reject)=>{const timer=setTimeout(()=>{channel.close('连接启动器超时');reject(Error(channel.reason));},timeout);const fail=()=>{clearTimeout(timer);reject(Error(channel.reason));};socket.once('close',fail);socket.once('connect',()=>{clearTimeout(timer);socket.removeListener('close',fail);resolve();});});return channel;
}
