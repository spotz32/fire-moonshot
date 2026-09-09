"""Embedded HTML for the FireTrack Results view (served at /results).

A shared timeline for the 2D detection player and 3D reconstructed trajectory.
"""

RESULTS_HTML = b"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>FireTrack \xc2\xb7 Results</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@600;700&family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{--bg:#0B0F14;--panel:#141B24;--raised:#1B2531;--line:#263241;--line-soft:#1d2733;
 --text:#E8EEF4;--muted:#8895A7;--faint:#5C6B7E;--ember:#FF6A2B;--cyan:#36C5D9;--green:#46D17F;
 --disp:'Chakra Petch',system-ui,sans-serif;--mono:'IBM Plex Mono',ui-monospace,monospace;--body:'Inter',system-ui,sans-serif;}
*{box-sizing:border-box}html,body{height:100%}body{margin:0;background:var(--panel);color:var(--text);font-family:var(--body);font-size:14px;overflow:hidden}
a{color:var(--cyan);text-decoration:none}
.wrap{height:100vh;width:100%;margin:0;padding:12px;display:grid;grid-template-columns:1.25fr .75fr;grid-template-rows:auto minmax(0,1fr) auto;gap:12px}
.runbar{grid-column:1/-1;background:transparent;border-bottom:1px solid var(--line-soft);padding:10px 2px 12px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.runbar h2{font-family:var(--disp);font-weight:600;font-size:14px;letter-spacing:.06em;margin:0;text-transform:uppercase}
.runbar label{margin-left:auto;font-family:var(--mono);font-size:10px;color:var(--faint);letter-spacing:.08em;text-transform:uppercase;display:flex;align-items:center;gap:8px}
.nomocap .runbar label{display:none}
#selected-run{display:none;min-width:0;overflow-wrap:anywhere;color:var(--muted);font-size:12px}
.nomocap #selected-run{display:block}
.nomocap.embedded .runbar{display:none}
.nomocap.embedded .wrap{grid-template-rows:minmax(0,1fr) auto}
.nomocap:not(.editing) #camera-picker{display:none}
.nomocap:not(.editing) .detpane.active h3{color:var(--muted)}
@media (max-width:920px){body{overflow:auto}.wrap{height:auto;min-height:100vh;grid-template-columns:1fr;grid-template-rows:auto auto auto}.detgrid{grid-template-columns:1fr}}
@media (max-width:920px){.nomocap.embedded .wrap{grid-template-rows:auto auto auto}}
.pane{min-height:0;overflow:hidden;display:flex;flex-direction:column}
.pane .hd{padding:0 0 10px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.pane .hd h2{font-family:var(--disp);font-weight:600;font-size:13px;letter-spacing:.06em;margin:0;text-transform:uppercase}
.pane .bd{min-height:0;padding:0;display:flex;flex-direction:column;flex:1}
select{font-family:var(--mono);font-size:12px;background:var(--raised);color:var(--text);border:1px solid var(--line);border-radius:7px;padding:6px 9px;max-width:100%}
.pick{margin-left:auto;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.pick label{font-family:var(--mono);font-size:10px;color:var(--faint);letter-spacing:.08em;text-transform:uppercase;display:flex;align-items:center;gap:6px}
.stage2d{position:relative;flex:1;min-height:0;background:#06090d;border:1px solid var(--line);border-radius:9px;overflow:hidden;display:flex;align-items:center;justify-content:center}
.detgrid{width:100%;height:100%;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;padding:8px;align-items:center}
.detpane{min-width:0;height:100%;display:flex;flex-direction:column;gap:6px}
.detpane h3{font-family:var(--mono);font-size:11px;color:var(--muted);font-weight:500;margin:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.detpane.active h3{color:var(--cyan)}
.fwrap{position:relative;display:flex;align-items:center;justify-content:center;min-height:0;flex:1;line-height:0;background:#030507;border:1px solid var(--line-soft);border-radius:7px;overflow:hidden}
.fwrap img{display:block;max-width:100%;max-height:100%;width:auto;height:auto}
.fwrap canvas{position:absolute}
@media (max-width:920px){.detgrid{grid-template-columns:1fr}}
.scene3d{flex:1;min-height:0;width:100%;height:auto;background:#06090d;border:1px solid var(--line);border-radius:9px}
.controls{display:flex;align-items:center;gap:10px;margin-top:8px}
.playback{grid-column:1/-1;min-width:0;border-top:1px solid var(--line-soft);padding-top:10px;margin:0}
.playback input{min-width:40px}
.playback .meta{margin:0;white-space:nowrap;font-size:11px}
#play{width:42px;height:38px;flex-shrink:0;padding:0;font-size:18px}
button{font-family:var(--disp);font-weight:600;font-size:13px;color:var(--text);background:var(--raised);border:1px solid var(--line);border-radius:8px;padding:8px 13px;cursor:pointer}
button:hover{border-color:var(--cyan)}
button.primary{background:var(--ember);border-color:var(--ember);color:#1a0d05}
button.on{background:var(--cyan);border-color:var(--cyan);color:#06121a}
input[type=range]{flex:1;accent-color:var(--ember)}
.meta{font-family:var(--mono);font-size:12px;color:var(--muted);margin-top:8px;display:flex;gap:16px;flex-wrap:wrap}
.meta b{color:var(--text);font-weight:500}
.legend{display:flex;gap:14px;font-family:var(--mono);font-size:11px;color:var(--muted);margin-top:8px;flex-wrap:wrap}
.legend i{display:inline-block;width:14px;height:3px;border-radius:2px;margin-right:5px;vertical-align:middle}
.empty{color:var(--muted);text-align:center;padding:40px 10px;font-family:var(--mono);font-size:13px}
.hint{color:var(--faint);font-size:11px;font-family:var(--mono);margin-top:6px}
:focus-visible{outline:2px solid var(--cyan);outline-offset:2px}
body.nomocap{
 --bg:#000000;--panel:#171717;--raised:#262626;--line:#484848;--line-soft:#333333;
 --text:#FAFAFA;--muted:#BDBDBD;--faint:#A3A3A3;--ember:#CFB991;--cyan:#CFB991;
 --current-point:#FFFFFF;--green:#FFFFFF;
}
.nomocap .stage2d,.nomocap .scene3d,.nomocap .fwrap{background:#000000}
.nomocap button.primary,.nomocap button.on{color:#000000}
#show-reference{display:none;accent-color:var(--green);margin:0 4px 0 0;vertical-align:middle}
.nomocap #show-reference{display:inline-block}
#reference-info{font-size:11px;line-height:1.5;overflow-wrap:anywhere}
#reference-info:empty{display:none}
</style></head>
<body>
<div class="wrap">

  <section class="runbar">
    <h2>Selected Run</h2>
    <span id="selected-run"></span>
    <label>Run <select id="runselect"></select></label>
  </section>

  <section class="pane">
    <div class="hd"><h2>Detection \xc2\xb7 2D</h2><div class="pick" id="camera-picker"><label>Active camera <select id="detsel"></select></label></div></div>
    <div class="bd">
      <div class="stage2d" id="detstage"><div class="empty" id="detempty">Run detection, then pick a run.</div>
        <div class="detgrid" id="detgrid" style="display:none"></div></div>
      <div class="controls" id="detedit" style="display:none">
        <button id="editbtn" onclick="toggleEdit()">Edit track</button>
        <button id="clearframe" onclick="clearFrame()" style="display:none">Clear frame</button>
        <button id="savedits" class="primary" onclick="saveEdits()" style="display:none">Save</button>
        <span class="meta" id="editinfo"></span>
      </div>
      <div class="meta" id="detmeta"></div>
    </div>
  </section>

  <section class="pane">
    <div class="hd"><h2>Reconstruction \xc2\xb7 3D</h2></div>
    <div class="bd">
      <canvas class="scene3d" id="scene" width="520" height="360"></canvas>
      <div class="legend">
        <span><i style="background:var(--cyan)"></i>smoothed</span>
        <span><i style="background:var(--faint)"></i>raw</span>
        <label><input type="checkbox" id="show-reference" checked aria-label="Show flight-log reference"><i style="background:var(--green)"></i><span id="reference-label">ground truth</span></label>
        <span><i style="background:var(--current-point,var(--ember))"></i>current frame</span>
      </div>
      <div class="meta" id="trimeta"></div>
      <div class="meta" id="reference-info"></div>
      <div class="controls" id="tridl" style="display:none">
        <button onclick="downloadTraj('csv')">Download CSV</button>
        <button onclick="downloadTraj('npz')">Download .npz</button>
        <button id="comparison-download" onclick="downloadTraj('comparison')" style="display:none">Comparison CSV</button>
      </div>
      <div class="hint">Drag to orbit \xc2\xb7 scroll to zoom</div>
    </div>
  </section>

  <div class="controls playback" id="playback" style="display:none">
    <button id="play" title="Play" aria-label="Play">&#9654;</button>
    <input type="range" id="scrub" min="0" value="0" step="1" aria-label="Playback position">
    <span class="meta" id="playposition"></span>
  </div>
</div>
<script>
const resultParams=new URLSearchParams(window.location.search);
const noMocap=resultParams.get('mode')==='nomocap';
document.body.classList.toggle('nomocap',noMocap);
document.body.classList.toggle('embedded',noMocap&&resultParams.get('embedded')==='1');
if(noMocap){document.getElementById('reference-label').textContent='flight-log reference';document.getElementById('show-reference').disabled=true;}
async function getResults(){return (await fetch('/api/results')).json();}
function opt(v,t){const o=document.createElement('option');o.value=v;o.textContent=t;return o;}

/* ---------- Detection 2D ---------- */
let det={cams:[],dir:'',frame:0,n_frames:0,editing:false,edits:{}};
let runVersion=0;
async function loadOneDet(dir){
 return await (await fetch('/api/centroids?dir='+encodeURIComponent(dir))).json();
}
async function loadDet(dir){
 stopPlayback();
 const item=(window.allDetections||[]).find(d=>d.dir===dir);
 if(!item)return;
 await loadDetRun((window.allDetections||[]).filter(d=>(d.run||'')===(item.run||'')),dir);
}
async function loadDetRun(items,activeDir,version=runVersion){
 stopPlayback();
 const cams=await Promise.all(items.map(async item=>({dir:item.dir,label:item.label.split('/').pop().trim(),data:await loadOneDet(item.dir),frame:0})));
 if(version!==runVersion)return;
 det.cams=[];
 det.dir=activeDir||(items[0]&&items[0].dir)||'';
 det.frame=0;det.edits={};det.editing=false;
 det.cams=cams;
 det.n_frames=det.cams.length?Math.min(...det.cams.map(c=>c.data.n_frames)):0;
 document.getElementById('detempty').style.display='none';
 const grid=document.getElementById('detgrid');
 grid.style.display='grid';
 grid.innerHTML=det.cams.map(c=>`<div class="detpane ${c.dir===det.dir?'active':''}" data-dir="${c.dir}">
   <h3>${c.label}</h3><div class="fwrap"><img alt=""><canvas></canvas></div>
 </div>`).join('');
 grid.querySelectorAll('.detpane').forEach(p=>p.onclick=()=>{if(!noMocap||det.editing)setActiveDet(p.dataset.dir);});
 document.getElementById('detedit').style.display='flex';
 document.getElementById('editbtn').textContent='Edit track';document.getElementById('editbtn').classList.remove('on');
 document.getElementById('clearframe').style.display='none';
 document.getElementById('savedits').style.display='none';
 document.getElementById('editinfo').textContent='';
 document.getElementById('detmeta').innerHTML=det.cams.map(c=>{
   const m=c.data;
   return `<span>${c.label} <b>${m.n_detected}/${m.n_frames}</b> (${(100*m.n_detected/m.n_frames||0).toFixed(0)}%)</span>`;
 }).join('');
}
function activeCam(){return det.cams.find(c=>c.dir===det.dir)||det.cams[0];}
function setActiveDet(dir,keepEdits=false){
 const same=det.dir===dir;
 det.dir=dir;
 const sel=document.getElementById('detsel'); if(sel&&sel.value!==dir)sel.value=dir;
 document.querySelectorAll('.detpane').forEach(p=>p.classList.toggle('active',p.dataset.dir===dir));
 if(!keepEdits&&!same)det.edits={};
 updateEditInfo();drawAllOverlays();
}
function drawPaneOverlay(pane,frameOverride){
 const cam=det.cams.find(c=>c.dir===pane.dataset.dir); if(!cam)return;
 const img=pane.querySelector('img'),canvas=pane.querySelector('canvas');
 const w=img.clientWidth,h=img.clientHeight;canvas.width=w;canvas.height=h;
 canvas.style.width=w+'px';canvas.style.height=h+'px';
 canvas.style.left=img.offsetLeft+'px';canvas.style.top=img.offsetTop+'px';
 const ctx=canvas.getContext('2d');ctx.clearRect(0,0,w,h);
 const frame=(frameOverride==null?cam.frame:frameOverride);
 const c=cam.data.centroids[frame],edited=cam.dir===det.dir&&det.edits&&(frame in det.edits);
 if(!c||c[0]==null){ctx.fillStyle='#FFC857';ctx.font='12px monospace';
   ctx.fillText(edited?'cleared (edited)':'no detection this frame',10,18);return;}
 const sx=w/cam.data.width,sy=h/cam.data.height,x=c[0]*sx,y=c[1]*sy;
 ctx.strokeStyle=noMocap?(edited?'#FFFFFF':(det.editing&&cam.dir===det.dir?'#DAAA00':'#CFB991')):
   (edited?'#FF6A2B':(cam.dir===det.dir?'#36C5D9':'#46D17F'));ctx.lineWidth=2;
 ctx.beginPath();ctx.arc(x,y,11,0,7);ctx.stroke();
 ctx.beginPath();ctx.moveTo(x-16,y);ctx.lineTo(x+16,y);ctx.moveTo(x,y-16);ctx.lineTo(x,y+16);ctx.stroke();
}
function drawAllOverlays(){document.querySelectorAll('.detpane').forEach(p=>drawPaneOverlay(p));}
function updateEditInfo(){const n=Object.keys(det.edits||{}).length,info=document.getElementById('editinfo');
 info.textContent=!det.editing?'':(n?`${n} unsaved edit(s)`:'Click the drone to set it \\u00b7 Clear = no detection');
 document.getElementById('savedits').style.display=n?'inline-flex':'none';}
function toggleEdit(){det.editing=!det.editing;stopPlayback();
 if(noMocap)document.body.classList.toggle('editing',det.editing);
 const b=document.getElementById('editbtn');b.textContent=det.editing?'Done':'Edit track';b.classList.toggle('on',det.editing);
 document.getElementById('clearframe').style.display=det.editing?'inline-flex':'none';
 document.querySelectorAll('.fwrap canvas').forEach(c=>c.style.cursor=det.editing?'crosshair':'');updateEditInfo();
 if(noMocap)drawAllOverlays();}
function clearFrame(){const cam=activeCam();if(!det.editing||!cam)return;
 cam.data.centroids[cam.frame]=[null,null];det.edits[cam.frame]={clear:true};drawAllOverlays();updateEditInfo();}
async function saveEdits(){
 const edits=Object.entries(det.edits).map(([f,v])=>v.clear?{frame:+f,clear:true}:{frame:+f,x:v.x,y:v.y});
 if(!edits.length)return;
 const r=await fetch('/api/centroids/edit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({dir:det.dir,edits})});
 const d=await r.json().catch(()=>({}));const info=document.getElementById('editinfo');
 if(!r.ok){info.textContent='Save failed: '+(d.error||'error');return;}
 det.edits={};
 const cam=activeCam();
 if(cam&&d.n_detected!=null){cam.data.n_detected=d.n_detected;}
 document.getElementById('detmeta').innerHTML=det.cams.map(c=>{
   const m=c.data;return `<span>${c.label} <b>${m.n_detected}/${m.n_frames}</b> (${(100*m.n_detected/m.n_frames||0).toFixed(0)}%)</span>`;
 }).join('');
 info.textContent='Saved \\u2014 re-run Triangulate to update 3D.';document.getElementById('savedits').style.display='none';
}
document.getElementById('detgrid').addEventListener('click',e=>{
 if(!det.editing)return;
 const canvas=e.target.closest('canvas'); if(!canvas)return;
 const pane=canvas.closest('.detpane'); setActiveDet(pane.dataset.dir,true);
 const cam=activeCam(); if(!cam)return;
 const r=canvas.getBoundingClientRect();
 const x=(e.clientX-r.left)*cam.data.width/r.width,y=(e.clientY-r.top)*cam.data.height/r.height;
 cam.data.centroids[cam.frame]=[x,y];det.edits[cam.frame]={x,y};drawAllOverlays();updateEditInfo();
});
function clearDet(){
 stopPlayback();det={cams:[],dir:'',frame:0,n_frames:0,editing:false,edits:{}};
 if(noMocap)document.body.classList.remove('editing');
 document.getElementById('detempty').style.display='block';
 document.getElementById('detgrid').style.display='none';
 document.getElementById('detgrid').innerHTML='';
 document.getElementById('detedit').style.display='none';
 document.getElementById('detmeta').innerHTML='';
}
window.addEventListener('resize',()=>{if(det.cams.length)drawAllOverlays();});

/* ---------- Reconstruction 3D ---------- */
let tri={data:null,yaw:0.6,pitch:-0.5,zoom:1,marker:0};
const cv=document.getElementById('scene');
function resizeScene(){
 const r=cv.getBoundingClientRect();
 const w=Math.max(320,Math.round(r.width)),h=Math.max(260,Math.round(r.height));
 if(cv.width!==w||cv.height!==h){cv.width=w;cv.height=h;}
}
function finitePts(arrs){const o=[];for(const a of arrs)if(a)for(const p of a)if(p&&p[0]!=null)o.push(p);return o;}
function setupView(){
 const pts=finitePts([tri.data.smooth,tri.data.raw,tri.data.gt,visibleReference()]);
 if(!pts.length){tri.center=[0,0,0];tri.extent=1;return;}
 const lo=[1e9,1e9,1e9],hi=[-1e9,-1e9,-1e9];
 for(const p of pts)for(let k=0;k<3;k++){lo[k]=Math.min(lo[k],p[k]);hi[k]=Math.max(hi[k],p[k]);}
 tri.center=[0,1,2].map(k=>(lo[k]+hi[k])/2);
 tri.extent=Math.max(hi[0]-lo[0],hi[1]-lo[1],hi[2]-lo[2],1e-6);
}
function project(p){
 const cy=Math.cos(tri.yaw),sy=Math.sin(tri.yaw),cp=Math.cos(tri.pitch),sp=Math.sin(tri.pitch);
 const x=p[0]-tri.center[0],y=p[1]-tri.center[1],z=p[2]-tri.center[2];
 const x1=cy*x+sy*z,z1=-sy*x+cy*z,y1=y;
 const y2=cp*y1-sp*z1;
 const s=tri.zoom*0.7*Math.min(cv.width,cv.height)/tri.extent;
 return [cv.width/2+x1*s, cv.height/2-y2*s];
}
function poly(ctx,arr,color,dash){
 ctx.strokeStyle=color;ctx.lineWidth=2;ctx.setLineDash(dash||[]);ctx.beginPath();
 let pen=false;
 for(const p of arr){if(!p||p[0]==null){pen=false;continue;}const s=project(p);if(!pen){ctx.moveTo(s[0],s[1]);pen=true;}else ctx.lineTo(s[0],s[1]);}
 ctx.stroke();ctx.setLineDash([]);
}
function axes(ctx){
 const L=tri.extent*0.45,o=tri.center;const cols=noMocap?['#A3A3A3','#DCDCDC','#CFB991']:['#c0566f','#6fae6f','#5aa0c0'];const lbl=['x','y','z'];
 for(let k=0;k<3;k++){const e=o.slice();e[k]+=L;const a=project(o),b=project(e);
   ctx.strokeStyle=cols[k];ctx.lineWidth=1;ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();ctx.setLineDash([]);
   ctx.fillStyle=cols[k];ctx.font='11px monospace';ctx.fillText(lbl[k],b[0]+3,b[1]);}
}
function drawScene(){
 resizeScene();
 const ctx=cv.getContext('2d');ctx.clearRect(0,0,cv.width,cv.height);
 if(!tri.data){return;}
 axes(ctx);
 const raw=tri.data.raw;ctx.fillStyle=noMocap?'#A3A3A3':'#5C6B7E';
 for(const p of raw){if(p&&p[0]!=null){const s=project(p);ctx.fillRect(s[0]-1,s[1]-1,2,2);}}
 if(tri.data.gt)poly(ctx,tri.data.gt,'#46D17F',[5,4]);
 const reference=visibleReference();
 if(reference)poly(ctx,reference,noMocap?'#FFFFFF':'#46D17F',[5,4]);
 poly(ctx,tri.data.smooth,noMocap?'#CFB991':'#36C5D9');
 const m=tri.data.smooth[tri.marker]||tri.data.raw[tri.marker];
 if(m&&m[0]!=null){const s=project(m);ctx.fillStyle=noMocap?'#FFFFFF':'#FF6A2B';ctx.beginPath();ctx.arc(s[0],s[1],5,0,7);ctx.fill();}
 const referencePoint=reference?.[tri.marker];
 if(referencePoint&&referencePoint[0]!=null){const s=project(referencePoint);ctx.strokeStyle=noMocap?'#FFFFFF':'#46D17F';ctx.lineWidth=3;ctx.beginPath();ctx.arc(s[0],s[1],7,0,7);ctx.stroke();}
}
function visibleReference(){return noMocap&&document.getElementById('show-reference').checked?tri.data?.reference?.points:null;}
document.getElementById('show-reference').onchange=()=>{if(tri.data){setupView();drawScene();}};
function downloadTraj(fmt){if(tri.dir)window.location.href='/api/trajectory/download?dir='+encodeURIComponent(tri.dir)+'&fmt='+fmt;}
async function loadTri(dir,version=runVersion){
 stopPlayback();
 const data=await (await fetch('/api/trajectory?dir='+encodeURIComponent(dir))).json();
 if(version!==runVersion)return;
 tri.dir=dir;
 tri.data=data;
 tri.marker=0;setupView();drawScene();
 document.getElementById('tridl').style.display='flex';
 const mt=tri.data.metrics,classification=tri.data.classification;document.getElementById('trimeta').innerHTML=
   `<span>triangulated <b>${mt.n_triangulated}</b>/${tri.data.n_frames}</span>`+
   (mt.median_reproj_px!=null?`<span>reproj <b>${mt.median_reproj_px.toFixed(1)}px</b></span>`:'')+
   (mt.has_gt&&mt.rmse_smooth_m!=null?`<span>RMSE vs GT <b>${mt.rmse_smooth_m.toFixed(3)}m</b></span>`:(noMocap?'':'<span>no ground truth</span>'))+
   (classification?`<span title="${classification.reason||''}">firmware <b>${classification.prediction}</b></span>`:'');
 const reference=noMocap?data.reference:null;
 const info=document.getElementById('reference-info');info.textContent='';
 document.getElementById('comparison-download').style.display=reference?.points?'inline-flex':'none';
 if(noMocap){
   document.getElementById('reference-label').textContent='flight-log reference';
   document.getElementById('show-reference').disabled=!reference?.points;
   if(reference?.points){
     const metrics=reference.metrics;
     const error=metrics.rmse_m==null?'No comparable reconstructed positions':`RMSE vs reference: ${metrics.rmse_m.toFixed(2)} m; median: ${metrics.median_error_m.toFixed(2)} m`;
     info.textContent=`${reference.log}: ${metrics.n_reference}/${data.n_frames} frames covered; ${metrics.n_compared} compared. ${error}. `+(reference.warnings||[]).join(' ');
     info.title=`${reference.position_source}. ${reference.time_basis}. Horizontal RMSE: ${metrics.horizontal_rmse_m?.toFixed(3)??'n/a'} m. Vertical RMSE: ${metrics.vertical_rmse_m?.toFixed(3)??'n/a'} m.`;
   }else{info.textContent=reference?.error||'No flight-log reference loaded.';}
 }
}
let drag=null;
cv.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];cv.setPointerCapture(e.pointerId);});
cv.addEventListener('pointermove',e=>{if(!drag)return;tri.yaw+=(e.clientX-drag[0])*0.01;tri.pitch+=(e.clientY-drag[1])*0.01;
 tri.pitch=Math.max(-1.5,Math.min(1.5,tri.pitch));drag=[e.clientX,e.clientY];drawScene();});
cv.addEventListener('pointerup',()=>drag=null);
cv.addEventListener('wheel',e=>{e.preventDefault();tri.zoom*=e.deltaY<0?1.1:0.9;tri.zoom=Math.max(0.2,Math.min(6,tri.zoom));drawScene();},{passive:false});
window.addEventListener('resize',drawScene);

/* ---------- Shared playback ---------- */
const playback={frame:0,count:0,fps:30,start:0,playing:false,timer:null,request:0};
function stopPlayback(){
 playback.playing=false;clearTimeout(playback.timer);playback.timer=null;
 playback.request++;
 const b=document.getElementById('play');b.innerHTML='&#9654;';b.title='Play';b.setAttribute('aria-label','Play');
}
function cameraStart(cam){
 const key=cam.dir.split('/').pop(),data=tri.data;
 const saved=data?.camera_start_epochs_s?.[key];
 if(Number.isFinite(saved))return saved;
 const start=cam.data.start_epoch_s;
 if(Number.isFinite(start))return start+(data?.camera_time_offsets_s?.[key]||0);
 return data?.epoch_times_s?.[0]||0;
}
function configurePlayback(){
 stopPlayback();
 const times=tri.data?.epoch_times_s;
 const stride=tri.data?.stride||1;
 const interval=times?.length>1?(times[1]-times[0])/stride:1/30;
 playback.fps=det.cams[0]?.data.fps||(interval>0?1/interval:30);
 if(!Number.isFinite(playback.fps)||playback.fps<=0)playback.fps=30;
 det.cams.forEach(c=>{c.start=cameraStart(c);});
 playback.start=det.cams.length?Math.max(...det.cams.map(c=>c.start)):(times?.[0]||0);
 const end=det.cams.length?Math.min(...det.cams.map(c=>c.start+(c.data.n_frames-1)/c.data.fps)):
   playback.start+((tri.data?.n_frames||0)-1)/playback.fps;
 playback.count=Math.max(0,Math.floor((end-playback.start)*playback.fps+1e-4)+1);
 playback.frame=0;
 const scrub=document.getElementById('scrub');scrub.max=Math.max(0,playback.count-1);scrub.value=0;
 document.getElementById('playback').style.display=playback.count?'flex':'none';
 if(playback.count)showPlaybackFrame(0);
}
function trajectoryMarker(epoch,frame){
 if(!tri.data)return -1;
 const times=tri.data.epoch_times_s;
 if(!times?.length)return Math.min(tri.data.smooth.length-1,Math.round(frame/(tri.data.stride||1)));
 const halfStep=(tri.data.stride||1)/playback.fps/2;
 if(epoch<times[0]-halfStep||epoch>times[times.length-1]+halfStep)return -1;
 let lo=0,hi=times.length-1;
 while(lo<hi){const mid=(lo+hi)>>1;if(times[mid]<epoch)lo=mid+1;else hi=mid;}
 return lo>0&&epoch-times[lo-1]<times[lo]-epoch?lo-1:lo;
}
async function showPlaybackFrame(frame){
 if(!playback.count)return false;
 const request=++playback.request;
 frame=Math.max(0,Math.min(Number(frame)||0,playback.count-1));
 const epoch=playback.start+frame/playback.fps;
 // Load all camera images before committing either view to the new time.
 const images=await Promise.all(det.cams.map(cam=>new Promise(resolve=>{
   const index=Math.max(0,Math.min(cam.data.n_frames-1,Math.round((epoch-cam.start)*cam.data.fps)));
   const img=new Image();
   img.onload=()=>resolve({cam,index,img,ok:true});
   img.onerror=()=>resolve({cam,index,img,ok:false});
   img.src='/api/result-frame?dir='+encodeURIComponent(cam.dir)+'&frame='+index;
 })));
 if(request!==playback.request)return false;
 for(const {cam,index,img,ok} of images){
   cam.frame=index;
   const pane=Array.from(document.querySelectorAll('.detpane')).find(p=>p.dataset.dir===cam.dir);
   if(!pane)continue;
   img.alt=ok?'':'Frame unavailable';
   pane.querySelector('img').replaceWith(img);
   if(ok)drawPaneOverlay(pane,index);
   else {const canvas=pane.querySelector('canvas');canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height);}
 }
 playback.frame=frame;
 tri.marker=trajectoryMarker(epoch,frame);drawScene();
 document.getElementById('scrub').value=frame;
 document.getElementById('playposition').textContent=`${(frame/playback.fps).toFixed(2)} / ${((playback.count-1)/playback.fps).toFixed(2)} s`;
 return true;
}
async function playNext(){
 if(!playback.playing)return;
 const next=playback.frame>=playback.count-1?0:playback.frame+1;
 const shown=await showPlaybackFrame(next);
 if(!shown||!playback.playing)return;
 if(playback.frame>=playback.count-1){stopPlayback();return;}
 playback.timer=setTimeout(playNext,1000/Math.min(120,playback.fps*2));
}
document.getElementById('play').onclick=()=>{
 if(playback.playing){stopPlayback();return;}
 if(!playback.count)return;
 if(det.editing)toggleEdit();
 playback.playing=true;
 const b=document.getElementById('play');b.innerHTML='&#10074;&#10074;';b.title='Pause';b.setAttribute('aria-label','Pause');
 playNext();
};
document.getElementById('scrub').oninput=e=>{stopPlayback();showPlaybackFrame(e.target.value);};

/* ---------- init ---------- */
function uniqueRuns(items){return [...new Set(items.map(x=>x.run||'').filter(Boolean))];}
function fillRunSelect(sel,runs,empty){
 sel.innerHTML='';
 if(!runs.length){sel.appendChild(opt('',empty));return '';}
 runs.forEach(r=>sel.appendChild(opt(r,r)));
 return runs[0];
}
function fillDetSelect(run,version){
 const ds=document.getElementById('detsel');
 ds.innerHTML='';
 const items=(window.allDetections||[]).filter(d=>(d.run||'')===run);
 if(!items.length){ds.appendChild(opt('','no cameras for this run'));clearDet();return;}
 items.forEach(d=>ds.appendChild(opt(d.dir,d.label.split('/').pop().trim())));
 return loadDetRun(items,items[0].dir,version);
}
function clearTri(){
 stopPlayback();tri.data=null;tri.dir='';
 if(noMocap)document.getElementById('show-reference').disabled=true;
 document.getElementById('tridl').style.display='none';
 document.getElementById('trimeta').innerHTML='<span>no reconstruction for this run</span>';
 document.getElementById('reference-info').textContent='';
 document.getElementById('comparison-download').style.display='none';
 drawScene();
}
function loadTriForRun(run,version){
 const items=(window.allTrajectories||[]).filter(t=>(t.run||'')===run);
 if(!items.length){clearTri();return;}
 return loadTri(items[0].dir,version);
}
async function selectRun(run){
 const version=++runVersion;
 clearDet();clearTri();
 playback.count=0;document.getElementById('playback').style.display='none';
 const loaded=await Promise.allSettled([fillDetSelect(run,version),loadTriForRun(run,version)]);
 if(version!==runVersion)return;
 if(loaded[0].status==='rejected'){clearDet();document.getElementById('detempty').textContent='Could not load detections.';}
 if(loaded[1].status==='rejected'){clearTri();document.getElementById('trimeta').textContent='Could not load reconstruction.';}
 configurePlayback();
}
(async()=>{
 const r=await getResults();
 window.allDetections=r.detections||[];
 window.allTrajectories=r.trajectories||[];
 if(noMocap){
   const run=resultParams.get('run')||'';
   const prefix=resultParams.get('scope')==='calibration'?'uploads/':'tracking_uploads/';
   window.allDetections=window.allDetections.filter(d=>d.dir.startsWith(prefix));
   window.allTrajectories=window.allTrajectories.filter(t=>t.dir.startsWith(prefix)||t.dir===prefix.slice(0,-1));
   document.getElementById('selected-run').textContent=run||'No tracking run selected';
   document.getElementById('detempty').textContent=run?'No detections for this run.':'No tracking run selected.';
   await selectRun(run);
   document.getElementById('detsel').onchange=e=>{if(e.target.value)setActiveDet(e.target.value);};
   return;
 }
 const ds=document.getElementById('detsel'),rs=document.getElementById('runselect');
 const runs=[...new Set([...uniqueRuns(window.allTrajectories),...uniqueRuns(window.allDetections)])];
 const firstRun=fillRunSelect(rs,runs,'no runs yet');
 selectRun(firstRun);
 rs.onchange=e=>selectRun(e.target.value);
 ds.onchange=e=>{if(e.target.value)setActiveDet(e.target.value);};
})();
</script>
</body></html>"""
