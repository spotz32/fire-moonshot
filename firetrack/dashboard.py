"""Embedded HTML for the FireTrack ground-control dashboard.

Kept in its own module so webui.py stays focused on routing/logic. The page is a
single self-contained document (styles + script inline) served at ``/``; it talks
to the same JSON API the server exposes.
"""

DASHBOARD_HTML = b"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FireTrack \xc2\xb7 Ground Control</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#0B0F14; --panel:#141B24; --raised:#1B2531; --line:#263241; --line-soft:#1d2733;
  --text:#E8EEF4; --muted:#8895A7; --faint:#5C6B7E;
  --ember:#FF6A2B; --ember-soft:#FFA666; --cyan:#36C5D9; --green:#46D17F; --warn:#FFC857;
  --disp:'Chakra Petch',system-ui,sans-serif;
  --mono:'IBM Plex Mono',ui-monospace,Menlo,monospace;
  --body:'Inter',system-ui,sans-serif;
  --r:12px;
}
*{box-sizing:border-box}
html,body{margin:0}
body{background:var(--bg);color:var(--text);font-family:var(--body);font-size:14px;line-height:1.5;
  background-image:radial-gradient(circle at 18% -10%,rgba(255,106,43,.10),transparent 42%),
    radial-gradient(circle at 92% 0%,rgba(54,197,217,.08),transparent 40%);
  background-attachment:fixed;}
a{color:var(--cyan)}
.wrap{max-width:1540px;margin:0 auto;padding:0 24px 60px}

/* ---------- HUD header ---------- */
header{position:sticky;top:0;z-index:30;border-bottom:1px solid var(--line);
  background:rgba(11,15,20,.82);backdrop-filter:blur(10px)}
.hud{max-width:1540px;margin:0 auto;padding:14px 24px;display:flex;align-items:center;gap:16px;flex-wrap:wrap}
.brand{display:flex;align-items:center;gap:12px}
.flame{width:30px;height:30px;flex:0 0 auto;filter:drop-shadow(0 0 9px rgba(255,106,43,.55))}
.brand h1{font-family:var(--disp);font-weight:700;font-size:21px;letter-spacing:.14em;margin:0;line-height:1}
.brand .ey{font-family:var(--mono);font-size:10px;letter-spacing:.34em;color:var(--ember);text-transform:uppercase}
.chips{display:flex;gap:8px;flex-wrap:wrap}
.chip{font-family:var(--mono);font-size:11px;letter-spacing:.05em;color:var(--muted);
  border:1px solid var(--line);border-radius:99px;padding:5px 11px;display:flex;align-items:center;gap:7px;background:var(--panel)}
.dot{width:8px;height:8px;border-radius:50%;background:var(--faint);box-shadow:0 0 0 0 transparent}
.dot.on{background:var(--green);box-shadow:0 0 8px var(--green)}
.dot.off{background:#3a4756}
.dot.run{background:var(--ember);animation:pulse 1.1s infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(255,106,43,.6)}70%{box-shadow:0 0 0 7px rgba(255,106,43,0)}100%{box-shadow:0 0 0 0 transparent}}
.tagline{width:100%;color:var(--muted);font-size:13px;margin-top:2px}

/* ---------- pipeline rail (signature) ---------- */
.rail{margin:16px 0 8px}
.rail-h{font-family:var(--mono);font-size:11px;letter-spacing:.28em;color:var(--faint);text-transform:uppercase;margin:0 0 12px}
.rail-row{display:grid;grid-template-columns:150px 1fr;gap:10px;align-items:stretch;margin-bottom:10px}
.rail-label{border:1px solid var(--line);border-radius:8px;background:var(--raised);padding:10px 12px;display:flex;flex-direction:column;justify-content:center;min-height:76px}
.rail-label b{font-family:var(--disp);font-size:13px;letter-spacing:.06em;text-transform:uppercase}
.rail-label span{font-family:var(--mono);font-size:11px;color:var(--muted);margin-top:6px}
.stages{display:grid;grid-template-columns:repeat(4,1fr);gap:0;position:relative}
.stages.three{grid-template-columns:repeat(3,1fr)}
.stages.five{grid-template-columns:repeat(5,minmax(0,1fr))}
.stage{position:relative;padding:10px 12px 11px;border:1px solid var(--line);background:var(--panel);
  border-right-width:0}
.stage:first-child{border-radius:8px 0 0 8px}
.stage:last-child{border-radius:0 8px 8px 0;border-right-width:1px}
.stage .top{display:flex;align-items:center;gap:8px;margin-bottom:5px}
.stage .num{font-family:var(--mono);font-size:11px;color:var(--faint)}
.stage .ico{width:16px;height:16px;color:var(--muted)}
.stage h3{font-family:var(--disp);font-weight:600;font-size:13px;margin:0;letter-spacing:.02em}
.stage p{margin:0;color:var(--muted);font-size:11.5px;min-height:30px;line-height:1.35}
.stage .foot{display:flex;align-items:center;justify-content:space-between;margin-top:7px}
.pill{font-family:var(--mono);font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;
  padding:3px 9px;border-radius:99px;border:1px solid var(--line);color:var(--faint)}
.metric{font-family:var(--mono);font-size:12px;color:var(--muted)}
.st-done h3,.st-done .ico{color:var(--green)}
.st-done .pill{color:var(--green);border-color:rgba(70,209,127,.4);background:rgba(70,209,127,.08)}
.st-done{box-shadow:inset 0 -2px 0 var(--green)}
.st-active h3,.st-active .ico{color:var(--ember)}
.st-active .pill{color:var(--ember);border-color:rgba(255,106,43,.5);background:rgba(255,106,43,.1)}
.st-active{box-shadow:inset 0 -2px 0 var(--ember)}
.st-partial .pill{color:var(--cyan);border-color:rgba(54,197,217,.4);background:rgba(54,197,217,.08)}
.st-partial{box-shadow:inset 0 -2px 0 var(--cyan)}
.connect{position:absolute;top:50%;right:-7px;width:14px;height:14px;z-index:5;color:var(--line)}
.stage:last-child .connect{display:none}

/* ---------- layout ---------- */
.workflow-shell{margin-top:18px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:var(--r);overflow:hidden}
.card>.hd{padding:13px 16px;border-bottom:1px solid var(--line-soft);display:flex;align-items:center;gap:10px}
.card>.hd h2{font-family:var(--disp);font-weight:600;font-size:14px;letter-spacing:.06em;margin:0;text-transform:uppercase}
.card>.bd{padding:16px}

/* tabs */
.tabs{display:flex;gap:6px}
.mode-tabs{margin-left:auto}
.tab{font-family:var(--disp);font-weight:600;letter-spacing:.04em;font-size:13px;padding:8px 15px;border-radius:8px;
  border:1px solid var(--line);background:var(--raised);color:var(--muted);cursor:pointer}
.tab.active{color:var(--bg);background:var(--ember);border-color:var(--ember)}
.tab .sub{font-family:var(--mono);font-weight:400;font-size:10px;display:block;letter-spacing:.04em;opacity:.8}

/* buttons */
.actions{display:flex;flex-wrap:wrap;gap:8px;margin:4px 0 14px}
button{font-family:var(--disp);font-weight:600;font-size:13px;letter-spacing:.03em;color:var(--text);
  background:var(--raised);border:1px solid var(--line);border-radius:8px;padding:9px 14px;cursor:pointer;
  display:inline-flex;align-items:center;gap:7px;transition:border-color .15s,background .15s,transform .05s}
button:hover:not(:disabled){border-color:var(--cyan)}
button:active:not(:disabled){transform:translateY(1px)}
button:disabled{opacity:.38;cursor:not-allowed}
button svg{width:15px;height:15px}
button.primary{background:var(--ember);border-color:var(--ember);color:#1a0d05}
button.primary:hover:not(:disabled){background:var(--ember-soft);border-color:var(--ember-soft)}
button.upload{background:var(--cyan);border-color:var(--cyan);color:#041316}
button.upload:hover:not(:disabled){background:#74DCEA;border-color:#74DCEA}
.hint{color:var(--faint);font-size:12px;margin-top:4px}

/* video list */
.listhd{display:flex;align-items:center;justify-content:space-between;font-family:var(--mono);font-size:11px;
  color:var(--faint);letter-spacing:.1em;text-transform:uppercase;margin:8px 2px 6px}
.vid{display:flex;align-items:center;gap:11px;padding:9px 11px;border:1px solid var(--line-soft);
  border-radius:9px;margin-bottom:7px;background:var(--raised)}
.vid input{accent-color:var(--ember);width:15px;height:15px}
.vid .nm{font-family:var(--mono);font-size:13px;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tag{font-family:var(--mono);font-size:10px;letter-spacing:.06em;padding:2px 8px;border-radius:99px}
.tag.ok{color:var(--green);background:rgba(70,209,127,.1);border:1px solid rgba(70,209,127,.3)}
.tag.no{color:var(--faint);border:1px solid var(--line)}
.empty{text-align:center;color:var(--muted);padding:26px 14px;border:1px dashed var(--line);border-radius:10px}
.empty .big{font-family:var(--disp);font-size:15px;color:var(--text);margin-bottom:5px}

button.danger{border-color:rgba(255,106,43,.4);color:var(--ember-soft)}
button.danger:hover:not(:disabled){border-color:var(--ember);background:rgba(255,106,43,.1)}
.prog{height:6px;border-radius:4px;background:var(--line);overflow:hidden;margin-top:8px}
.prog>i{display:block;height:100%;width:0;background:var(--cyan);transition:width .2s}

/* calibration */
.calib{margin-top:14px;border:1px solid var(--line);border-radius:10px;background:var(--raised);padding:14px}
.calib-row{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.calib-row>div:first-child{flex:1;min-width:200px}
.cal-state{font-family:var(--mono);font-size:11px;padding:2px 9px;border-radius:99px;border:1px solid var(--line);color:var(--faint);white-space:nowrap;min-width:max-content}
.cal-state.ok{color:var(--green);border-color:rgba(70,209,127,.3);background:rgba(70,209,127,.08)}
.cal-state.bad{color:var(--ember);border-color:rgba(255,106,43,.4);background:rgba(255,106,43,.1)}
.runpick{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:8px 0 10px}
.runpick select{font-family:var(--mono);font-size:12px;background:var(--bg);color:var(--text);border:1px solid var(--line);border-radius:7px;padding:7px 9px}
.flow{display:flex;flex-direction:column;gap:16px}
.flow-block{border:1px solid var(--line);background:var(--panel);border-radius:var(--r);padding:16px;overflow:hidden}
#mode-dataset,#mode-mocap_raw{background:var(--panel);border:1px solid var(--line);border-radius:var(--r);padding:16px;overflow:hidden}
.flow-head{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin-bottom:12px}
.flow-head h3{font-family:var(--disp);font-weight:600;font-size:15px;letter-spacing:.05em;text-transform:uppercase;margin:0}
.flow-head p{color:var(--muted);font-size:12.5px;margin:4px 0 0}
#mode-upload .runpick select{min-width:0;max-width:100%;flex:1}
#mode-upload .workflow-inputs{margin:0 0 12px}
#mode-upload .workflow-stages{margin-bottom:12px}
#mode-upload .workflow-utilities{display:grid;grid-template-columns:minmax(0,1fr) auto;align-items:center;gap:12px;border-top:1px solid var(--line-soft);padding-top:12px}
#mode-upload .utility-links{display:flex;align-items:center;flex-wrap:wrap;gap:12px;min-width:0}
#mode-upload .camera-repair{min-width:0}
#mode-upload .camera-repair summary{cursor:pointer;font-size:13px;color:var(--muted);padding:8px 0}
#mode-upload .camera-repair .actions{align-items:center;margin:8px 0 0}
#mode-upload .camera-repair select{min-width:0;max-width:100%;background:var(--raised);color:var(--text);border:1px solid var(--line);border-radius:6px;padding:8px;font-family:var(--mono);font-size:12px}
#mode-upload .workflow-utilities>.danger{grid-column:2;justify-self:end}
#mode-upload button{max-width:100%;letter-spacing:0;white-space:normal}
@media (max-width:560px){
  #mode-upload .workflow-utilities{grid-template-columns:minmax(0,1fr)}
  #mode-upload .workflow-utilities>.danger{grid-column:1}
  #mode-upload .runpick select{flex-basis:calc(100% - 48px)}
}

/* console */
.console{margin-top:18px}
.console .hd{justify-content:space-between}
.jobchip{font-family:var(--mono);font-size:11px;color:var(--muted);display:flex;align-items:center;gap:8px}
.jprogtext{font-family:var(--mono);font-size:11px;color:var(--ember);margin-left:auto;white-space:nowrap}
.jprog{height:5px;background:var(--line);display:none}
.jprog>i{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--ember),var(--ember-soft));transition:width .35s}
#log{font-family:var(--mono);font-size:12px;line-height:1.65;background:#070A0E;color:#bfe6d6;
  padding:14px 16px;height:300px;overflow:auto;white-space:pre-wrap;border-top:1px solid var(--line-soft)}
#log:empty:before{content:'Console idle. Run a stage to stream its log here.';color:var(--faint)}
.err{color:var(--ember-soft)}
.results-embed{margin-top:18px}
.results-embed>.hd{position:sticky;top:0;z-index:3;background:var(--panel)}
#results-run{display:none;min-width:0;flex:1;color:var(--muted);font-family:var(--mono);font-size:12px;overflow-wrap:anywhere}
.no-mocap #results-run{display:block}
.results-embed iframe{width:100%;height:calc(100vh - 96px);min-height:760px;border:0;display:block;background:var(--bg)}

/* annotator */
.annot{position:fixed;z-index:60;left:18px;right:18px;top:82px;bottom:18px;margin:0;box-shadow:0 18px 60px rgba(0,0,0,.45)}
.annot>.hd{position:sticky;top:0;z-index:3;background:var(--panel)}
.annot iframe{width:100%;height:calc(100% - 48px);border:0;display:block;background:var(--bg)}
.hidden{display:none!important}

@media (max-width:880px){
  .stages.five{grid-template-columns:1fr 1fr}
  .rail-row{grid-template-columns:1fr}
  .stages{grid-template-columns:1fr 1fr}
  .stages.three{grid-template-columns:1fr}
  .stage{border-right-width:1px;border-bottom-width:0}
  .stage:nth-child(1){border-radius:var(--r) 0 0 0} .stage:nth-child(2){border-radius:0 var(--r) 0 0}
  .stage:nth-child(3){border-radius:0;border-bottom-width:0} .stage:nth-child(4){border-radius:0;border-bottom-width:1px}
  .stage:nth-child(-n+2){border-bottom-width:0}
  .connect{display:none}
}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
:focus-visible{outline:2px solid var(--cyan);outline-offset:2px}
/* Purdue palette is scoped to the no-mocap view. */
body.no-mocap{
  --bg:#000000;--panel:#171717;--raised:#262626;--line:#484848;--line-soft:#333333;
  --text:#FAFAFA;--muted:#BDBDBD;--faint:#A3A3A3;
  --ember:#CFB991;--ember-soft:#E5D6BA;--cyan:#CFB991;--green:#CFB991;--warn:#DAAA00;--error:#FFA3A3;
  background-image:none;
}
.no-mocap header{background:rgba(0,0,0,.95)}
.no-mocap .flame{filter:none}
.no-mocap .flame path:first-child{fill:#CFB991}
.no-mocap .flame path:last-child{fill:#FFFFFF}
.no-mocap button.primary,.no-mocap button.upload{color:#000000}
.no-mocap button.upload:hover:not(:disabled){background:#E5D6BA;border-color:#E5D6BA}
.no-mocap button.danger{color:#FFA3A3;border-color:#754141}
.no-mocap button.danger:hover:not(:disabled){background:#331E1E;border-color:#FFA3A3}
.no-mocap .cal-state.bad,.no-mocap .err{color:#FFA3A3;border-color:#754141}
.no-mocap .st-done .pill,.no-mocap .cal-state.ok,.no-mocap .tag.ok{color:#CFB991;border-color:#8C7D62;background:rgba(207,185,145,.08)}
.no-mocap .st-active h3,.no-mocap .st-active .ico{color:#E5D6BA}
.no-mocap .st-active .pill{color:#000000;border-color:#CFB991;background:#CFB991}
.no-mocap .st-partial .pill{color:#CFB991;border-color:#8C7D62;background:rgba(207,185,145,.08)}
.no-mocap .dot.on{box-shadow:none}
.no-mocap .dot.off{background:#777777}
.no-mocap .dot.run{animation:purdue-pulse 1.1s infinite}
@keyframes purdue-pulse{0%{box-shadow:0 0 0 0 rgba(207,185,145,.5)}70%{box-shadow:0 0 0 7px rgba(207,185,145,0)}100%{box-shadow:0 0 0 0 rgba(207,185,145,0)}}
.no-mocap #log{background:#080808;color:#DCDCDC}
.no-mocap .jprog>i{background:#CFB991}
.no-mocap .results-embed>.hd{flex-wrap:wrap}
@media (prefers-reduced-motion:reduce){.no-mocap .dot.run{animation:none}}
</style></head>
<body class="no-mocap">
<header><div class="hud">
  <div class="brand">
    <svg class="flame" viewBox="0 0 24 24" fill="none"><path d="M12 2c1 3.5-2 4.8-2 7.5C10 11.4 11 12.5 11 12.5S9 12 8.2 10.4C7 12 6.5 13.6 6.5 15.2 6.5 18.9 9 21.5 12 21.5s5.5-2.6 5.5-6.3c0-3.6-2.4-5.5-3.7-8.2C12.9 5.3 13 3.4 12 2Z" fill="#FF6A2B"/><path d="M12 21.5c1.7 0 3-1.4 3-3.2 0-1.9-1.4-2.8-2-4.3-.7 1-1.6 1.6-1.6 2.9 0 .8.5 1.4.5 1.4s-1-.4-1.4-1.4c-.5.7-.8 1.5-.8 2.4 0 1.8 1.1 2.2 2.3 2.2Z" fill="#FFC857"/></svg>
  <div><h1>FIRETRACK</h1><span class="ey">Ground Control</span></div>
  </div>
  <div class="tabs mode-tabs">
    <div class="tab active" id="tab-upload" onclick="setMode('upload')">No-Mocap<span class="sub">camera clips + calibration</span></div>
    <div class="tab" id="tab-mocap_raw" onclick="setMode('mocap_raw')">Mocap Raw<span class="sub">6D TSV, no format</span></div>
    <div class="tab" id="tab-dataset" onclick="setMode('dataset')">Mocap<span class="sub">legacy 6D TSV</span></div>
  </div>
  <div class="chips" id="chips"></div>
  <div class="tagline">Track a drone across synced phone cameras &mdash; segment it with SAM3, then solve its 3D flight path with mocap or flight-log calibration.</div>
</div></header>

<div class="wrap">

<section class="rail hidden" id="mission-rail">
  <p class="rail-h" id="rail-title">Mission pipeline</p>
  <div id="stages"></div>
</section>

<main class="workflow-shell">

      <div id="mode-dataset" class="hidden">
        <p class="hint" style="margin:0 0 12px">Upload a 5-27 run zip containing camera folders plus the 6D mocap TSV. The app extracts it under <code>/work/dataset_uploads</code> and prepares it for the 5-27 pipeline.</p>
        <div class="actions">
          <button onclick="document.getElementById('dataset-zip').click()" data-keep>@FMT Upload zip</button>
          <button class="danger" onclick="clearDataset()" data-keep>Clear dataset</button>
        </div>
        <input type="file" id="dataset-zip" accept=".zip,application/zip" class="hidden">
        <div id="ds-progress"></div>
        <div class="actions">
          <button onclick="run('format')" data-st="format">@FMT Format</button>
          <button onclick="annotate('dataset')" data-keep>@CLK Annotate</button>
          <button onclick="run('detect')" data-st="detect">@DET Detect</button>
          <button onclick="run('triangulate')" data-st="triangulate">@TRI Triangulate</button>
          <button class="primary" onclick="run('run-all')" data-st="run-all">@RUN Run all</button>
        </div>
        <p class="hint">Leave all clips unchecked to process every formatted camera.</p>
        <div class="listhd"><span>Camera clips</span><label style="text-transform:none;letter-spacing:0;cursor:pointer"><input type="checkbox" id="ds-all" onchange="toggleAll('dataset')"> select all</label></div>
        <div id="ds-list"></div>
      </div>

      <div id="mode-mocap_raw" class="hidden">
        <p class="hint" style="margin:0 0 12px">Upload a 5-27 run zip containing camera folders plus the 6D mocap TSV. This diagnostic path uses the raw uploaded videos directly and does not run Format.</p>
        <div class="actions">
          <button onclick="document.getElementById('raw-dataset-zip').click()" data-keep>@FMT Upload zip</button>
          <button class="danger" onclick="clearDataset()" data-keep>Clear dataset</button>
        </div>
        <input type="file" id="raw-dataset-zip" accept=".zip,application/zip" class="hidden">
        <div id="raw-progress"></div>
        <div class="actions">
          <button onclick="annotate('mocap_raw')" data-keep>@CLK Annotate</button>
          <button onclick="runFor('detect','mocap_raw')" data-st="detect">@DET Detect</button>
          <button onclick="runFor('triangulate','mocap_raw')" data-st="triangulate">@TRI Triangulate</button>
        </div>
        <p class="hint">Leave all clips unchecked to process every raw camera. Outputs are written separately under <code>/work/detections/mocap_raw</code> and <code>/work/triangulation/mocap_raw</code>.</p>
        <div class="listhd"><span>Raw camera clips</span><label style="text-transform:none;letter-spacing:0;cursor:pointer"><input type="checkbox" id="raw-all" onchange="toggleAll('mocap_raw')"> select all</label></div>
        <div id="raw-list"></div>
      </div>

      <div id="mode-upload">
        <div class="flow">
          <section class="flow-block">
            <div class="flow-head">
              <div><h3>Camera Calibration</h3></div>
              <span id="cal-state" class="cal-state">not loaded</span>
            </div>
            <input type="file" id="cal-run-zip" accept=".zip,application/zip" class="hidden">
            <input type="file" id="calfile" accept=".json,application/json" class="hidden">
            <div id="calibration-setup">
            <div class="actions workflow-inputs">
              <button class="upload" onclick="document.getElementById('cal-run-zip').click()" data-keep>@FMT Upload calibration zip</button>
              <button onclick="document.getElementById('calfile').click()" data-keep>Load calibration JSON</button>
              <span id="cal-log-state" class="hint"></span>
            </div>
            <div id="up-progress"></div>
            <div id="calibration-stages" class="workflow-stages"></div>
            <div class="actions">
              <button onclick="annotate('upload')" data-keep>@CLK Annotate</button>
              <button class="primary" onclick="runFor('detect','upload')" data-st="detect">@DET Detect</button>
              <button onclick="runFor('calibrate','upload')" data-st="calibrate">@TRI Calibrate</button>
            </div>
            <div class="workflow-utilities">
              <div class="utility-links">
                <button id="cal-download" onclick="downloadCalibrationJson()" data-keep disabled>Download calibration JSON</button>
                <button type="button" onclick="window.open('/results?mode=nomocap&amp;scope=calibration&amp;run=Calibration%20Flight','_blank','noopener')" data-keep>Review calibration detections</button>
                <details class="camera-repair">
                  <summary>Reprocess camera</summary>
                  <div class="actions">
                    <label for="cal-repair-camera">Camera</label>
                    <select id="cal-repair-camera" onchange="updateCameraRepair('calibration')" disabled></select>
                    <button id="cal-clear-camera-outputs" class="danger" onclick="runCameraAction('clear-outputs','calibration')" disabled>Clear camera outputs</button>
                    <button id="cal-annotate-camera" onclick="annotate('upload',document.getElementById('cal-repair-camera').value)" disabled>Annotate camera</button>
                    <button id="cal-detect-camera" onclick="runCameraAction('detect','calibration')" disabled>Detect camera</button>
                  </div>
                </details>
              </div>
              <button id="clear-calibration" class="danger" onclick="clearCalibration()">Clear calibration</button>
            </div>
            </div>
          </section>

          <section class="flow-block">
            <div class="flow-head">
              <div><h3>Tracking Runs</h3></div>
            </div>
            <div class="runpick workflow-inputs">
              <label for="tracking-run">Run</label>
              <select id="tracking-run" onchange="selectTrackingRun(this.value)"></select>
              <input type="file" id="tracking-run-zip" accept=".zip,application/zip" class="hidden">
              <button class="upload" onclick="document.getElementById('tracking-run-zip').click()" data-keep>@FMT Upload tracking zip</button>
            </div>
            <div id="tracking-progress"></div>
            <div id="tracking-stages" class="workflow-stages"></div>
            <div class="actions">
              <button onclick="annotate('tracking')" data-keep>@CLK Annotate</button>
              <button class="primary" onclick="runFor('detect','tracking')" data-st="detect">@DET Detect</button>
              <button id="btn-up-tri" onclick="runFor('triangulate','tracking')" data-st="triangulate">@TRI Triangulate</button>
              <button id="btn-up-classify" onclick="runFor('classify','tracking')" data-st="classify">@CLS Classify</button>
            </div>
            <div class="workflow-utilities">
              <details class="camera-repair">
                <summary>Reprocess camera</summary>
                <div class="actions">
                  <label for="repair-camera">Camera</label>
                  <select id="repair-camera" onchange="updateCameraRepair()" disabled></select>
                  <button id="clear-camera-outputs" class="danger" onclick="runCameraAction('clear-outputs')" disabled>Clear camera outputs</button>
                  <button id="annotate-camera" onclick="annotate('tracking',document.getElementById('repair-camera').value)" disabled>Annotate camera</button>
                  <button id="detect-camera" onclick="runCameraAction('detect')" disabled>Detect camera</button>
                </div>
              </details>
              <button id="clear-tracking-outputs" class="danger" onclick="clearTrackingOutputs()" disabled>Clear outputs</button>
            </div>
          </section>
        </div>
      </div>

</main>

<section class="card console">
  <div class="hd"><h2>Console</h2><span class="jobchip" id="jobchip"></span><span class="jprogtext" id="jprogtext"></span></div>
  <div class="jprog" id="jprog"><i></i></div>
  <div id="log"></div>
</section>

<section class="card results-embed">
  <div class="hd"><h2>Results</h2><span id="results-run" aria-live="polite">No tracking run selected</span><button onclick="reloadResults()" data-keep style="margin-left:auto">Refresh results</button></div>
  <iframe id="results-frame" title="FireTrack results" src="/results?mode=nomocap&amp;embedded=1&amp;run="></iframe>
</section>

<section class="card annot hidden" id="annot-panel">
  <div class="hd"><h2>Annotator</h2><button onclick="closeAnnot()" data-keep style="margin-left:auto">Close</button></div>
  <iframe id="annot" title="Click annotator"></iframe>
</section>

</div>

<script>
const ICONS={
 '@FMT':'M3 7h18M3 12h18M3 17h10','@CLK':'M12 2v4M12 18v4M2 12h4M18 12h4M7 7l3 3M14 14l3 3',
 '@DET':'M12 5C6 5 2.7 12 2.7 12S6 19 12 19s9.3-7 9.3-7S18 5 12 5Z','@TRI':'M12 3l9 16H3z',
 '@CLS':'M4 5h16M4 12h10M4 19h7M17 10l3 2-3 2',
 '@RUN':'M5 4l14 8-14 8z'};
const DATASET_STAGES=[
 {id:'format',icon:'@FMT',name:'Format',desc:'Straighten and standardize the raw phone clips.'},
 {id:'clicks',icon:'@CLK',name:'Annotate',desc:'Click the drone once per clip to lock onto it.'},
 {id:'detect',icon:'@DET',name:'Detect',desc:'Track the drone frame-by-frame in 2D with SAM3.'},
 {id:'triangulate',icon:'@TRI',name:'Triangulate',desc:'Fuse the camera tracks into one 3D flight path.'},
];
const MOCAP_RAW_STAGES=[
 {id:'clicks',icon:'@CLK',name:'Annotate',desc:'Click the drone once per raw clip to lock onto it.'},
 {id:'detect',icon:'@DET',name:'Detect',desc:'Track the drone in the raw uploaded videos.'},
 {id:'triangulate',icon:'@TRI',name:'Triangulate',desc:'Fuse raw-video detections with the 6D mocap data.'},
];
const CALIBRATION_STAGES=[
 {id:'upload',icon:'@FMT',name:'Upload',desc:'Load synced calibration clips and the matching flight logs.'},
 {id:'clicks',icon:'@CLK',name:'Annotate',desc:'Click the calibration drone once per camera.'},
 {id:'detect',icon:'@DET',name:'Detect',desc:'Track the calibration flight in each camera.'},
 {id:'calibrate',icon:'@TRI',name:'Calibrate',desc:'Estimate camera poses, one focal length per camera, and the shared calibration lag.'},
];
const TRACKING_STAGES=[
 {id:'upload',icon:'@FMT',name:'Upload',desc:'Select or upload a tracking flight.'},
 {id:'clicks',icon:'@CLK',name:'Annotate',desc:'Click the tracked drone once per camera.'},
 {id:'detect',icon:'@DET',name:'Detect',desc:'Track the selected run frame-by-frame.'},
 {id:'triangulate',icon:'@TRI',name:'Triangulate',desc:'Fuse the selected run into one 3D trajectory.'},
 {id:'classify',icon:'@CLS',name:'Classify',desc:'Classify the reconstructed trajectory as ArduPilot or PX4.'},
];
const HELP={
 upload:[
	  ['Calibrate','Upload a calibration zip with camera clips, intrinsics, metadata, and flight logs; then annotate, detect, and calibrate.'],
  ['Save calibration','Keep the downloaded calibration JSON so fixed camera poses can be reused without recalibrating.'],
  ['Add tracking runs','Upload each tracking zip as its own run, then select it from the run table or dropdown.'],
	  ['Reconstruct and classify','For the selected run, annotate, detect, triangulate, and classify; inspect outputs in Results below.'],
 ],
 dataset:[
  ['Upload mocap run','Upload a legacy 5-27 zip containing camera folders and the matching 6D mocap TSV.'],
  ['Format clips','Run Format to prepare the uploaded camera videos for the mocap pipeline.'],
  ['Annotate and detect','Click the drone once per clip if needed, then run SAM3 detection to create 2D tracks.'],
  ['Triangulate with mocap','Run Triangulate to fuse camera detections with the 6D mocap data and review outputs in Results.'],
 ],
 mocap_raw:[
  ['Upload mocap run','Upload a legacy 5-27 zip containing camera folders and the matching 6D mocap TSV.'],
  ['Skip formatting','This diagnostic path uses raw uploaded videos directly.'],
  ['Annotate and detect','Click the drone once per raw clip if needed, then run SAM3 detection.'],
  ['Triangulate with mocap','Run Triangulate to compare raw-video geometry against the 6D mocap ground truth.'],
 ],
};
function svg(cls,d){return `<svg class="${cls}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="${d}"/></svg>`;}
let mode='upload',since=0,polling=false,lastJob='',activeTrackingRun='',calibrationDownloadArmed=false;
let resultsKey='upload:';
let clearingTrackingOutputs=false;
let clearingCalibration=false;
const cameraRepairs={tracking:{pending:false,run:'',cameras:[],running:false},calibration:{pending:false,run:'',cameras:[],running:false}};

function setMode(m){mode=m;
 document.body.classList.toggle('no-mocap',m==='upload');
 for(const k of ['upload','mocap_raw','dataset']){
   document.getElementById('tab-'+k).classList.toggle('active',k===m);
   document.getElementById('mode-'+k).classList.toggle('hidden',k!==m);}
 renderHelp();syncResults();}
function renderHelp(){
 const el=document.getElementById('getting-started'); if(!el)return;
 el.innerHTML=(HELP[mode]||HELP.upload).map(([h,t])=>`<li><b>${h}.</b> ${t}</li>`).join('');
}
function selected(which){return [...document.querySelectorAll('.cb-'+which+':checked')].map(c=>c.value);}
function toggleAll(which){const on=document.getElementById((which==='dataset'?'ds':'raw')+'-all').checked;
 document.querySelectorAll('.cb-'+which).forEach(c=>c.checked=on);}
function trackingRunNameForFile(file){
 return (file.name||'tracking_run').replace(/\.[^.]+$/,'')||'tracking_run';
}
async function selectTrackingRun(run){
 if(!run)return;
 const r=await fetch('/api/tracking-run/select',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({run})});
 if(!r.ok){flash('Could not select the tracking run.');refresh();return;}
 activeTrackingRun=run;syncResults();refresh();
}
function displayJobName(name){
 if(!name)return '';
 const p=String(name).split(':');
 const stage=(p[p.length-1]||'').replace(/-/g,' ');
 const niceStage=stage?stage.charAt(0).toUpperCase()+stage.slice(1):'';
 if(p[0]==='tracking')return ['Tracking',...p.slice(1,-1),niceStage].filter(Boolean).join(': ');
 if(p[0]==='upload')return ['Calibration',niceStage].filter(Boolean).join(': ');
 if(p[0]==='mocap_raw')return ['Mocap Raw',niceStage].filter(Boolean).join(': ');
 if(p[0]==='dataset')return ['Mocap',niceStage].filter(Boolean).join(': ');
 return name;
}

function chip(label,on,text){return `<div class="chip"><span class="dot ${on?'on':'off'}"></span>${label}${text?' \\u00b7 '+text:''}</div>`;}
function renderChips(s){const e=s.env||{};const j=s.job;
 let run=j.status==='running'?`<div class="chip"><span class="dot run"></span>${displayJobName(j.name)||'Job'}</div>`:'';
 document.getElementById('chips').innerHTML=run+
  chip('GPU',e.gpu)+chip('WEIGHTS',e.weights_cached,e.offline?'offline':'')+
  chip('SAM3',e.weights_cached?true:false);}

function sourceJobRunning(s,source,id,run){
 const j=s.job,parts=(j.name||'').split(':'),jstage=parts[parts.length-1]||parts[0];
 if(j.status!=='running' || jstage!==id)return false;
 if(source==='tracking')return parts[0]==='tracking' && (!run || parts[1]===run);
 return parts[0]===source || (source==='dataset' && parts.length===1);
}
function stageState(s,id){
 if(sourceJobRunning(s,mode,id)){ return 'active'; }
 const j=s.job,parts=(j.name||'').split(':'),jstage=parts[parts.length-1]||parts[0];
 if(j.status==='running'&&jstage==='run-all'&&id!=='clicks') return 'active';
 if(id==='format') return (s.outputs.formatted&&s.dataset.length)?'done':'idle';
 if(id==='clicks'){const c=mode==='upload'?s.tracking_uploads_clicks:(mode==='mocap_raw'?s.mocap_raw_clicks:s.clicks); if(c&&c.clicked>0) return c.pending?'partial':'done'; return 'idle';}
 if(id==='detect'){const all=mode==='upload'?s.tracking_uploads:(mode==='mocap_raw'?s.mocap_raw:s.dataset);const tot=all.length;
   const done=all.filter(v=>v.has_detection).length;
   if(!tot)return 'idle'; return done>=tot?'done':(done?'partial':'idle');}
 if(id==='calibrate') return (s.calibration&&s.calibration.present)?'done':'idle';
 if(id==='triangulate') return (mode==='upload'?s.outputs.tracking_uploads_triangulation:(mode==='mocap_raw'?s.outputs.mocap_raw_triangulation:s.outputs.triangulation))?'done':'idle';
 return 'idle';}
function stageMetric(s,id){
 if(id==='format') return s.dataset.length?s.dataset.length+' clips':'';
 if(id==='clicks'){const c=mode==='upload'?s.tracking_uploads_clicks:(mode==='mocap_raw'?s.mocap_raw_clicks:s.clicks); return c?c.clicked+'/'+c.total:'';}
 if(id==='detect'){const all=mode==='upload'?s.tracking_uploads:(mode==='mocap_raw'?s.mocap_raw:s.dataset);
   return all.length?all.filter(v=>v.has_detection).length+'/'+all.length:'';}
 if(id==='calibrate') return (s.calibration&&s.calibration.present)?((s.calibration.n_cameras||0)+' cameras'):'';
 if(id==='triangulate'){ if(mode==='upload')return s.outputs.tracking_uploads_triangulation?'solved':''; if(mode==='mocap_raw')return s.outputs.mocap_raw_triangulation?'solved':''; return s.outputs.triangulation?'solved':''; }
 return '';}
const LABELS={idle:'Idle',active:'Running',partial:'Partial',done:'Complete'};
function uploadStageState(s,scope,id){
 if(id==='upload'){
   if(scope==='calibration')return (s.calibration_summary&&s.calibration_summary.clips>0)?'done':'idle';
   return (s.tracking_uploads&&s.tracking_uploads.length>0)?'done':'idle';
 }
 if(scope==='calibration'){
   if(sourceJobRunning(s,'upload',id))return 'active';
   const c=s.calibration_summary||{};
   if(id==='clicks'){const a=c.annotations; return a&&a.clicked>0 ? (a.pending?'partial':'done') : 'idle';}
   if(id==='detect')return c.clips ? (c.detections>=c.clips?'done':(c.detections?'partial':'idle')) : 'idle';
   if(id==='calibrate')return c.calibrated?'done':'idle';
 }
 if(scope==='tracking'){
   if(sourceJobRunning(s,'tracking',id,activeTrackingRun))return 'active';
   if(id==='clicks'){const a=s.tracking_uploads_clicks; return a&&a.clicked>0 ? (a.pending?'partial':'done') : 'idle';}
   if(id==='detect'){const all=s.tracking_uploads||[],done=all.filter(v=>v.has_detection).length; return all.length ? (done>=all.length?'done':(done?'partial':'idle')) : 'idle';}
   if(id==='triangulate')return s.outputs.tracking_uploads_triangulation?'done':'idle';
   if(id==='classify')return s.classification&&s.classification.result?'done':'idle';
 }
 return 'idle';
}
function uploadStageMetric(s,scope,id){
 if(scope==='calibration'){
   const c=s.calibration_summary||{};
   if(id==='upload')return c.clips?`${c.clips} clips`:'';
   if(id==='clicks')return c.annotations?`${c.annotations.clicked||0}/${c.annotations.total||0}`:'';
   if(id==='detect')return c.clips?`${c.detections||0}/${c.clips}`:'';
   if(id==='calibrate')return c.calibrated?`${c.cameras||0} cameras`:'';
 }
 if(scope==='tracking'){
   const all=s.tracking_uploads||[];
   if(id==='upload')return activeTrackingRun?activeTrackingRun:'';
   if(id==='clicks'){const a=s.tracking_uploads_clicks; return a?`${a.clicked||0}/${a.total||0}`:'';}
   if(id==='detect')return all.length?`${all.filter(v=>v.has_detection).length}/${all.length}`:'';
   if(id==='triangulate')return s.outputs.tracking_uploads_triangulation?'solved':'';
   if(id==='classify')return s.classification&&s.classification.result?s.classification.result.prediction:'';
 }
 return '';
}
function stageCard(st,i,state,m){
 return `<div class="stage st-${state}">
    <div class="top"><span class="num">0${i+1}</span>${svg('ico',ICONS[st.icon])}<h3>${st.name}</h3></div>
    <p>${st.desc}</p>
    <div class="foot"><span class="pill">${LABELS[state]}</span><span class="metric">${m}</span></div>
    ${svg('connect','M9 6l6 6-6 6')}</div>`;
}
function renderUploadStages(s){
 const cal=CALIBRATION_STAGES.map((st,i)=>stageCard(st,i,uploadStageState(s,'calibration',st.id),uploadStageMetric(s,'calibration',st.id))).join('');
 const tr=TRACKING_STAGES.map((st,i)=>stageCard(st,i,uploadStageState(s,'tracking',st.id),uploadStageMetric(s,'tracking',st.id))).join('');
 document.getElementById('mission-rail').classList.add('hidden');
 document.getElementById('stages').innerHTML='';
 const calTarget=document.getElementById('calibration-stages');
 if(calTarget)calTarget.innerHTML=
  `<div class="stages">${cal}</div>`;
 const trackTarget=document.getElementById('tracking-stages');
 if(trackTarget)trackTarget.innerHTML=
  `<div class="stages five">${tr}</div>`;
}
function renderStages(s){
	 if(mode==='upload'){renderUploadStages(s);return;}
	 document.getElementById('mission-rail').classList.remove('hidden');
	 document.getElementById('rail-title').textContent='Mission pipeline';
	 const calTarget=document.getElementById('calibration-stages'); if(calTarget)calTarget.innerHTML='';
	 const target=document.getElementById('tracking-stages'); if(target)target.innerHTML='';
	 const stages=mode==='mocap_raw'?MOCAP_RAW_STAGES:DATASET_STAGES;
	 const cls=stages.length===3?'stages three':'stages';
	 document.getElementById('stages').innerHTML=`<div class="${cls}">`+stages.map((st,i)=>{
  const state=stageState(s,st.id),m=stageMetric(s,st.id);
  return stageCard(st,i,state,m);}).join('')+'</div>';}

function vidRow(which,v){return `<div class="vid"><input type="checkbox" class="cb-${which}" value="${v.label}">
  <span class="nm">${v.label}</span><span class="tag ${v.has_detection?'ok':'no'}">${v.has_detection?'detected':'pending'}</span></div>`;}
function renderList(id,which,items,emptyTitle,emptyBody){
 const el=document.getElementById(id);
 el.innerHTML=items.length?items.map(v=>vidRow(which,v)).join('')
  :`<div class="empty"><div class="big">${emptyTitle}</div>${emptyBody}</div>`;}
function renderTrackingRuns(s){
 const sel=document.getElementById('tracking-run'); if(!sel)return;
 const runs=s.tracking_runs||[];
 activeTrackingRun=s.selected_tracking_run||runs[0]||'';
 const current=sel.value;
 sel.innerHTML=runs.length?runs.map(r=>`<option value="${r}">${r}</option>`).join(''):'<option value="">no tracking flights yet</option>';
 if(activeTrackingRun)sel.value=activeTrackingRun;
 else if(current&&runs.includes(current))sel.value=current;
 syncResults();
}
function renderCameraRepair(s,scope='tracking'){
 const state=cameraRepairs[scope],prefix=scope==='calibration'?'cal-':'';
 const sel=document.getElementById(prefix+'repair-camera'),run=scope==='tracking'?activeTrackingRun:'calibration';
 const previous=state.run===run?sel.value:'';
 state.run=run;state.cameras=(scope==='tracking'?s.tracking_uploads:s.uploads)||[];state.running=s.job.status==='running';
 sel.replaceChildren();
 for(const camera of state.cameras){
   const option=document.createElement('option');option.value=camera.label;option.textContent=camera.label;sel.appendChild(option);
 }
 if(state.cameras.some(c=>c.label===previous))sel.value=previous;
 updateCameraRepair(scope);
}
function updateCameraRepair(scope='tracking'){
 const state=cameraRepairs[scope],prefix=scope==='calibration'?'cal-':'';
 const sel=document.getElementById(prefix+'repair-camera'),camera=state.cameras.find(c=>c.label===sel.value);
 const disabled=state.running||state.pending||!state.run||!camera;
 sel.disabled=state.running||state.pending||!state.cameras.length;
 document.getElementById(prefix+'clear-camera-outputs').disabled=disabled;
 document.getElementById(prefix+'annotate-camera').disabled=disabled;
 document.getElementById(prefix+'detect-camera').disabled=disabled||camera.has_detection;
 document.getElementById(prefix+'detect-camera').title=camera&&camera.has_detection?"Clear this camera's outputs before detecting it again.":'Run SAM3 for this camera only';
}

async function refresh(){
 let s; try{ s=await(await fetch('/api/status')).json(); }catch(e){ return; }
 renderTrackingRuns(s);
 renderChips(s); renderStages(s);
 renderList('ds-list','dataset',s.dataset,'No formatted clips yet','Upload a 5-27 run zip, then run Format.');
 renderList('raw-list','mocap_raw',s.mocap_raw||[],'No raw clips yet','Upload a 5-27 run zip to use raw videos directly.');
 const j=s.job,running=j.status==='running';
 document.getElementById('jobchip').innerHTML=j.name?
   `<span class="dot ${running?'run':(j.status==='failed'?'off':'on')}"></span>${displayJobName(j.name)} \\u2014 ${j.status}${j.error?' <span class="err">('+j.error+')</span>':''}`:'';
 document.querySelectorAll('main button,.console button').forEach(b=>{if(!b.dataset.keep)b.disabled=running;});
 // calibration state + 3D-button gating (upload mode)
 const cal=s.calibration||{present:false,n_cameras:0};
 const cs=document.getElementById('cal-state');
 if(cs){cs.className='cal-state'+(cal.present?' ok':(cal.error?' bad':''));
   cs.textContent=cal.present?(cal.n_cameras+' cameras'):(cal.error?'invalid file':'not loaded');}
 if(calibrationDownloadArmed&&j.name==='upload:calibrate'&&j.status==='done'&&cal.present){
   calibrationDownloadArmed=false;
   downloadCalibrationJson();
 }
 document.getElementById('cal-download').disabled=!cal.present;
 document.getElementById('clear-calibration').disabled=running||clearingCalibration;
 const logState=document.getElementById('cal-log-state');
 if(logState){logState.textContent=(s.calibration_log&&s.calibration_log.present)?'flight log loaded':'no flight log loaded';}
 const detUp=(s.tracking_uploads||[]).filter(v=>v.has_detection).length;
 const tri=document.getElementById('btn-up-tri');
 if(tri)tri.disabled=running||!cal.present||detUp<2;
 const classify=document.getElementById('btn-up-classify'),classification=s.classification||{};
 if(classify){classify.disabled=running||!classification.ready;classify.title=classification.reason||'Classify the reconstructed trajectory';}
 document.getElementById('clear-tracking-outputs').disabled=running||clearingTrackingOutputs||!activeTrackingRun;
 renderCameraRepair(s);
 renderCameraRepair(s,'calibration');
 updateProgress(j);
 if(running&&!polling)pollLog();
}
function updateProgress(j){
 const bar=document.getElementById('jprog'),txt=document.getElementById('jprogtext'),p=j&&j.progress;
 if(j&&j.status==='running'&&p&&p.total){
   const pct=Math.min(100,Math.round(100*(p.frame||0)/p.total));
   bar.style.display='block';bar.firstElementChild.style.width=pct+'%';
   txt.textContent=`${p.label||''} ${p.frame||0}/${p.total} (${pct}%)`+(p.n_videos>1?` \\u00b7 clip ${p.video}/${p.n_videos}`:'');
 }else{bar.style.display='none';bar.firstElementChild.style.width='0';txt.textContent='';}
}
async function run(stage){
 return runFor(stage,mode);
}
async function clearTrackingOutputs(){
 const run=activeTrackingRun;
 if(!run||clearingTrackingOutputs)return;
 if(!confirm(`Clear all camera detections, the reconstruction, and classification for "${run}"? Videos, annotations, and calibration will be kept.`))return;
 clearingTrackingOutputs=true;
 document.getElementById('clear-tracking-outputs').disabled=true;
 try{
   const r=await fetch('/api/tracking-run/clear-outputs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({run})});
   const d=await r.json().catch(()=>({}));
   if(!r.ok){flash(d.error||'Could not clear outputs.');return;}
   since=0;document.getElementById('log').textContent='';pollLog();
 }catch(e){flash('Could not clear outputs. Check the connection and try again.');}
 finally{clearingTrackingOutputs=false;refresh();}
}
async function clearCalibration(){
 if(clearingCalibration)return;
 if(!confirm('Delete all calibration clips, metadata, flight logs, annotations, detections, reconstruction outputs, and the saved intrinsics/extrinsics JSON? Tracking runs and their saved results, mocap data, and original files outside the work folder will be kept. This cannot be undone.'))return;
 clearingCalibration=true;
 document.getElementById('clear-calibration').disabled=true;
 try{
   const r=await fetch('/api/calibration/clear',{method:'POST'});
   const d=await r.json().catch(()=>({}));
   if(!r.ok){flash(d.error||'Could not clear calibration.');return;}
   calibrationDownloadArmed=false;
   closeAnnot();
   document.getElementById('up-progress').textContent='';
   since=0;document.getElementById('log').textContent='';pollLog();
 }catch(e){flash('Could not clear calibration. Check the connection and try again.');}
 finally{clearingCalibration=false;refresh();}
}
async function runCameraAction(action,scope='tracking'){
 const state=cameraRepairs[scope],prefix=scope==='calibration'?'cal-':'';
 const run=activeTrackingRun,camera=document.getElementById(prefix+'repair-camera').value;
 if((scope==='tracking'&&!run)||!camera||state.pending||state.running)return;
 const message=scope==='tracking'?`Clear detections for "${camera}" and the reconstruction for "${run}"? All videos, annotations, calibration, and other cameras' detections will be kept. SAM3 will not run until you click Detect camera.`:
   `Clear calibration detections for "${camera}", the calibration reconstruction, and the active calibration JSON? Other cameras' detections, all videos, intrinsics input files, annotations, flight logs, tracking results, and saved files outside the work folder will be kept. Run Calibrate again after repairing detections. SAM3 will not start automatically.`;
 if(action==='clear-outputs'&&!confirm(message))return;
 state.pending=true;updateCameraRepair(scope);
 try{
   const base=scope==='tracking'?'/api/tracking-run/camera/':'/api/calibration/camera/';
   const r=await fetch(base+action,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(scope==='tracking'?{run,camera}:{camera})});
   const d=await r.json().catch(()=>({}));
   if(!r.ok){flash(d.error||'Could not start the camera action.');return;}
   since=0;document.getElementById('log').textContent='';pollLog();
 }catch(e){flash('Could not start the camera action. Check the connection and try again.');}
 finally{state.pending=false;refresh();}
}
async function runFor(stage,source){
 const payload={stage,source};
 if(source==='dataset'||source==='mocap_raw')payload.only=selected(source);
 if(source==='tracking')payload.tracking_run=activeTrackingRun;
 const r=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify(payload)});
 if(r.status===409){flash('A job is already running.');return;}
 if(!r.ok){const d=await r.json().catch(()=>({}));flash(d.error||'Could not start.');return;}
 if(stage==='calibrate'&&source==='upload')calibrationDownloadArmed=true;
 since=0;document.getElementById('log').textContent='';pollLog();refresh();
}
async function downloadCalibrationJson(){
 const r=await fetch('/api/calibration');
 const d=await r.json().catch(()=>({}));
 if(!r.ok||!d.cameras||!d.cameras.length){flash('Calibration finished, but no calibration JSON was available to download.');return;}
 const stamp=new Date().toISOString().replace(/[:.]/g,'-');
 const blob=new Blob([JSON.stringify(d,null,2)],{type:'application/json'});
 const a=document.createElement('a');
 a.href=URL.createObjectURL(blob);
 a.download='firetrack_camera_calibration_'+stamp+'.json';
 document.body.appendChild(a);
 a.click();
 a.remove();
 setTimeout(()=>URL.revokeObjectURL(a.href),1000);
 flash('Downloaded camera intrinsics/extrinsics JSON.');
}
async function pollLog(){polling=true;
 let r; try{ r=await(await fetch('/api/log?since='+since)).json(); }catch(e){ polling=false; return; }
 if(r.lines&&r.lines.length){since=r.next;const l=document.getElementById('log');
   l.textContent+=r.lines.join('\\n')+'\\n';l.scrollTop=l.scrollHeight;}
 updateProgress({status:r.status,progress:r.progress});
 if(r.status==='running'){setTimeout(pollLog,1000);}else{
   polling=false;refresh();
   if(r.name==='upload:clear-calibration')reloadResults();
   if(mode==='upload'&&(r.name||'').startsWith('upload:')&&(r.name||'').endsWith(':clear-camera-outputs'))reloadResults();
   if(mode==='upload'&&(r.name||'').startsWith('tracking:')&&[':clear-outputs',':clear-camera-outputs',':detect',':reference',':triangulate',':classify'].some(suffix=>(r.name||'').endsWith(suffix)))reloadResults();
 }
}
function flash(msg){const l=document.getElementById('log');l.textContent+='! '+msg+'\\n';l.scrollTop=l.scrollHeight;}
function resultsUrl(){
 return mode==='upload'?'/results?mode=nomocap&embedded=1&run='+encodeURIComponent(activeTrackingRun):'/results';
}
function syncResults(){
 document.getElementById('results-run').textContent=activeTrackingRun||'No tracking run selected';
 const key=mode==='upload'?'upload:'+activeTrackingRun:'legacy';
 if(key===resultsKey)return;
 resultsKey=key;reloadResults();
}
function reloadResults(){const f=document.getElementById('results-frame');if(f){const url=resultsUrl();f.src=url+(url.includes('?')?'&':'?')+'t='+Date.now();}}

/* 5-27 dataset uploads with progress */
function uploadDatasetZip(file,progressId='ds-progress',after='Run Format next.'){return new Promise((res)=>{
 const id='zip'+Math.abs([...file.name].reduce((a,c)=>a*31+c.charCodeAt(0)|0,13));
 const box=document.getElementById(progressId);
 box.insertAdjacentHTML('beforeend',`<div style="margin:8px 0"><div style="font-family:var(--mono);font-size:12px;color:var(--muted)">${file.name}</div><div class="prog"><i id="${id}"></i></div></div>`);
 const x=new XMLHttpRequest();x.open('POST','/api/dataset-upload-zip?name='+encodeURIComponent(file.name));
 x.upload.onprogress=e=>{if(e.lengthComputable)document.getElementById(id).style.width=(100*e.loaded/e.total)+'%';};
 x.onload=()=>{const bar=document.getElementById(id);bar.style.width='100%';bar.style.background=x.status<300?'var(--green)':'var(--error,var(--ember))';
   if(x.status<300){let d={};try{d=JSON.parse(x.responseText||'{}')}catch(_){};flash('Zip extracted: '+(d.extracted||0)+' files. '+after);}
   else flash('Zip upload failed: '+file.name);res();};
 x.onerror=()=>{flash('Zip upload failed: '+file.name);res();};x.send(file);});}
async function clearDataset(){
 if(!confirm('Clear uploaded 5-27 files and generated dataset outputs?'))return;
 await fetch('/api/dataset/clear',{method:'POST'});
 document.getElementById('ds-progress').textContent='';
 refresh();
}
document.getElementById('dataset-zip').addEventListener('change',async e=>{
 const f=e.target.files[0]; if(!f)return;
 await uploadDatasetZip(f); e.target.value=''; refresh();
});
document.getElementById('raw-dataset-zip').addEventListener('change',async e=>{
 const f=e.target.files[0]; if(!f)return;
 await uploadDatasetZip(f,'raw-progress','Annotate or run Detect next; this path skips Format.'); e.target.value=''; refresh();
});

function uploadNoMocapZip(file,bucket){return new Promise((res)=>{
 const id='nz'+Math.abs([...file.name].reduce((a,c)=>a*31+c.charCodeAt(0)|0,17));
 const box=document.getElementById(bucket==='tracking'?'tracking-progress':'up-progress');
 box.insertAdjacentHTML('beforeend',`<div style="margin:8px 0"><div style="font-family:var(--mono);font-size:12px;color:var(--muted)">${file.name}</div><div class="prog"><i id="${id}"></i></div></div>`);
 const run=bucket==='tracking'?trackingRunNameForFile(file):'';
 const x=new XMLHttpRequest();x.open('POST','/api/nomocap-upload-zip?bucket='+bucket+'&name='+encodeURIComponent(file.name)+'&run='+encodeURIComponent(run));
 x.upload.onprogress=e=>{if(e.lengthComputable)document.getElementById(id).style.width=(100*e.loaded/e.total)+'%';};
 x.onload=()=>{const bar=document.getElementById(id);bar.style.width='100%';bar.style.background=x.status<300?'var(--green)':'var(--error,var(--ember))';
   if(x.status<300){let d={};try{d=JSON.parse(x.responseText||'{}')}catch(_){};flash((bucket==='tracking'?'Tracking':'Calibration')+' zip uploaded: '+(d.extracted||0)+' files, '+(d.indexed||0)+' clips indexed.');}
   else flash('No-Mocap zip upload failed: '+file.name);res();};
 x.onerror=()=>{flash('No-Mocap zip upload failed: '+file.name);res();};x.send(file);});}
document.getElementById('cal-run-zip').addEventListener('change',async e=>{
 const f=e.target.files[0]; if(!f)return;
 await uploadNoMocapZip(f,'calibration'); e.target.value=''; refresh();
});
document.getElementById('tracking-run-zip').addEventListener('change',async e=>{
 const f=e.target.files[0]; if(!f)return;
 await uploadNoMocapZip(f,'tracking'); e.target.value=''; refresh();
});

/* calibration upload */
document.getElementById('calfile').addEventListener('change',async e=>{
 const f=e.target.files[0]; if(!f)return;
 let text; try{ text=await f.text(); JSON.parse(text); }catch(_){ flash('Calibration is not valid JSON.'); return; }
 const r=await fetch('/api/calibration',{method:'POST',headers:{'Content-Type':'application/json'},body:text});
 const d=await r.json().catch(()=>({}));
 if(!r.ok){flash('Calibration rejected: '+(d.error||'unknown'));}
 else{flash('Calibration loaded: '+d.n_cameras+' cameras.');}
 e.target.value=''; refresh();
});

async function annotate(source,camera=''){
 const payload={source}; if(source==='tracking')payload.tracking_run=activeTrackingRun;
 const r=await fetch('/api/clicks/select',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
 if(!r.ok){flash('Could not open the annotator.');return;}
 document.getElementById('annot-panel').classList.remove('hidden');
 document.getElementById('annot').src='/clicks?t='+Date.now()+((source==='tracking'||source==='upload')&&camera?'&camera='+encodeURIComponent(camera):'');
 document.getElementById('annot-panel').scrollIntoView({behavior:'smooth',block:'start'});
}
function closeAnnot(){document.getElementById('annot-panel').classList.add('hidden');
 document.getElementById('annot').src='about:blank';refresh();}

/* render @ICON tokens in button labels into svg */
document.querySelectorAll('button[data-st],button[data-keep]').forEach(b=>{
 b.innerHTML=b.innerHTML.replace(/@\\w+/,m=>svg('',ICONS[m]||'M5 4l14 8-14 8z'));});
refresh();setInterval(refresh,2500);
</script>
</body></html>"""
