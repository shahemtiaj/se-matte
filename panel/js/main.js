/* SE Matte panel UI */
(function () {
  'use strict';
  var B = window.SEB;
  var $ = function (id) { return document.getElementById(id); };
  var SESSION = Math.random().toString(36).slice(2, 10);
  var LS = window.localStorage;

  function jget(k, d) { try { var v = JSON.parse(LS.getItem(k)); return v === null ? d : v; } catch (e) { return d; } }

  var S = {
    engine: 'off', engineInfo: null, startedEngine: false, startAt: 0,
    layer: null,             // info about the locked AE layer
    keysByLayer: {},         // layerKey → { frame: {points:[[x,y]], labels:[1|0]} }
    keys: {},
    frame: null, wantFrame: null, img: null, loadingFrame: false, pendingFrame: null,
    mask: null,              // { frame, img, ov, cut } – latest selection / Fast preview for a frame
    maskSeq: 0, fastSeq: 0, fastT: null, busy: 0,
    tool: 'add', view: LS.getItem('sem.view') || 'overlay',
    mode: LS.getItem('sem.mode') || 'fast', quality: 'standard',
    jobs: [], jobSig: {}, busyPoll: false,
    imported: new Set(jget('sem.imported', [])),
    hidden: new Set(jget('sem.hidden', [])),
    autoImport: new Set()     // jobs submitted from this panel session
  };

  /* ---------------------------------------------------------------- toast */
  var toastT;
  function toast(msg, kind, ms) {
    var t = $('toast');
    t.textContent = msg;
    t.className = kind || '';
    clearTimeout(toastT);
    toastT = setTimeout(function () { t.className = 'hidden'; }, ms || (kind === 'bad' ? 7000 : 3500));
  }
  $('toast').onclick = function () { this.className = 'hidden'; };

  function fmtTime(s) {
    s = Math.max(0, Math.round(s));
    var h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60), r = s % 60;
    if (h) return h + 'h ' + (m < 10 ? '0' : '') + m + 'm';
    return m ? m + 'm ' + (r < 10 ? '0' : '') + r + 's' : r + 's';
  }
  function fmtDur(s) { s = Math.floor(s + 0.001); var m = Math.floor(s / 60), r = s % 60; return m + ':' + (r < 10 ? '0' : '') + r; }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }

  /* ---------------------------------------------------------------- engine */
  function engineReady() { return S.engine === 'on' || S.engine === 'busy'; }

  function setEngine(state, text) {
    var was = S.engine;
    S.engine = state;
    var p = $('engine');
    p.className = 'status ' + state;
    p.querySelector('span').textContent = text;
    updateRun();
    // the engine just came up: load the preview that was waiting for it
    if (!(was === 'on' || was === 'busy') && engineReady() && S.layer && !S.img) {
      S.loadingFrame = false;
      loadFrame(S.wantFrame !== null ? S.wantFrame : S.layer.current);
    }
  }

  function pollEngine() {
    B.api('/health', undefined, 2000).then(function (h) {
      S.engineInfo = h;
      var dev = h.device.kind === 'mps' ? h.device.name : h.device.kind.toUpperCase();
      var q = h.queued ? ' · ' + h.queued + ' queued' : '';
      setEngine(h.busy ? 'busy' : 'on', (h.busy ? 'Working' : 'Ready') + ' · ' + dev + q);
      if (h.fast === false && S.mode === 'fast') setMode('pro');
    }, function () {
      if (S.engine === 'starting' && Date.now() - S.startAt < 90000) return;
      if (B.hasNode && !S.startedEngine) { startEngine(); return; }
      setEngine('off', 'Engine offline – click to start');
    });
  }

  function startEngine() {
    S.startedEngine = true;
    S.startAt = Date.now();
    setEngine('starting', 'Starting engine…');
    B.startEngine().catch(function (e) {
      setEngine('off', 'Engine offline');
      toast(e.message, 'bad');
    });
  }
  $('engine').onclick = function () { if (S.engine === 'off') startEngine(); };

  /* ---------------------------------------------------------------- settings */
  var SLIDERS = ['choke', 'feather', 'contrast', 'temporal'];
  var SETTINGS = ['range', 'decon'].concat(SLIDERS);
  function loadSettings() {
    var s = jget('sem.settings', {});
    SETTINGS.forEach(function (k) {
      var el = $(k);
      if (s[k] === undefined) return;
      if (el.type === 'checkbox') el.checked = s[k]; else el.value = s[k];
    });
    setQuality(s.quality === 'max' ? 'max' : 'standard', true);
    SLIDERS.forEach(function (k) { $(k + 'V').textContent = $(k).value; paintSlider($(k)); });
    if (s.lookOpen) $('look').open = true;
  }
  function saveSettings() {
    var s = { quality: S.quality, lookOpen: $('look').open };
    SETTINGS.forEach(function (k) { var el = $(k); s[k] = el.type === 'checkbox' ? el.checked : el.value; });
    LS.setItem('sem.settings', JSON.stringify(s));
    updateEstimate();
  }
  SETTINGS.forEach(function (k) {
    $(k).addEventListener('input', function () {
      if ($(k + 'V')) $(k + 'V').textContent = $(k).value;
      saveSettings();
      if (S.mode === 'fast' && SLIDERS.indexOf(k) >= 0 && k !== 'temporal') requestFastMask(true);
    });
  });
  function paintSlider(el) {
    var lo = +el.min, hi = +el.max, p = (el.value - lo) / (hi - lo) * 100, z = lo < 0 ? (-lo) / (hi - lo) * 100 : 0;
    el.style.setProperty('--a', Math.min(p, z) + '%');
    el.style.setProperty('--b', Math.max(p, z) + '%');
  }
  SLIDERS.forEach(function (k) {
    $(k).addEventListener('input', function () { paintSlider(this); });
    $(k).addEventListener('dblclick', function () { this.value = 0; this.dispatchEvent(new Event('input')); });
  });
  $('btnLookReset').onclick = function (e) {
    e.preventDefault(); e.stopPropagation();
    SLIDERS.forEach(function (k) { $(k).value = 0; $(k + 'V').textContent = '0'; paintSlider($(k)); });
    saveSettings();
    if (S.mode === 'fast') requestFastMask(true);
  };
  $('look').addEventListener('toggle', saveSettings);

  function setQuality(q, quiet) {
    S.quality = q;
    document.querySelectorAll('#qualitySeg button').forEach(function (b) { b.classList.toggle('on', b.getAttribute('data-q') === q); });
    if (quiet) return;
    saveSettings();
    if (S.mode === 'fast') requestFastMask(true);
  }
  document.querySelectorAll('#qualitySeg button').forEach(function (b) {
    b.onclick = function () { setQuality(b.getAttribute('data-q')); };
  });

  function look() {
    return { choke: +$('choke').value, feather: +$('feather').value, contrast: +$('contrast').value,
      temporal: +$('temporal').value, decontaminate: $('decon').checked };
  }

  /* ---------------------------------------------------------------- layer */
  function layerKey(L) { return L.path + '|' + L.compId + '|' + L.sourceId; }

  function useLayer(L) {
    if (S.layer) S.keysByLayer[layerKey(S.layer)] = S.keys;
    var same = S.layer && layerKey(S.layer) === layerKey(L);
    S.layer = L;
    S.keys = S.keysByLayer[layerKey(L)] || {};
    document.body.classList.remove('no-layer');
    $('lName').textContent = L.layerName;
    $('lName').title = L.path;
    var dur = (L.outFrame - L.inFrame) / L.fps;
    var meta = [L.width + '×' + L.height, (+L.fps.toFixed(3)) + ' fps', fmtDur(dur)];
    $('lMeta').innerHTML = meta.map(esc).join('<i>·</i>');
    $('lMeta').title = (L.outFrame - L.inFrame) + ' frames' + (L.compName ? ' · in "' + L.compName + '"' : '');
    setProcessed(!!L.processed);
    fitCanvas();
    if (L.hasEffects) toast('Note: the AI sees the original footage, not the effects on this layer.', '', 5000);
    if (!same) { S.frame = null; S.img = null; S.mask = null; }
    var keyFrames = sortedKeys();
    var f = (L.current >= L.inFrame && L.current < L.outFrame) ? L.current
      : (keyFrames.length ? keyFrames[0] : L.inFrame);
    loadFrame(f);
    renderKeys();
    updateRun();
    updateEstimate();
  }

  function pickLayer() {
    if (B.inAE) {
      B.jsx('SEM_selectedLayer').then(function (r) {
        if (!r.ok) { toast(r.error, 'bad'); return; }
        useLayer(r);
      });
      return;
    }
    // dev mode: a file path stands in for the AE layer
    var p = $('devPathInput').value.trim();
    if (!p) { toast('Dev mode: paste a video file path first.', 'bad'); return; }
    LS.setItem('sem.devPath', p);
    B.api('/probe?path=' + encodeURIComponent(p)).then(function (i) {
      useLayer({ compId: 0, compName: '', layerIndex: 1, layerName: p.split(/[\\/]/).pop(), sourceId: 0,
        path: p, width: i.width, height: i.height, fps: i.fps, frames: i.frames,
        inFrame: 0, outFrame: i.frames, current: 0, stretch: 100 });
    }, function (e) { toast(e.message, 'bad'); });
  }
  $('btnLayer').onclick = pickLayer;
  $('btnLayer2').onclick = pickLayer;

  function layerRef() {
    var L = S.layer;
    return { compId: L.compId, compName: L.compName, layerIndex: L.layerIndex, layerName: L.layerName, sourceId: L.sourceId };
  }

  /* ---------------------------------------------------------------- preview */
  var cv = $('cv'), ctx = cv.getContext('2d');

  /* scale the canvas to fit the stage (letterboxed, never cropped) */
  function fitCanvas() {
    var st = $('stage');
    var bw = st.clientWidth - 2, bh = st.clientHeight - 2;
    if (bw <= 0 || bh <= 0 || !cv.width || !cv.height) return;
    var k = Math.min(bw / cv.width, bh / cv.height);
    cv.style.width = Math.floor(cv.width * k) + 'px';
    cv.style.height = Math.floor(cv.height * k) + 'px';
  }
  var refit = function () { fitCanvas(); if (S.mode === "pro") draw(); };
  if (window.ResizeObserver) new ResizeObserver(refit).observe($("stage"));
  else window.addEventListener("resize", refit);

  function stageMsg(t) {
    var m = $('stageMsg');
    if (t) { m.textContent = t; m.classList.remove('hidden'); } else m.classList.add('hidden');
  }

  var spinT;
  function busy(d) {
    S.busy = Math.max(0, S.busy + d);
    clearTimeout(spinT);
    if (S.busy) spinT = setTimeout(function () { $('stageSpin').classList.remove('hidden'); }, 180);
    else $('stageSpin').classList.add('hidden');
  }

  function loadFrame(f) {
    var L = S.layer;
    if (!L) return;
    f = Math.max(0, Math.min(L.frames - 1, f | 0));
    S.wantFrame = f;
    if (S.loadingFrame) { S.pendingFrame = f; return; }
    if (f === S.frame && S.img) return;
    if (!engineReady()) { stageMsg('Waiting for the engine…'); return; }
    S.loadingFrame = true;
    if (S.mode === 'fast') requestFastMask(false);   // in parallel with the frame itself
    busy(1);
    var img = new Image();
    img.onload = function () {
      busy(-1);
      S.loadingFrame = false;
      S.img = img; S.frame = f;
      if (S.thumbFor !== L.path) { S.thumbFor = L.path; $('layerThumb').style.backgroundImage = 'url("' + img.src + '")'; $('layerThumb').classList.add('has'); }
      if (cv.width !== img.naturalWidth || cv.height !== img.naturalHeight) { cv.width = img.naturalWidth; cv.height = img.naturalHeight; fitCanvas(); }
      if (S.mode === 'pro') { stageMsg(''); if (S.keys[f]) requestMask(); else S.mask = null; }
      updateFrameLabel();
      draw();
      renderKeys();
      var p = S.pendingFrame; S.pendingFrame = null;
      if (p !== null && p !== f) loadFrame(p);
    };
    img.onerror = function () {
      busy(-1);
      S.loadingFrame = false; S.pendingFrame = null;
      if (!engineReady()) stageMsg('Waiting for the engine…');
      else stageMsg('Could not read frame ' + f);
    };
    img.src = B.url('/frame', { path: L.path, frame: f, max: 960 });
  }

  function updateFrameLabel() {
    var L = S.layer, f = S.frame;
    if (!L || f === null) return;
    var out = f < L.inFrame || f >= L.outFrame;
    var ks = sortedKeys();
    $('frameLabel').textContent = 'Frame ' + f + (out ? ' · outside the layer' : '') +
      (S.mode === 'pro' && S.keys[f] ? ' · ' + (ks[0] === f ? 'start' : 'correction') : '');
  }

  /* mask image → cached canvases for the two preview views */
  function maskData(m) {
    if (m.data && m.data.width === cv.width && m.data.height === cv.height) return m.data;
    var c = document.createElement('canvas');
    c.width = cv.width; c.height = cv.height;
    var x = c.getContext('2d');
    x.drawImage(m.img, 0, 0, c.width, c.height);
    m.data = x.getImageData(0, 0, c.width, c.height);
    m.ov = m.cut = null;
    return m.data;
  }
  function overlayCanvas(m) {
    var md = maskData(m);
    if (m.ov) return m.ov;
    var c = document.createElement('canvas');
    c.width = md.width; c.height = md.height;
    var x = c.getContext('2d'), d = x.createImageData(md.width, md.height), o = d.data, s = md.data;
    var t = (getComputedStyle(document.documentElement).getPropertyValue('--m') || '22,190,215').split(',').map(Number);
    for (var i = 0; i < s.length; i += 4) {
      var a = s[i] / 255;                     // 1 = subject
      // dim the background, light blue tint on the subject
      o[i] = t[0] * a + 8 * (1 - a); o[i + 1] = t[1] * a + 9 * (1 - a); o[i + 2] = t[2] * a + 14 * (1 - a);
      o[i + 3] = (1 - a) * 175 + a * 42;
    }
    x.putImageData(d, 0, 0);
    return (m.ov = c);
  }
  function cutoutCanvas(m) {
    var md = maskData(m);
    if (m.cut && m.cutImg === S.img) return m.cut;
    var c = document.createElement('canvas');
    c.width = md.width; c.height = md.height;
    var x = c.getContext('2d');
    x.drawImage(S.img, 0, 0, c.width, c.height);
    var d = x.getImageData(0, 0, c.width, c.height), o = d.data, s = md.data;
    for (var i = 3; i < o.length; i += 4) o[i] = s[i - 3];
    x.putImageData(d, 0, 0);
    m.cutImg = S.img;
    return (m.cut = c);
  }

  function draw() {
    if (!S.img) return;
    var m = S.mask && S.mask.frame === S.frame && S.view !== 'original' ? S.mask : null;
    if (m && S.view === 'cutout') {
      ctx.clearRect(0, 0, cv.width, cv.height);
      ctx.drawImage(cutoutCanvas(m), 0, 0);
    } else {
      ctx.drawImage(S.img, 0, 0, cv.width, cv.height);
      if (m) ctx.drawImage(overlayCanvas(m), 0, 0);
    }
    var k = S.keys[S.frame];
    if (!k || S.mode === 'fast') return;
    var sx = cv.width / S.layer.width, sy = cv.height / S.layer.height;
    var px = cv.width / (cv.clientWidth || cv.width);    // canvas pixels per screen pixel
    var r = 6.5 * px;
    k.points.forEach(function (p, i) {
      ctx.beginPath();
      ctx.arc(p[0] * sx, p[1] * sy, r, 0, Math.PI * 2);
      ctx.fillStyle = k.labels[i] ? '#36d17a' : '#ff5a4f';
      ctx.fill();
      ctx.lineWidth = Math.max(1.5, r / 3);
      ctx.strokeStyle = '#fff';
      ctx.stroke();
      ctx.fillStyle = '#fff';
      ctx.font = 'bold ' + Math.round(r * 1.6) + 'px sans-serif';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(k.labels[i] ? '+' : '−', p[0] * sx, p[1] * sy + 0.5);
    });
  }

  function setMask(frame, b64) {
    var mi = new Image();
    mi.onload = function () { S.mask = { frame: frame, img: mi }; draw(); };
    mi.src = 'data:image/png;base64,' + b64;
  }

  function requestMask() {
    var L = S.layer, f = S.frame, k = S.keys[f];
    var seq = ++S.maskSeq;
    if (!k || !k.points.length) { S.mask = null; draw(); return; }
    if (k.labels.indexOf(1) < 0) { S.mask = null; draw(); stageMsg('Add at least one + click on the subject'); return; }
    var slow = setTimeout(function () { if (seq === S.maskSeq) stageMsg('Selecting… (the first click loads the AI model)'); }, 700);
    busy(1);
    B.api('/segment', { path: L.path, frame: f, points: k.points, labels: k.labels, max: 960 }, 120000).then(function (r) {
      clearTimeout(slow); busy(-1);
      if (seq !== S.maskSeq || f !== S.frame) return;
      stageMsg('');
      setMask(f, r.mask);
    }, function (e) {
      clearTimeout(slow); busy(-1);
      if (seq === S.maskSeq) { stageMsg(''); toast('Selection failed: ' + e.message, 'bad'); }
    });
  }

  /* Fast preview: what Fast mode keeps on this frame, with the edge-look sliders applied */
  var fastInflight = false, fastAgain = false;
  function requestFastMask(debounce) {
    clearTimeout(S.fastT);
    if (debounce) { S.fastT = setTimeout(function () { requestFastMask(false); }, 40); return; }
    var L = S.layer, f = S.wantFrame;
    if (!L || f === null || S.mode !== 'fast' || !engineReady()) return;
    if (fastInflight) { fastAgain = true; return; }   // one at a time; the latest settings win
    fastInflight = true;
    var seq = ++S.fastSeq;
    var lk = look();
    busy(1);
    B.api('/fastpreview', { path: L.path, frame: f, quality: S.quality, max: 960,
      refine: { choke: lk.choke, feather: lk.feather, contrast: lk.contrast } }, 60000).then(function (r) {
      done();
      if (seq !== S.fastSeq || f !== S.wantFrame || S.mode !== 'fast') return;
      stageMsg(r.coverage < 0.002 ? 'No person found on this frame' : '');
      setMask(f, r.mask);
    }, function (e) {
      done();
      if (seq === S.fastSeq) stageMsg(/person/i.test(e.message) ? 'No person found on this frame' : 'Preview failed: ' + e.message);
    });
    function done() {
      busy(-1);
      fastInflight = false;
      if (fastAgain) { fastAgain = false; requestFastMask(false); }
    }
  }

  cv.addEventListener('click', function (e) {
    if (!S.img || !S.layer) return;
    if (S.mode === 'fast') { toast('Fast mode finds people automatically. Switch to Pro to pick a subject yourself.'); return; }
    var r = cv.getBoundingClientRect();
    var x = (e.clientX - r.left) / r.width * S.layer.width;
    var y = (e.clientY - r.top) / r.height * S.layer.height;
    var label = (S.tool === 'remove' || e.altKey) ? 0 : 1;
    var k = S.keys[S.frame] || (S.keys[S.frame] = { points: [], labels: [] });
    k.points.push([Math.round(x * 10) / 10, Math.round(y * 10) / 10]);
    k.labels.push(label);
    draw(); renderKeys(); updateFrameLabel(); updateEstimate();
    requestMask();
  });

  function setTool(t) {
    S.tool = t;
    $('toolAdd').classList.toggle('on', t === 'add');
    $('toolRemove').classList.toggle('on', t === 'remove');
  }
  $('toolAdd').onclick = function () { setTool('add'); };
  $('toolRemove').onclick = function () { setTool('remove'); };

  function undoClick() {
    var k = S.keys[S.frame];
    if (!k) return;
    k.points.pop(); k.labels.pop();
    if (!k.points.length) delete S.keys[S.frame];
    afterKeysChanged();
  }
  $('btnUndo').onclick = undoClick;
  $('btnClear').onclick = function () { delete S.keys[S.frame]; afterKeysChanged(); };

  function afterKeysChanged() {
    S.mask = null;
    draw(); renderKeys(); updateFrameLabel(); updateEstimate();
    stageMsg('');
    requestMask();
  }

  function setView(v) {
    S.view = v;
    LS.setItem('sem.view', v);
    document.querySelectorAll('#viewSeg button').forEach(function (b) { b.classList.toggle('on', b.getAttribute('data-view') === v); });
    draw();
  }
  document.querySelectorAll('#viewSeg button').forEach(function (b) {
    b.onclick = function () { setView(b.getAttribute('data-view')); };
  });

  /* ---------------------------------------------------------------- keyframe chips */
  function sortedKeys() {
    return Object.keys(S.keys).map(Number).filter(function (f) { return S.keys[f].points.length; })
      .sort(function (a, b) { return a - b; });
  }

  function renderKeys() {
    var box = $('keys');
    var ks = sortedKeys();
    box.innerHTML = '';
    ks.forEach(function (f, i) {
      var k = S.keys[f];
      var el = document.createElement('span');
      el.className = 'key' + (f === S.frame ? ' cur' : '') + (i === 0 ? ' first' : '');
      el.title = (i === 0 ? 'Start frame' : 'Correction') + ' · ' + k.points.length + ' click(s) · click to jump';
      el.innerHTML = 'Frame ' + f + ' <span class="muted">· ' + k.points.length + '</span><span class="x" title="Delete">×</span>';
      el.onclick = function (e) {
        if (e.target.classList.contains('x')) { delete S.keys[f]; afterKeysChanged(); return; }
        goTo(f);
      };
      box.appendChild(el);
    });
  }

  function goTo(f) {
    if (B.inAE && S.layer && $('follow').checked) {
      B.jsx('SEM_gotoFrame', [layerRef(), String(f)]).then(function () { loadFrame(f); });
    } else loadFrame(f);
  }

  /* follow the AE playhead */
  setInterval(function () {
    if (!B.inAE || !S.layer || !$('follow').checked || S.busyPoll || document.hidden) return;
    S.busyPoll = true;
    B.jsx('SEM_layerFrame', [layerRef()]).then(function (r) {
      S.busyPoll = false;
      if (r.ok && r.active) {
        if (r.layerIndex) S.layer.layerIndex = r.layerIndex;
        if (r.frame !== S.wantFrame) loadFrame(r.frame);
      }
    });
  }, 300);

  /* keyboard: ← → step frames in dev mode, Cmd/Ctrl+Z undo a click (Pro) */
  document.addEventListener('keydown', function (e) {
    if (e.target.tagName === 'INPUT' && e.target.type === 'text' || e.target.tagName === 'SELECT') return;
    if (e.key === 'z' && (e.metaKey || e.ctrlKey) && S.mode === 'pro' && S.keys[S.frame]) { undoClick(); e.preventDefault(); }
    if (!B.inAE && S.layer && (e.key === 'ArrowRight' || e.key === 'ArrowLeft') && e.target.type !== 'range') {
      loadFrame((S.wantFrame || 0) + (e.key === 'ArrowRight' ? 1 : -1) * (e.shiftKey ? 10 : 1));
    }
  });

  /* ---------------------------------------------------------------- Fast / Pro */
  function setMode(m) {
    if (m === 'fast' && S.engineInfo && S.engineInfo.fast === false) m = 'pro';
    S.mode = m;
    LS.setItem('sem.mode', m);
    document.body.classList.toggle('mode-fast', m === 'fast');
    document.body.classList.toggle('mode-pro', m === 'pro');
    document.querySelectorAll('#modeSeg button').forEach(function (b) { b.classList.toggle('on', b.getAttribute('data-mode') === m); });
    S.mask = null; stageMsg('');
    if (S.img) { draw(); if (m === 'fast') requestFastMask(false); else if (S.keys[S.frame]) requestMask(); }
    updateFrameLabel();
    updateEstimate();
  }
  document.querySelectorAll('#modeSeg button').forEach(function (b) {
    b.onclick = function () {
      if (b.getAttribute('data-mode') === 'fast' && S.engineInfo && S.engineInfo.fast === false) {
        toast('Fast mode is not available on this computer yet (Mac only). Use Pro.', 'bad'); return;
      }
      setMode(b.getAttribute('data-mode'));
    };
  });

  /* ---------------------------------------------------------------- run */
  function range() {
    var L = S.layer;
    return $('range').value === 'clip' ? [0, L.frames] : [L.inFrame, L.outFrame];
  }

  function updateRun() {
    $('btnRun').disabled = !engineReady() || !S.layer;
  }

  function updateEstimate() {
    if (!S.layer) return;
    var r = range(), n = r[1] - r[0];
    var max = S.quality === 'max';
    var mp = (S.layer.width * S.layer.height) / 2073600;   // relative to 1080p
    // measured on an M4: Fast 48 / 27 fps at 1080p; Pro ~1.75 / 0.7 fps (its stages run in parallel)
    var sec = S.mode === 'fast' ? n / (max ? 27 : 48) * Math.max(0.6, mp) + 4
                                : n / (max ? 0.7 : 1.75) * Math.max(0.6, mp) + 20;
    var gb = n * 1.05 * mp / 1000;   // ProRes 4444 with alpha: ~1 MB per 1080p frame (measured)
    var size = gb < 1 ? Math.max(1, Math.round(gb * 1000)) + ' MB' : gb.toFixed(1) + ' GB';
    var it = function (ic, t, tip) { return '<span title="' + tip + '"><svg class="ic"><use href="#i-' + ic + '"/></svg>' + t + '</span>'; };
    $('estimate').innerHTML = it('film', n.toLocaleString() + ' frames', 'Frames to process') +
      it('clock', '~' + fmtTime(sec), 'Estimated time on this computer') + it('disk', size, 'ProRes 4444 file with alpha');
  }

  $('btnRun').onclick = function () {
    var L = S.layer;
    if (!L) return;
    var r = range();
    var meta = { ref: layerRef(), label: L.layerName, panel: SESSION, comp: L.compName, mode: S.mode };
    if (S.mode === 'fast') { submit({ source: L.path, start: r[0], end: r[1], mode: 'fast', quality: S.quality, refine: look(), meta: meta }); return; }
    var ks = sortedKeys().filter(function (f) { return f >= r[0] && f < r[1]; });
    var bad = ks.filter(function (f) { return S.keys[f].labels.indexOf(1) < 0; });
    if (bad.length) { toast('Frame ' + bad[0] + ' has only − clicks. Add a + click on the subject there.', 'bad'); return; }
    if (!ks.length) {
      toast(sortedKeys().length ? 'Your clicks are outside the processed range. Click on a frame inside the layer.'
        : 'Click on the subject in the preview first.', 'bad');
      return;
    }
    submit({ source: L.path, start: r[0], end: r[1], mode: 'pro',
      keyframes: ks.map(function (f) { return { frame: f, points: S.keys[f].points, labels: S.keys[f].labels }; }),
      quality: S.quality, outputs: ['prores4444'], refine: look(), meta: meta });
  };

  function submit(body) {
    $('btnRun').disabled = true;
    B.api('/jobs', body).then(function (j) {
      S.autoImport.add(j.id);
      toast('Added to the queue – you can keep working in AE.', 'good');
      pollJobs();
    }, function (e) { toast(e.message, 'bad'); }).then(updateRun);
  }

  /* ---------------------------------------------------------------- queue */
  function pollJobs() {
    B.api('/jobs', undefined, 4000).then(function (list) {
      S.jobs = list;
      renderJobs();
      list.forEach(function (j) {
        if (j.state === 'done' && S.autoImport.has(j.id) && !S.imported.has(j.id)) importJob(j);
        if (j.state !== 'done' && j.state !== 'queued' && j.state !== 'running') S.autoImport.delete(j.id);
      });
    }, function () {});
  }

  function persist() {
    LS.setItem('sem.imported', JSON.stringify(Array.from(S.imported).slice(-200)));
    LS.setItem('sem.hidden', JSON.stringify(Array.from(S.hidden).slice(-200)));
  }

  function importJob(j) {
    S.autoImport.delete(j.id);   // auto-import is tried once; after that "Apply to layer" retries
    var res = j.result || {};
    if (!B.inAE || !j.meta || !j.meta.ref || !res.prores4444) { renderJobs(); return; }
    S.imported.add(j.id); persist();
    B.jsx('SEM_import', [{ ref: j.meta.ref, prores: res.prores4444, start: res.start }]).then(function (r) {
      if (r.ok) {
        toast('Background removed on "' + r.layer + '"  (Cmd/Ctrl+Z to undo)', 'good', 5000);
        if (S.layer && S.layer.sourceId === j.meta.ref.sourceId && S.layer.compId === j.meta.ref.compId) setProcessed(true);
      } else { S.imported.delete(j.id); persist(); toast(r.error, 'bad', 10000); }
      S.jobSig = {}; renderJobs();
    });
  }

  function setProcessed(on) {
    if (S.layer) S.layer.processed = on;
    $('processedRow').classList.toggle('hidden', !on);
  }
  $('btnRestore').onclick = function () {
    if (!S.layer) return;
    B.jsx('SEM_restore', [layerRef()]).then(function (r) {
      if (r.ok) { setProcessed(false); toast('Original footage restored.', 'good'); }
      else toast(r.error, 'bad');
    });
  };

  function jobHTML(j, now) {
    var pct = Math.round(j.progress * 100);
    var info = '';
    if (j.state === 'running' && j.started) {
      var el = now - j.started;
      info = pct + '%' + (j.progress > 0.04 ? ' · ' + fmtTime(el / j.progress - el) + ' left' : ' · ' + fmtTime(el));
    } else if (j.state === 'done' && j.result) info = (j.end - j.start) + ' frames · ' + fmtTime(j.result.seconds);
    var label = (j.meta && j.meta.label) || j.source.split(/[\\/]/).pop();
    var mode = '<span class="tag">' + ((j.meta && j.meta.mode === 'fast') ? 'FAST' : 'PRO') + '</span>';
    var acts = '';
    if (j.state === 'queued' || j.state === 'running') acts += '<button class="link small" data-a="cancel">Cancel</button>';
    if (j.state === 'done' && B.inAE && j.meta && j.meta.ref && j.meta.ref.compId) acts += '<button class="link small" data-a="import">' + (S.imported.has(j.id) ? 'Apply again' : 'Apply to layer') + '</button>';
    if (j.state === 'done' && j.result && B.hasNode) acts += '<button class="link small" data-a="show">Show</button>';
    if (j.state !== 'queued' && j.state !== 'running') acts += '<button class="link x" data-a="hide" title="Remove from the list">×</button>';
    var msg = j.state === 'done' ? (S.imported.has(j.id) ? 'Applied to the layer' : 'Done') : j.message;
    return '<div class="top"><i class="st">' + (j.state === 'done' ? '✓' : (j.state === 'error' || j.state === 'cancelled') ? '!' : '') + '</i>' +
      '<span class="name" title="' + esc(j.source) + ' · frames ' + j.start + '–' + (j.end - 1) + '">' + mode + esc(label) + '</span>' +
      '<span class="acts">' + acts + '</span></div>' +
      '<div class="bar"><i style="width:' + (j.state === 'error' || j.state === 'cancelled' ? 100 : pct) + '%"></i></div>' +
      '<div class="msg small"><span class="muted" title="' + esc(msg) + '">' + esc(msg) + '</span><span class="muted">' + info + '</span></div>' +
      (j.error ? '<div class="err small">' + esc(j.error) + '</div>' : '');
  }

  function renderJobs() {
    var vis = S.jobs.filter(function (j) { return !S.hidden.has(j.id); }).reverse();
    $('secQueue').classList.toggle('hidden', !vis.length);
    var lastDone = S.jobs.filter(function (j) { return j.state === 'done' && j.result && j.result.prores4444; }).pop();
    $('openFolder').classList.toggle('hidden', !(lastDone && B.hasNode));
    $('openFolder').onclick = function (e) { e.preventDefault(); if (lastDone) B.revealPath(lastDone.result.prores4444); };
    var box = $('jobs'), now = Date.now() / 1000;
    var ids = vis.map(function (j) { return j.id; });
    // only touch cards whose content changed (keeps hover states and clicks reliable while polling)
    Array.prototype.slice.call(box.children).forEach(function (el) {
      if (ids.indexOf(el.getAttribute('data-id')) < 0) { box.removeChild(el); delete S.jobSig[el.getAttribute('data-id')]; }
    });
    vis.forEach(function (j, i) {
      var html = jobHTML(j, now);
      var el = box.querySelector('[data-id="' + j.id + '"]');
      if (!el) {
        el = document.createElement('div');
        el.setAttribute('data-id', j.id);
        el.addEventListener('click', function (e) {
          var a = e.target.getAttribute && e.target.getAttribute('data-a');
          if (!a) return;
          var jj = S.jobs.filter(function (x) { return x.id === j.id; })[0] || j;
          if (a === 'cancel') B.api('/jobs/' + jj.id + '/cancel', {}).then(pollJobs, function (er) { toast(er.message, 'bad'); });
          if (a === 'import') importJob(jj);
          if (a === 'show') B.revealPath(jj.result.prores4444 || jj.result.matte || jj.result.png);
          if (a === 'hide') { S.hidden.add(jj.id); persist(); renderJobs(); }
        });
      }
      if (box.children[i] !== el) box.insertBefore(el, box.children[i] || null);
      el.className = 'job ' + j.state;
      if (S.jobSig[j.id] !== html) { el.innerHTML = html; S.jobSig[j.id] = html; }
    });
  }
  $('btnClearDone').onclick = function () {
    S.jobs.forEach(function (j) { if (j.state !== 'queued' && j.state !== 'running') S.hidden.add(j.id); });
    persist(); renderJobs();
  };

  /* ---------------------------------------------------------------- boot */
  if (!B.inAE) {
    $('devPath').classList.remove('hidden');
    $('btnLayer').querySelector('span').textContent = 'Load video';
    $('follow').parentNode.classList.add('hidden');
    $('btnLayer2').classList.add('hidden');
    $('devPathInput').value = LS.getItem('sem.devPath') || '';
  }
  loadSettings();
  setView(S.view);
  setMode(S.mode);
  pollEngine();
  setInterval(pollEngine, 2500);
  pollJobs();
  setInterval(function () {
    var active = S.jobs.some(function (j) { return j.state === 'running' || j.state === 'queued'; });
    if (active || S.autoImport.size) pollJobs();
  }, 1000);
})();
