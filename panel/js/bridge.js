/* SE Matte – bridge to After Effects (ExtendScript) and the local engine process.
   Works inside AE (CEP, with Node) and, for UI development, in a normal browser ("dev mode"). */
(function () {
  'use strict';
  var PORT = 47321;
  var cep = window.__adobe_cep__ || null;
  var node = (typeof require === 'function') ? require : null;

  var B = {
    inAE: !!cep,
    hasNode: !!node,
    base: 'http://127.0.0.1:' + PORT,
    os: (navigator.platform || '').toLowerCase().indexOf('win') === 0 ? 'win' : 'mac'
  };

  /* ---------------- ExtendScript */
  B.jsx = function (fn, args) {
    // args are passed as JSON strings, safely quoted
    var call = fn + '(' + (args || []).map(function (a) {
      return JSON.stringify(typeof a === 'string' ? a : JSON.stringify(a));
    }).join(',') + ')';
    return new Promise(function (resolve) {
      if (!cep) { resolve({ ok: false, error: 'Not running inside After Effects.' }); return; }
      cep.evalScript(call, function (res) {
        try { resolve(JSON.parse(res)); }
        catch (e) { resolve({ ok: false, error: 'AE: ' + res }); }
      });
    });
  };

  /* ---------------- engine http */
  B.api = function (path, body, timeoutMs) {
    var ctl = new AbortController();
    var t = setTimeout(function () { ctl.abort(); }, timeoutMs || 15000);
    var opt = { signal: ctl.signal, cache: 'no-store' };
    if (body !== undefined) {
      opt.method = 'POST';
      opt.headers = { 'Content-Type': 'application/json' };
      opt.body = JSON.stringify(body);
    }
    return fetch(B.base + path, opt).then(function (r) {
      clearTimeout(t);
      return r.json().then(function (j) {
        if (!r.ok) throw new Error((j && j.detail) ? (typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail)) : r.statusText);
        return j;
      });
    }, function (e) { clearTimeout(t); throw e; });
  };
  B.url = function (path, params) {
    var q = Object.keys(params || {}).map(function (k) {
      return encodeURIComponent(k) + '=' + encodeURIComponent(params[k]);
    }).join('&');
    return B.base + path + (q ? '?' + q : '');
  };

  /* ---------------- engine process (needs Node, i.e. inside AE) */
  function panelDir() {
    var p = decodeURIComponent(location.pathname);
    if (/^\/[A-Za-z]:/.test(p)) p = p.slice(1);           // Windows: /C:/... → C:/...
    var path = node('path'), fs = node('fs');
    try { p = fs.realpathSync(p); } catch (e) {}           // dev install is a symlink
    return path.dirname(p);
  }

  B.findEngine = function () {
    if (!node) return null;
    var path = node('path'), fs = node('fs'), os = node('os');
    var home = os.homedir();
    var cands = [];
    try {
      var cfg = JSON.parse(fs.readFileSync(path.join(home, '.se_matte', 'config.json'), 'utf8'));
      if (cfg.engine) cands.push(cfg.engine);
    } catch (e) {}
    var pd = panelDir();
    cands.push(path.join(pd, '..', 'engine'), path.join(pd, 'engine'));
    if (B.os === 'mac') cands.push(path.join(home, 'Library', 'Application Support', 'SE Matte', 'engine'));
    else cands.push(path.join(process.env.APPDATA || home, 'SE Matte', 'engine'));
    for (var i = 0; i < cands.length; i++) {
      var d = path.resolve(cands[i]);
      var py = B.os === 'win' ? path.join(d, '.venv', 'Scripts', 'python.exe') : path.join(d, '.venv', 'bin', 'python');
      if (fs.existsSync(path.join(d, 'semat', 'server.py')) && fs.existsSync(py)) return { dir: d, python: py };
    }
    return null;
  };

  B.startEngine = function () {
    if (!node) return Promise.reject(new Error('Start the engine manually: python -m semat.server'));
    var eng = B.findEngine();
    if (!eng) return Promise.reject(new Error('SE Matte engine not found. Re-run the installer.'));
    var cp = node('child_process'), fs = node('fs'), path = node('path'), os = node('os');
    var logDir = path.join(os.homedir(), '.se_matte');
    try { fs.mkdirSync(logDir, { recursive: true }); } catch (e) {}
    var log = fs.openSync(path.join(logDir, 'engine.log'), 'a');
    var env = Object.assign({}, process.env, { PYTHONUNBUFFERED: '1', SEMATTE_PORT: String(PORT) });
    var child = cp.spawn(eng.python, ['-m', 'semat.server', '--port', String(PORT)], {
      cwd: eng.dir, env: env, detached: true, stdio: ['ignore', log, log], windowsHide: true
    });
    child.unref();
    return Promise.resolve(eng);
  };

  B.revealPath = function (p) {
    if (!node || !p) return;
    var cp = node('child_process');
    if (B.os === 'mac') cp.spawn('open', ['-R', p], { detached: true });
    else cp.spawn('explorer', ['/select,', p], { detached: true });
  };

  /* ---------------- the panel uses its own dark palette; only a light AE UI gets a flag (for future styling) */
  function applySkin() {
    try {
      var c0 = JSON.parse(cep.getHostEnvironment()).appSkinInfo.panelBackgroundColor.color;
      document.documentElement.classList.toggle('ae-light', (c0.red + c0.green + c0.blue) / 3 > 120);
      return;
      var c = JSON.parse(cep.getHostEnvironment()).appSkinInfo.panelBackgroundColor.color;
      var rgb = [c.red, c.green, c.blue].map(Math.round);
      if ((rgb[0] + rgb[1] + rgb[2]) / 3 > 120) return;          // light UI: keep our dark palette
      var sh = function (d) { return 'rgb(' + rgb.map(function (v) { return Math.max(0, Math.min(255, v + d)); }).join(',') + ')'; };
      var st = document.documentElement.style;
      st.setProperty('--bg', sh(0)); st.setProperty('--bg2', sh(8)); st.setProperty('--bg3', sh(18));
      st.setProperty('--line', sh(23)); st.setProperty('--line2', sh(39));
    } catch (e) {}
  }
  if (cep) {
    applySkin();
    try { cep.addEventListener('com.adobe.csxs.events.ThemeColorChanged', applySkin); } catch (e) {}
  }

  window.SEB = B;
})();
