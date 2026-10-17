window.__sg = {
  ensure() {
    if (document.getElementById('__sg_cap')) return;
    const c = document.createElement('div'); c.id = '__sg_cap';
    c.style.cssText = 'position:fixed;left:50%;bottom:24px;transform:translateX(-50%);max-width:80%;padding:12px 20px;' +
      'background:rgba(17,17,17,.85);color:#fff;font:600 18px/1.4 system-ui,sans-serif;border-radius:10px;' +
      'z-index:2147483647;pointer-events:none;display:none';
    const h = document.createElement('div'); h.id = '__sg_hl';
    h.style.cssText = 'position:fixed;border:3px solid #ff3d71;border-radius:8px;box-shadow:0 0 0 9999px rgba(0,0,0,.25);' +
      'transition:all .3s ease;z-index:2147483646;pointer-events:none;display:none';
    document.documentElement.append(c, h);
  },
  caption(t) { this.ensure(); const c = document.getElementById('__sg_cap'); c.textContent = t || ''; c.style.display = t ? 'block' : 'none'; },
  highlight(x, y, w, h) {
    this.ensure(); const e = document.getElementById('__sg_hl');
    Object.assign(e.style, { left: (x - 6) + 'px', top: (y - 6) + 'px', width: (w + 12) + 'px', height: (h + 12) + 'px', display: 'block' });
  },
  cursor(x, y) {
    let c = document.getElementById('__sg_cur');
    if (!c) { c = document.createElement('div'); c.id = '__sg_cur';
      c.style.cssText = 'position:fixed;width:18px;height:18px;border-radius:50%;background:rgba(255,61,113,.9);' +
        'box-shadow:0 0 0 6px rgba(255,61,113,.25);z-index:2147483647;pointer-events:none;transition:all .5s ease;left:0;top:0';
      document.documentElement.append(c); }
    c.style.left = (x - 9) + 'px'; c.style.top = (y - 9) + 'px'; c.style.display = 'block';
  },
  zoom(x, y, s) {
    const d = document.documentElement;
    d.style.transition = 'transform .4s ease'; d.style.transformOrigin = `${x}px ${y}px`; d.style.transform = `scale(${s})`;
  },
  unzoom() { const d = document.documentElement; d.style.transform = ''; d.style.transformOrigin = ''; },
  clear() { ['__sg_cap', '__sg_hl', '__sg_cur'].forEach(id => { const e = document.getElementById(id); if (e) e.style.display = 'none'; }); },
};
