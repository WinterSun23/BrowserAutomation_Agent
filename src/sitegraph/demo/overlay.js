window.__sg = {
  ensure() {
    if (document.getElementById('__sg_cap')) return;
    const c = document.createElement('div'); c.id = '__sg_cap';
    c.style.cssText = 'position:fixed;left:50%;bottom:24px;transform:translateX(-50%);max-width:72%;padding:14px 22px;' +
      'background:rgba(17,17,17,.88);color:#fff;font:600 19px/1.45 system-ui,sans-serif;border-radius:12px;' +
      'z-index:2147483647;pointer-events:none;display:none;text-align:center;box-shadow:0 6px 24px rgba(0,0,0,.25)';
    const h = document.createElement('div'); h.id = '__sg_hl';
    h.style.cssText = 'position:fixed;border:3px solid #ff3d71;border-radius:8px;box-shadow:0 0 0 9999px rgba(0,0,0,.25);' +
      'transition:all .3s ease;z-index:2147483646;pointer-events:none;display:none';
    const s = document.createElement('div'); s.id = '__sg_step';
    s.style.cssText = 'position:fixed;top:18px;left:18px;padding:6px 12px;background:#ff3d71;color:#fff;' +
      'font:700 15px/1 system-ui,sans-serif;border-radius:999px;z-index:2147483647;pointer-events:none;display:none';
    document.documentElement.append(c, h, s);
  },
  caption(t) { this.ensure(); const c = document.getElementById('__sg_cap'); c.textContent = t || ''; c.style.display = t ? 'block' : 'none'; },
  step(t) { this.ensure(); const s = document.getElementById('__sg_step'); s.textContent = t || ''; s.style.display = t ? 'block' : 'none'; },
  highlight(x, y, w, h) {
    this.ensure(); const e = document.getElementById('__sg_hl');
    Object.assign(e.style, { left: (x - 6) + 'px', top: (y - 6) + 'px', width: (w + 12) + 'px', height: (h + 12) + 'px', display: 'block' });
  },
  cursor(x, y) {
    let c = document.getElementById('__sg_cur');
    if (!c) { c = document.createElement('div'); c.id = '__sg_cur';
      c.style.cssText = 'position:fixed;width:18px;height:18px;border-radius:50%;background:rgba(255,61,113,.9);' +
        'box-shadow:0 0 0 6px rgba(255,61,113,.25);z-index:2147483647;pointer-events:none;' +
        'transition:left .7s cubic-bezier(.4,0,.2,1),top .7s cubic-bezier(.4,0,.2,1);' +
        'left:' + (window.innerWidth / 2) + 'px;top:' + (window.innerHeight / 2) + 'px';
      document.documentElement.append(c); c.getBoundingClientRect(); }
    c.style.left = (x - 9) + 'px'; c.style.top = (y - 9) + 'px'; c.style.display = 'block';
  },
  zoom(x, y, s) {
    const d = document.documentElement;
    d.style.transition = 'transform .4s ease'; d.style.transformOrigin = `${x}px ${y}px`; d.style.transform = `scale(${s})`;
  },
  unzoom() { const d = document.documentElement; d.style.transform = ''; d.style.transformOrigin = ''; },
  // draw attention to the highlighted element without moving anything (page zoom pushes the caption off screen)
  pulse() {
    const e = document.getElementById('__sg_hl'); if (!e || !e.animate) return;
    e.animate([{ boxShadow: '0 0 0 9999px rgba(0,0,0,.25), 0 0 0 0 rgba(255,61,113,.8)' },
               { boxShadow: '0 0 0 9999px rgba(0,0,0,.25), 0 0 0 16px rgba(255,61,113,0)' }], { duration: 900, iterations: 2 });
  },
  // a full-screen title / closing card (the recording shows it like a slide)
  card(title, text) {
    let k = document.getElementById('__sg_card');
    if (!k) { k = document.createElement('div'); k.id = '__sg_card';
      k.style.cssText = 'position:fixed;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;' +
        'gap:18px;padding:8%;background:linear-gradient(135deg,#1b1f3b,#3a1c4a);color:#fff;text-align:center;' +
        'font-family:system-ui,sans-serif;z-index:2147483647;pointer-events:none';
      document.documentElement.append(k); }
    k.innerHTML = '';
    const h = document.createElement('div'); h.textContent = title || '';
    h.style.cssText = 'font:800 44px/1.2 system-ui,sans-serif';
    const p = document.createElement('div'); p.textContent = text || '';
    p.style.cssText = 'font:500 24px/1.5 system-ui,sans-serif;max-width:900px;opacity:.9';
    k.append(h, p); k.style.display = 'flex';
  },
  uncard() { const k = document.getElementById('__sg_card'); if (k) k.style.display = 'none'; },
  clear() { ['__sg_cap', '__sg_hl', '__sg_cur', '__sg_step', '__sg_card'].forEach(id => { const e = document.getElementById(id); if (e) e.style.display = 'none'; }); },
};
