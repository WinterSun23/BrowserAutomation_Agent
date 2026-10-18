(() => {
  const IMPLICIT = (el) => {
    const t = el.tagName.toLowerCase();
    const type = (el.getAttribute('type') || '').toLowerCase();
    if (t === 'a' && el.hasAttribute('href')) return 'link';
    if (t === 'button' || t === 'summary') return 'button';
    if (t === 'input') {
      if (['button', 'submit', 'reset', 'image'].includes(type)) return 'button';
      if (type === 'checkbox') return 'checkbox';
      if (type === 'radio') return 'radio';
      if (type === 'search') return 'searchbox';
      if (type === 'hidden') return null;
      return 'textbox';
    }
    if (t === 'select') return el.multiple ? 'listbox' : 'combobox';
    if (t === 'textarea') return 'textbox';
    if (t === 'nav') return 'navigation';
    if (t === 'header' && !el.closest('article,section,main,aside,nav')) return 'banner';
    if (t === 'footer' && !el.closest('article,section,main,aside,nav')) return 'contentinfo';
    if (t === 'main') return 'main';
    if (t === 'form') return 'form';
    if (t === 'aside') return 'complementary';
    if (t === 'dialog') return 'dialog';
    if (t === 'ul' || t === 'ol') return 'list';
    if (t === 'li') return 'listitem';
    if (t === 'table') return 'table';
    if (t === 'tr') return 'row';
    if (t === 'article') return 'article';
    if (t === 'details') return 'group';
    if (t === 'section' && (el.hasAttribute('aria-label') || el.hasAttribute('aria-labelledby'))) return 'region';
    return null;
  };
  const INTERACTIVE = new Set(['link', 'button', 'checkbox', 'radio', 'searchbox', 'textbox', 'combobox', 'listbox',
    'option', 'menuitem', 'menuitemcheckbox', 'menuitemradio', 'tab', 'switch', 'slider', 'spinbutton']);
  const STRUCTURAL = new Set(['navigation', 'banner', 'contentinfo', 'main', 'form', 'complementary', 'dialog',
    'alertdialog', 'list', 'listitem', 'table', 'row', 'article', 'group', 'region', 'search', 'menu', 'menubar',
    'tablist', 'tabpanel', 'toolbar', 'tooltip', 'status', 'alert', 'grid', 'feed']);
  const role = (el) => el.getAttribute('role') || IMPLICIT(el);
  const visible = (el) => {
    if (el.hidden) return false;
    const s = getComputedStyle(el);
    if (s.display === 'none' || s.visibility === 'hidden') return false;
    const r = el.getBoundingClientRect();
    return r.width > 0 && r.height > 0;
  };
  const text = (el) => (el.innerText || el.textContent || '').replace(/\s+/g, ' ').trim();
  const accName = (el) => {
    const al = el.getAttribute('aria-label'); if (al) return al.trim();
    const lb = el.getAttribute('aria-labelledby');
    if (lb) {
      const t = lb.split(/\s+/).map(id => { const n = document.getElementById(id); return n ? text(n) : ''; }).join(' ').trim();
      if (t) return t;
    }
    if (el.labels && el.labels.length) return text(el.labels[0]);
    const alt = el.getAttribute('alt'); if (alt) return alt.trim();
    const t = el.tagName.toLowerCase();
    if (t === 'input' || t === 'textarea' || t === 'select')
      return (el.getAttribute('placeholder') || el.getAttribute('name') || el.getAttribute('title') || '').trim();
    const r = role(el);
    if (INTERACTIVE.has(r) || r === 'listitem' || r === 'article' || r === 'row') return text(el).slice(0, 300);
    return (el.getAttribute('title') || '').trim();
  };
  const cssSeg = (el, generic) => {
    const t = el.tagName.toLowerCase();
    if (!generic && el.id && !/\d{3,}/.test(el.id)) return t + '#' + CSS.escape(el.id);
    const cls = [...el.classList].filter(c => !/\d{3,}|active|open|selected|hover|focus/.test(c))
      .slice(0, 2).map(c => '.' + CSS.escape(c)).join('');
    if (generic) return t + cls;
    const sibs = el.parentElement ? [...el.parentElement.children].filter(s => s.tagName === el.tagName) : [];
    return t + cls + (sibs.length > 1 ? `:nth-of-type(${sibs.indexOf(el) + 1})` : '');
  };
  const cssPath = (el, generic) => {
    const segs = []; let n = el;
    while (n && n.nodeType === 1 && n !== document.documentElement) {
      const s = cssSeg(n, generic); segs.unshift(s);
      if (!generic && s.includes('#')) break;
      n = n.parentElement;
    }
    return segs.join(' > ');
  };
  const isModal = (el) => {
    if (el.getAttribute('aria-modal') === 'true') return true;
    try { return el.tagName === 'DIALOG' && el.matches(':modal'); } catch (e) { return false; }
  };
  const hasHiddenPopup = (el) => {
    const p = el.parentElement; if (!p) return false;
    const sub = p.querySelector(':scope > ul, :scope > [role="menu"], :scope > [class*="dropdown"], ' +
      ':scope > [class*="submenu"], :scope > [class*="sub-menu"]');
    return !!sub && sub !== el && !visible(sub);
  };
  // Elements with no ARIA / implicit role that are clickable anyway (a <div onclick>, a framework-rendered button)
  // are taken to be buttons. Signals: an event listener found through the DevTools API (window.__sgClickable, filled
  // by the browser session) or a cheap hint (onclick, tabindex, a btn / button class, a pointer cursor that starts here).
  const NOT_CLICKABLE = new Set(['html', 'body', 'head', 'script', 'style', 'noscript', 'svg', 'path', 'br', 'hr']);
  const listened = window.__sgClickable;
  const hint = (el) => {
    if (listened && listened.has(el)) return true;
    if (el.hasAttribute('onclick') || el.hasAttribute('onmousedown') || el.hasAttribute('onmouseup')) return true;
    const ti = el.getAttribute('tabindex'); if (ti !== null && ti !== '-1') return true;
    if (typeof el.className === 'string' && /(^|[\s_-])(btn|button)([\s_-]|$)/i.test(el.className)) return true;
    if (getComputedStyle(el).cursor === 'pointer') {
      const p = el.parentElement; return !p || getComputedStyle(p).cursor !== 'pointer';
    }
    return false;
  };
  const everything = document.body ? [...document.body.querySelectorAll('*')] : [];
  const inferred = new Map();
  for (const el of everything) {
    if (role(el) || NOT_CLICKABLE.has(el.tagName.toLowerCase()) || el instanceof SVGElement) continue;
    if (!hint(el) || !visible(el)) continue;
    inferred.set(el, null);
  }
  if (inferred.size) {
    // a wrapper that holds a real control (or another candidate) is not itself the control; nor is a page-sized box
    const holdsControl = new Set();
    const markAncestors = (el) => {
      let p = el.parentElement;
      while (p && !holdsControl.has(p)) { holdsControl.add(p); p = p.parentElement; }
    };
    for (const el of everything) {
      const r = role(el);
      if ((r && INTERACTIVE.has(r) && visible(el)) || inferred.has(el)) markAncestors(el);
    }
    const area = Math.max(1, innerWidth * innerHeight);
    for (const el of [...inferred.keys()]) {
      const rc = el.getBoundingClientRect();
      const nm = (text(el).slice(0, 300) || el.getAttribute('aria-label') || el.getAttribute('title') ||
                  el.getAttribute('data-testid') || '').trim();
      if (holdsControl.has(el) || rc.width * rc.height > 0.6 * area || !nm) inferred.delete(el);
      else inferred.set(el, nm);
    }
  }
  const nodes = []; const index = new Map();
  for (const el of everything) {
    const wasInferred = inferred.has(el);
    const r = wasInferred ? 'button' : role(el); if (!r) continue;
    const interactive = INTERACTIVE.has(r);
    if (!interactive && !STRUCTURAL.has(r)) continue;
    if (!visible(el)) continue;
    let p = el.parentElement, parent = null, depth = 0;
    while (p) { if (index.has(p)) { parent = index.get(p); depth = nodes[parent].depth + 1; break; } p = p.parentElement; }
    const rect = el.getBoundingClientRect();
    const node = {
      idx: nodes.length, role: r, name: wasInferred ? inferred.get(el) : accName(el), inferred: wasInferred, tag: el.tagName.toLowerCase(),
      href: el.getAttribute('href') ? el.href : null, input_type: el.getAttribute('type'),
      parent, depth, interactive,
      states: {
        expanded: el.getAttribute('aria-expanded'), selected: el.getAttribute('aria-selected'),
        checked: el.checked === undefined ? el.getAttribute('aria-checked') : String(el.checked),
        disabled: String(!!el.disabled || el.getAttribute('aria-disabled') === 'true'),
        modal: String(isModal(el)),
        haspopup: el.getAttribute('aria-haspopup'), popup: String(hasHiddenPopup(el)),
      },
      bbox: [rect.x, rect.y, rect.width, rect.height],
      css: cssPath(el, false), gcss: cssPath(el, true), testid: el.getAttribute('data-testid'),
      attrs: { id: el.id || null, class: typeof el.className === 'string' && el.className ? el.className : null,
               name: el.getAttribute('name'), placeholder: el.getAttribute('placeholder') },
    };
    if (node.tag === 'input' || node.tag === 'textarea') {
      // limits of number / range / date inputs, the autocomplete hint (sensitive fields) and the radio group
      const lim = {};
      for (const a of ['min', 'max', 'step', 'autocomplete']) { const v = el.getAttribute(a); if (v !== null) lim[a] = v; }
      if ((node.input_type || '').toLowerCase() === 'radio' && el.name) lim.group = el.name;
      if (el.disabled) lim.disabled = true;
      node.limits = lim;
    }
    if (node.tag === 'select') {
      // the first 30 option labels and which one is selected now (sort / filter dropdowns)
      node.options = [...el.options].slice(0, 30).map(o => (o.label || o.text || '').replace(/\s+/g, ' ').trim());
      node.selected_index = el.selectedIndex;
    }
    index.set(el, node.idx); nodes.push(node);
  }
  return { url: location.href, title: document.title, nodes };
})()
