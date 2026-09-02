/* app.js — panels, filters, inspector, reactor. The graph itself is graph.js. */

import { Graph, colourFor } from '/ui/graph.js';

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};

const EXAMPLES = [
  '"brief me"', '"what did I miss this week?"', '"who owes me money?"',
  '"what is the margin model?"', '"plan my day"', '"what changed on Pennine?"',
];

const graph = new Graph($('#graph'));
const state = { data: null, status: null, hidden: new Set(), lastPath: null };

/* --------------------------------------------------------------- toast --- */
let toastTimer = null;
function toast(kicker, body, { warn = false, sticky = false } = {}) {
  const node = $('#toast');
  node.querySelector('.k').textContent = kicker;
  node.querySelector('.b').textContent = body;
  node.classList.toggle('warn', warn);
  node.classList.add('show');
  clearTimeout(toastTimer);
  if (!sticky) toastTimer = setTimeout(() => node.classList.remove('show'), 7000);
}

/* ------------------------------------------------------------- reactor --- */
const REACTOR_STATES = {
  idle:      { colour: '#58dbe6', speed: 0.10, spread: 0.30 },
  listening: { colour: '#3fcf7f', speed: 0.55, spread: 0.85 },
  thinking:  { colour: '#f2a93b', speed: 0.95, spread: 0.55 },
  speaking:  { colour: '#b18cf0', speed: 0.40, spread: 1.00 },
  error:     { colour: '#f4739f', speed: 0.05, spread: 0.20 },
};
let reactorState = 'idle';
let reactorLevel = 0;   // 0..1, driven by mic level once voice exists

function setReactor(next, level = 0) {
  reactorState = next in REACTOR_STATES ? next : 'idle';
  reactorLevel = level;
  $('#reactor-state').textContent = reactorState;
}

(function drawReactor() {
  const canvas = $('#reactor');
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  canvas.width = 150 * dpr; canvas.height = 150 * dpr;
  canvas.style.width = '150px'; canvas.style.height = '150px';
  ctx.scale(dpr, dpr);

  const TICKS = 56;
  function frame(now) {
    const cfg = REACTOR_STATES[reactorState];
    const t = now / 1000;
    ctx.clearRect(0, 0, 150, 150);
    ctx.save();
    ctx.translate(75, 75);

    for (let i = 0; i < TICKS; i++) {
      const angle = (i / TICKS) * Math.PI * 2 - Math.PI / 2;
      const wave = Math.sin(t * cfg.speed * 6 + i * 0.42);
      const amp = 0.35 + cfg.spread * (0.5 + 0.5 * wave) * (0.55 + reactorLevel);
      const inner = 44;
      const outer = inner + 5 + amp * 13;
      ctx.strokeStyle = cfg.colour;
      ctx.globalAlpha = 0.16 + 0.5 * amp;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(Math.cos(angle) * inner, Math.sin(angle) * inner);
      ctx.lineTo(Math.cos(angle) * outer, Math.sin(angle) * outer);
      ctx.stroke();
    }

    ctx.globalAlpha = 0.5;
    ctx.strokeStyle = cfg.colour;
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(0, 0, 38, 0, 7); ctx.stroke();

    const pulse = 0.5 + 0.5 * Math.sin(t * cfg.speed * 4);
    ctx.globalAlpha = 0.10 + 0.16 * pulse;
    ctx.fillStyle = cfg.colour;
    ctx.beginPath(); ctx.arc(0, 0, 26 + pulse * 4, 0, 7); ctx.fill();

    ctx.restore();
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();

/* ----------------------------------------------------------- inspector --- */
function renderInspector(node, path) {
  const box = $('#inspector');
  box.replaceChildren();
  if (!node) {
    box.append(el('p', 'hint', 'Click a node to focus it. Shift-click a second to trace the path.'));
    return;
  }
  box.append(el('h2', null, node.title));
  box.append(el('div', 'meta', `${node.type} · ${node.degree} links · ${node.rel}`));

  if (path && path.length > 1) {
    const line = el('div', 'qualifier',
      path.map((id) => graph.byId.get(id)?.title || id).join('  →  '));
    box.append(line);
  }

  fetch(`/api/node?id=${encodeURIComponent(node.id)}`)
    .then((r) => r.json())
    .then((full) => {
      if (graph.focus !== node.id) return;
      const front = Object.entries(full.front || {})
        .filter(([k]) => !['type', 'title'].includes(k));
      if (front.length) {
        const dl = el('dl', 'front');
        for (const [k, v] of front) {
          if (k === 'qualifier') continue;
          dl.append(el('dt', null, k), el('dd', null, String(v)));
        }
        box.append(dl);
      }
      // A qualifier is the difference between "discount" and "staged payment".
      // It gets its own line, never folded into the number.
      if (full.front && full.front.qualifier) {
        box.append(el('div', 'qualifier', `Qualifier — ${full.front.qualifier}`));
      }
      if (full.warning) box.append(el('div', 'qualifier', `⚠ ${full.warning}`));
      box.append(el('div', 'excerpt', full.excerpt));

      if (full.neighbours?.length) {
        box.append(el('span', 'label', 'Linked'));
        const list = el('ul', 'row-list');
        for (const n of full.neighbours.slice(0, 8)) {
          const li = el('li');
          const dot = el('span', 'swatch');
          dot.style.background = colourFor(n.type);
          li.append(dot, el('span', 'name', n.title), el('span', 'count', String(n.degree)));
          li.onclick = () => focusNode(n.id);
          list.append(li);
        }
        box.append(list);
      }
    })
    .catch(() => box.append(el('div', 'qualifier', '⚠ Could not load this note from the server.')));
}

function focusNode(id) {
  const node = graph.focusById(id);
  if (node) renderInspector(node);
}

graph.onFocus = (node, path) => renderInspector(node, path);

/* -------------------------------------------------------------- panels --- */
function renderHubs(hubs) {
  const list = $('#hubs');
  list.replaceChildren();
  for (const h of hubs) {
    const li = el('li');
    const dot = el('span', 'swatch');
    dot.style.background = colourFor(h.type);
    li.append(dot, el('span', 'name', h.title), el('span', 'count', String(h.degree)));
    li.onclick = () => focusNode(h.id);
    list.append(li);
  }
}

function renderTypes(counts) {
  const list = $('#types');
  list.replaceChildren();
  for (const { type, count } of counts) {
    const li = el('li', 'on');
    const dot = el('span', 'swatch');
    dot.style.background = colourFor(type);
    li.append(dot, el('span', 'name', type), el('span', 'count', String(count)));
    li.onclick = () => {
      if (state.hidden.has(type)) state.hidden.delete(type);
      else state.hidden.add(type);
      li.classList.toggle('off', state.hidden.has(type));
      graph.setHidden(state.hidden);
      graph.alpha = Math.max(graph.alpha, 0.28);
    };
    list.append(li);
  }
}

/* ------------------------------------------------------------ ask bar --- */
let exampleIndex = 0;
function rotateExample() {
  const input = $('#ask');
  if (!input.value) input.placeholder = EXAMPLES[exampleIndex % EXAMPLES.length];
  exampleIndex++;
}
rotateExample();
setInterval(rotateExample, 4200);

$('#ask').addEventListener('keydown', async (ev) => {
  if (ev.key !== 'Enter') return;
  const query = ev.target.value.trim();
  if (!query) return;
  setReactor('thinking');
  try {
    const res = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
    const { hits } = await res.json();
    if (hits.length) focusNode(hits[0].id);   // moves the camera to the top hit
    renderResults(hits, query);               // then the list replaces the panel
  } catch (err) {
    toast('search failed', String(err), { warn: true });
  } finally {
    setReactor('idle');
  }
});

// Focusing the top hit moves the camera; the results list then owns the panel.
function renderResults(hits, query) {
  const box = $('#inspector');
  box.replaceChildren();
  // This is keyword matching, not a model. Never let it read as conversation.
  box.append(el('div', 'qualifier',
    `Keyword search for “${query}” — no model is wired up yet (step 3).`));
  if (!hits.length) {
    box.append(el('p', 'hint', 'Nothing in the index matches that.'));
    return;
  }
  for (const hit of hits) {
    const row = el('div', 'result');
    row.append(el('div', 't', hit.title), el('div', 'e', hit.excerpt));
    row.onclick = () => focusNode(hit.id);
    box.append(row);
  }
}

/* -------------------------------------------------------------- viewbar --- */
$('#btn-fit').onclick = () => graph.fit(true);
$('#btn-labels').onclick = (ev) => {
  graph.showLabels = !graph.showLabels;
  ev.target.classList.toggle('on', graph.showLabels);
};
$('#btn-dim').onclick = (ev) => {
  document.body.style.filter = document.body.style.filter ? '' : 'contrast(1.18) brightness(1.1)';
  ev.target.classList.toggle('on');
};
window.addEventListener('keydown', (ev) => {
  if (ev.key === 'Escape') { graph.focus = null; graph.pathIds = new Set(); renderInspector(null); }
  if (ev.key === 'f' && ev.target.tagName !== 'INPUT') graph.fit(true);
});

/* ----------------------------------------------------------------- boot --- */
async function boot() {
  try {
    const [status, data] = await Promise.all([
      fetch('/api/status').then((r) => r.json()),
      fetch('/api/graph').then((r) => r.json()),
    ]);
    state.status = status; state.data = data;

    const badge = $('#mode-badge');
    badge.textContent = status.demo ? 'demo' : 'live';
    badge.classList.toggle('real', !status.demo);

    graph.setData(data);
    renderHubs(data.hubs);
    renderTypes(data.counts);

    // Degrade loudly: anything the indexer could not do gets said on screen.
    if (status.warnings?.length) {
      toast('indexer', status.warnings.join(' · '), { warn: true, sticky: true });
      setReactor('error');
    } else if (!data.nodes.length) {
      toast('empty index', 'No files were found in the configured folders.',
            { warn: true, sticky: true });
      setReactor('error');
    }
  } catch (err) {
    toast('server unreachable', `Could not load the index: ${err}`,
          { warn: true, sticky: true });
    setReactor('error');
  }
}

boot();
