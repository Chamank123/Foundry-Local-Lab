/* graph.js — force-directed note graph on a canvas.
 *
 * Canvas, not SVG: SVG needs a DOM node per element and stalls somewhere past
 * ~1,500 nodes. Repulsion runs over a spatial grid with a distance cutoff, so
 * cost stays near-linear in node count rather than quadratic.
 */

/* Colour carries the type, so it is the only strong colour on the page.
   Anchors (the five papers) are green; everything else reads against them. */
export const TYPE_COLOURS = {
  paper:       '#3fcf7f',   // the five Part I papers — the anchors
  lecture:     '#4a86e8',
  supervision: '#f0a04b',
  concept:     '#f5c542',
  note:        '#e6e8ea',
  person:      '#b18cf0',
  application: '#f4739f',
  message:     '#57c7f5',
  project:     '#2fd4c4',
  reading:     '#9aa3ad',
  log:         '#6b7280',
  // Kept so a real vault organised around work rather than a course still
  // gets sensible colours rather than a wall of grey.
  client:   '#3fcf7f', call: '#4a86e8', invoice: '#f4739f',
  proposal: '#f0a04b', sop: '#ef7a3d', brief: '#9aa3ad', campaign: '#7d8590',
  module:   '#3fcf7f', assignment: '#f0a04b', shift: '#2fd4c4',
};
const FALLBACK = '#7d8590';
export const colourFor = (t) => TYPE_COLOURS[t] || FALLBACK;

// -- simulation constants, all in one place so they can be tuned ------------
const REPULSION      = 15000;  // strength of node-node push
const CUTOFF         = 270;    // px beyond which repulsion is ignored
const SPRING         = 0.0055; // edge pull
const SPRING_LENGTH  = 98;
const CENTRE_PULL    = 0.0011;
const DAMPING        = 0.86;
const ALPHA_DECAY    = 0.985;
const ALPHA_FLOOR    = 0;      // settle completely. Any floor above zero and
                               // repulsion keeps inflating the layout for ever;
                               // the "still breathing" look is BREATH_* below,
                               // a render offset that never moves the physics.
const SETTLE_AT      = 0.012;  // alpha at which the layout counts as settled
const BREATH_AMP     = 0.9;    // px of render-only drift, so it looks alive
const BREATH_SPEED   = 0.0007;
const PULSE_EVERY    = 3400;   // ms between idle pulses
const PULSE_TIME     = 1100;   // ms for one pulse to travel its link
const LABEL_PAD      = 3;
const MAX_LABELS     = 22;     // a labelled hub cluster, not a wall of text

/* Deterministic PRNG, so the layout starts identically every load. */
function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class Graph {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.nodes = [];
    this.edges = [];
    this.byId = new Map();
    this.adj = new Map();

    this.view = { x: 0, y: 0, k: 1 };
    this.alpha = 1;
    this.hover = null;
    this.focus = null;
    this.pathIds = new Set();
    this.pathFrom = null;
    this.hidden = new Set();      // types switched off in the filter panel
    this.showLabels = true;
    this.dim = false;
    this.pulses = [];
    this.lastPulse = 0;
    this.drag = null;
    this.onFocus = () => {};
    this.onHover = () => {};

    this._grid = new Map();
    this._bind();
    this._resize();
    window.addEventListener('resize', () => this._resize());
    requestAnimationFrame((t) => this._frame(t));
  }

  // -- data ---------------------------------------------------------------
  setData(graph) {
    const rand = mulberry32(20260902);
    const w = this.canvas.clientWidth || 1200;
    const h = this.canvas.clientHeight || 800;
    this.nodes = graph.nodes.map((n) => {
      // Seed on a disc, hubs nearer the middle: settles faster and calmer.
      const a = rand() * Math.PI * 2;
      const r = (0.12 + 0.88 * rand()) * Math.min(w, h) * 0.42;
      const pull = 1 / (1 + n.degree * 0.06);
      return {
        ...n,
        x: w / 2 + Math.cos(a) * r * pull,
        y: h / 2 + Math.sin(a) * r * pull,
        vx: 0, vy: 0,
        phase: rand() * Math.PI * 2,
        radius: 3 + Math.sqrt(n.degree) * 2.2,
      };
    });
    this.byId = new Map(this.nodes.map((n) => [n.id, n]));
    this.adj = new Map(this.nodes.map((n) => [n.id, []]));
    this.edges = [];
    for (const e of graph.edges) {
      const s = this.byId.get(e.s); const t = this.byId.get(e.t);
      if (!s || !t) continue;
      this.edges.push({ s, t });
      this.adj.get(e.s).push(e.t);
      this.adj.get(e.t).push(e.s);
    }
    this.alpha = 1;
    this._settled = false;

    // Pre-settle before the first paint, so the graph appears as a graph
    // rather than flying apart from a seed disc. Bounded by wall clock as
    // well as steps, so a large vault cannot block the page.
    const deadline = performance.now() + 140;
    let steps = 0;
    while (this.alpha > 0.06 && steps < 700 && performance.now() < deadline) {
      this._tick();
      steps += 1;
    }
    this.fit(false);   // rAF finishes the settle and re-fits when it lands
  }

  visible(n) { return !this.hidden.has(n.type); }

  setHidden(types) { this.hidden = new Set(types); }

  shortestPath(aId, bId) {
    if (!this.byId.has(aId) || !this.byId.has(bId)) return [];
    if (aId === bId) return [aId];
    const prev = new Map([[aId, aId]]);
    const queue = [aId];
    while (queue.length) {
      const cur = queue.shift();
      for (const next of this.adj.get(cur) || []) {
        if (prev.has(next)) continue;
        prev.set(next, cur);
        if (next === bId) {
          const path = [bId];
          while (path[path.length - 1] !== aId) path.push(prev.get(path[path.length - 1]));
          return path.reverse();
        }
        queue.push(next);
      }
    }
    return [];
  }

  // -- view ---------------------------------------------------------------
  _resize() {
    const dpr = window.devicePixelRatio || 1;
    const w = window.innerWidth; const h = window.innerHeight;
    this.canvas.width = w * dpr; this.canvas.height = h * dpr;
    this.canvas.style.width = `${w}px`; this.canvas.style.height = `${h}px`;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  /* The visible middle: the window minus the floating panels. Centring on
     the raw window pushes the graph under the inspector and the ask bar. */
  _stage() {
    const w = window.innerWidth; const h = window.innerHeight;
    const inset = w > 1100
      ? { left: 380, right: 290, top: 90, bottom: 110 }
      : { left: 40, right: 40, top: 90, bottom: 110 };
    return {
      w: Math.max(220, w - inset.left - inset.right),
      h: Math.max(220, h - inset.top - inset.bottom),
      cx: inset.left + Math.max(220, w - inset.left - inset.right) / 2,
      cy: inset.top + Math.max(220, h - inset.top - inset.bottom) / 2,
    };
  }

  fit(animate = true) {
    const shown = this.nodes.filter((n) => this.visible(n));
    if (!shown.length) return;
    let minX = Infinity; let minY = Infinity; let maxX = -Infinity; let maxY = -Infinity;
    for (const n of shown) {
      minX = Math.min(minX, n.x); maxX = Math.max(maxX, n.x);
      minY = Math.min(minY, n.y); maxY = Math.max(maxY, n.y);
    }
    const stage = this._stage();
    const k = Math.min(
      stage.w / Math.max(maxX - minX, 1),
      stage.h / Math.max(maxY - minY, 1),
      2.4,
    );
    const target = {
      k,
      x: stage.cx - ((minX + maxX) / 2) * k,
      y: stage.cy - ((minY + maxY) / 2) * k,
    };
    if (!animate) { this.view = target; return; }
    const from = { ...this.view }; const t0 = performance.now();
    const step = (now) => {
      const p = Math.min(1, (now - t0) / 420);
      const e = 1 - (1 - p) ** 3;
      this.view = {
        k: from.k + (target.k - from.k) * e,
        x: from.x + (target.x - from.x) * e,
        y: from.y + (target.y - from.y) * e,
      };
      if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  toWorld(px, py) {
    return { x: (px - this.view.x) / this.view.k, y: (py - this.view.y) / this.view.k };
  }

  // -- physics ------------------------------------------------------------
  _buildGrid() {
    this._grid.clear();
    for (const n of this.nodes) {
      if (!this.visible(n)) continue;
      const key = `${Math.floor(n.x / CUTOFF)},${Math.floor(n.y / CUTOFF)}`;
      let cell = this._grid.get(key);
      if (!cell) { cell = []; this._grid.set(key, cell); }
      cell.push(n);
    }
  }

  _tick() {
    if (this.alpha <= ALPHA_FLOOR) this.alpha = ALPHA_FLOOR;
    this._buildGrid();
    const stage = this._stage();
    const cx = (stage.cx - this.view.x) / this.view.k;
    const cy = (stage.cy - this.view.y) / this.view.k;

    // Repulsion: only against the eight neighbouring cells, so each node
    // compares against a bounded handful rather than all of them.
    for (const n of this.nodes) {
      if (!this.visible(n)) continue;
      const gx = Math.floor(n.x / CUTOFF); const gy = Math.floor(n.y / CUTOFF);
      for (let ix = -1; ix <= 1; ix++) {
        for (let iy = -1; iy <= 1; iy++) {
          const cell = this._grid.get(`${gx + ix},${gy + iy}`);
          if (!cell) continue;
          for (const o of cell) {
            if (o === n) continue;
            let dx = n.x - o.x; let dy = n.y - o.y;
            let d2 = dx * dx + dy * dy;
            if (d2 > CUTOFF * CUTOFF) continue;
            if (d2 < 0.01) { dx = (Math.random() - 0.5) * 0.4; dy = (Math.random() - 0.5) * 0.4; d2 = 0.01; }
            const bulk = (n.radius + o.radius) / 9;
            const force = (REPULSION * this.alpha * bulk) / d2;
            n.vx += dx * force * 0.0016;
            n.vy += dy * force * 0.0016;
          }
        }
      }
    }

    for (const e of this.edges) {
      if (!this.visible(e.s) || !this.visible(e.t)) continue;
      const dx = e.t.x - e.s.x; const dy = e.t.y - e.s.y;
      const d = Math.hypot(dx, dy) || 1;
      const f = (d - SPRING_LENGTH) * SPRING * this.alpha;
      const ux = (dx / d) * f; const uy = (dy / d) * f;
      e.s.vx += ux; e.s.vy += uy;
      e.t.vx -= ux; e.t.vy -= uy;
    }

    for (const n of this.nodes) {
      if (!this.visible(n)) continue;
      if (this.drag && this.drag.node === n) continue;
      n.vx += (cx - n.x) * CENTRE_PULL * this.alpha;
      n.vy += (cy - n.y) * CENTRE_PULL * this.alpha;
      n.vx *= DAMPING; n.vy *= DAMPING;
      n.x += n.vx; n.y += n.vy;
    }
    this.alpha *= ALPHA_DECAY;
    if (!this._settled && this.alpha <= SETTLE_AT) {
      this._settled = true;
      this.fit(true);
    }
  }

  // -- drawing ------------------------------------------------------------
  _frame(now) {
    this._tick();
    this._pulseTick(now);
    this._draw(now);
    requestAnimationFrame((t) => this._frame(t));
  }

  _pulseTick(now) {
    if (now - this.lastPulse > PULSE_EVERY && this.edges.length) {
      const live = this.edges.filter((e) => this.visible(e.s) && this.visible(e.t));
      if (live.length) {
        this.pulses.push({ edge: live[(Math.random() * live.length) | 0], t0: now });
      }
      this.lastPulse = now;
    }
    this.pulses = this.pulses.filter((p) => now - p.t0 < PULSE_TIME);
  }

  _highlight() {
    // What is lit right now: the hovered node and its links, or a traced path.
    if (this.pathIds.size) return this.pathIds;
    const anchor = this.hover || (this.focus ? this.byId.get(this.focus) : null);
    if (!anchor) return null;
    return new Set([anchor.id, ...(this.adj.get(anchor.id) || [])]);
  }

  _breath(n, now) {
    const t = now * BREATH_SPEED;
    return {
      x: n.x + Math.cos(t + n.phase) * BREATH_AMP,
      y: n.y + Math.sin(t * 1.3 + n.phase) * BREATH_AMP,
    };
  }

  _draw(now) {
    const ctx = this.ctx;
    const w = window.innerWidth; const h = window.innerHeight;
    ctx.clearRect(0, 0, w, h);
    ctx.save();
    ctx.translate(this.view.x, this.view.y);
    ctx.scale(this.view.k, this.view.k);

    const lit = this._highlight();
    const faded = lit ? 0.1 : 1;

    // edges
    ctx.lineWidth = 0.7 / this.view.k;
    for (const e of this.edges) {
      if (!this.visible(e.s) || !this.visible(e.t)) continue;
      const on = lit && lit.has(e.s.id) && lit.has(e.t.id);
      ctx.strokeStyle = on
        ? 'rgba(88, 219, 230, 0.5)'
        : `rgba(255, 255, 255, ${0.11 * (lit ? faded : 1)})`;
      const a = this._breath(e.s, now); const b = this._breath(e.t, now);
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }

    // idle pulses travelling a link
    for (const p of this.pulses) {
      const { s, t } = p.edge;
      if (!this.visible(s) || !this.visible(t)) continue;
      const q = (now - p.t0) / PULSE_TIME;
      const fade = Math.sin(q * Math.PI);
      const a = this._breath(s, now); const b = this._breath(t, now);
      ctx.fillStyle = `rgba(88, 219, 230, ${0.5 * fade})`;
      ctx.beginPath();
      ctx.arc(a.x + (b.x - a.x) * q, a.y + (b.y - a.y) * q, 1.8 / this.view.k, 0, 7);
      ctx.fill();
    }

    // nodes, least connected first so hubs land on top
    const shown = this.nodes.filter((n) => this.visible(n))
      .sort((a, b) => a.degree - b.degree);
    for (const n of shown) {
      const on = !lit || lit.has(n.id);
      const alpha = on ? 1 : faded;
      const colour = colourFor(n.type);
      const lift = (this.hover === n || this.focus === n.id) ? 1.5 : 0;
      const p = this._breath(n, now);

      if (on && (n.degree > 6 || lift)) {           // soft bloom, hubs only
        ctx.globalAlpha = 0.13 * alpha;
        ctx.fillStyle = colour;
        ctx.beginPath();
        ctx.arc(p.x, p.y, n.radius + 5 + lift, 0, 7);
        ctx.fill();
      }
      ctx.globalAlpha = alpha;
      ctx.fillStyle = colour;
      ctx.beginPath();
      ctx.arc(p.x, p.y, n.radius + lift, 0, 7);
      ctx.fill();

      if (this.focus === n.id || this.pathIds.has(n.id)) {
        ctx.globalAlpha = 1;
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.4 / this.view.k;
        ctx.beginPath();
        ctx.arc(p.x, p.y, n.radius + 3.5, 0, 7);
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
    ctx.restore();

    if (this.showLabels) this._drawLabels(shown, lit, faded, now);
  }

  _drawLabels(shown, lit, faded, now) {
    const ctx = this.ctx;
    ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    // Most-connected first, and skip any label whose box hits one already
    // placed — otherwise the hub cluster turns to mush.
    const placed = [];
    const ranked = [...shown].sort((a, b) => b.degree - a.degree);
    const minDegree = this.view.k > 1.8 ? 0 : (this.view.k > 1.1 ? 6 : 10);

    for (const n of ranked) {
      const anchored = this.hover === n || this.focus === n.id || this.pathIds.has(n.id);
      if (!anchored && placed.length >= MAX_LABELS) continue;
      if (!anchored && n.degree < minDegree) continue;
      if (lit && !lit.has(n.id) && !anchored) continue;

      const p = this._breath(n, now);
      const sx = p.x * this.view.k + this.view.x;
      const sy = p.y * this.view.k + this.view.y - (n.radius * this.view.k + 9);
      if (sx < -80 || sx > window.innerWidth + 80 || sy < -20 || sy > window.innerHeight + 20) continue;

      const width = ctx.measureText(n.title).width;
      const box = { x: sx - width / 2 - LABEL_PAD, y: sy - 7, w: width + LABEL_PAD * 2, h: 14 };
      let clash = false;
      for (const p of placed) {
        if (box.x < p.x + p.w && box.x + box.w > p.x && box.y < p.y + p.h && box.y + box.h > p.y) {
          clash = true; break;
        }
      }
      if (clash && !anchored) continue;
      placed.push(box);

      const alpha = (!lit || lit.has(n.id)) ? (anchored ? 1 : 0.72) : faded;
      ctx.fillStyle = `rgba(255, 255, 255, ${alpha})`;
      ctx.fillText(n.title, sx, sy);
    }
  }

  // -- interaction --------------------------------------------------------
  _at(px, py) {
    const { x, y } = this.toWorld(px, py);
    let best = null; let bestD = Infinity;
    for (const n of this.nodes) {
      if (!this.visible(n)) continue;
      const d = Math.hypot(n.x - x, n.y - y);
      const hit = n.radius + 7 / this.view.k;
      if (d < hit && d < bestD) { best = n; bestD = d; }
    }
    return best;
  }

  _bind() {
    const c = this.canvas;

    c.addEventListener('mousemove', (ev) => {
      if (this.drag) {
        if (this.drag.node) {
          const { x, y } = this.toWorld(ev.clientX, ev.clientY);
          this.drag.node.x = x; this.drag.node.y = y;
          this.drag.node.vx = 0; this.drag.node.vy = 0;
          this.drag.moved = true;
          this.alpha = Math.max(this.alpha, 0.32);
        } else {
          this.view.x += ev.clientX - this.drag.px;
          this.view.y += ev.clientY - this.drag.py;
          this.drag.px = ev.clientX; this.drag.py = ev.clientY;
          this.drag.moved = true;
        }
        return;
      }
      const hit = this._at(ev.clientX, ev.clientY);
      if (hit !== this.hover) {
        this.hover = hit;
        c.classList.toggle('pointing', !!hit);
        this.onHover(hit);
      }
    });

    c.addEventListener('mousedown', (ev) => {
      const hit = this._at(ev.clientX, ev.clientY);
      this.drag = { node: hit, px: ev.clientX, py: ev.clientY, moved: false, shift: ev.shiftKey };
      c.classList.add('grabbing');
    });

    window.addEventListener('mouseup', (ev) => {
      const d = this.drag;
      this.drag = null;
      c.classList.remove('grabbing');
      if (!d || d.moved) return;
      if (!d.node) {                       // click on empty space clears
        this.focus = null; this.pathIds = new Set(); this.pathFrom = null;
        this.onFocus(null);
        return;
      }
      if (d.shift && this.focus && this.focus !== d.node.id) {
        const path = this.shortestPath(this.focus, d.node.id);
        this.pathIds = new Set(path);
        this.pathFrom = this.focus;
        this.onFocus(d.node, path);
      } else {
        this.pathIds = new Set(); this.pathFrom = null;
        this.focus = d.node.id;
        this.onFocus(d.node);
      }
    });

    c.addEventListener('wheel', (ev) => {
      ev.preventDefault();
      const factor = Math.exp(-ev.deltaY * 0.0016);
      const k = Math.min(5, Math.max(0.15, this.view.k * factor));
      // keep the point under the cursor fixed
      this.view.x = ev.clientX - ((ev.clientX - this.view.x) * k) / this.view.k;
      this.view.y = ev.clientY - ((ev.clientY - this.view.y) * k) / this.view.k;
      this.view.k = k;
    }, { passive: false });
  }

  focusById(id, { centre = true } = {}) {
    const node = this.byId.get(id);
    if (!node) return null;
    this.focus = id;
    this.pathIds = new Set();
    if (centre) {
      this.view.x = window.innerWidth / 2 - node.x * this.view.k;
      this.view.y = window.innerHeight / 2 - node.y * this.view.k;
    }
    return node;
  }
}
