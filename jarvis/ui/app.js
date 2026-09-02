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
  '"brief me"', '"what did I miss this week?"', '"what is due tomorrow"',
  '"do supervision marks count?"', '"plan my day"', '"who emailed me"',
  '"remember that I revise better in the morning"',
];

const graph = new Graph($('#graph'));
const state = { data: null, status: null, model: null, hidden: new Set() };

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

async function ask(text) {
  if (!text.trim()) return;
  $('#ask').value = '';
  const answer = await askAndAnswer(text);
  if (answer && voice.on && !voice.muted) await speakAloud(answer.spoken);
  else if (reactorState !== 'error') setReactor(voice.on ? 'listening' : 'idle');
}

$('#ask').addEventListener('keydown', (ev) => {
  if (ev.key === 'Enter') ask(ev.target.value);
});
$('#btn-brief').onclick = () => ask('brief me');
$('#btn-plan').onclick = () => ask('what should I do today');
$('#btn-memory').onclick = showMemory;

/* The spoken line goes to the toast. It is deliberately not the card text:
   one is what you would hear, the other is what you would read. */
function say(answer) {
  const engine = answer.engine === 'tool' ? answer.tool
    : answer.engine === 'model' ? 'jarvis'
    : answer.engine;
  toast(engine, answer.spoken, {
    warn: answer.engine === 'fallback' || answer.engine === 'error',
    sticky: (answer.warnings || []).length > 0,
  });
}

function renderModelBadge() {
  const badge = $('#model-badge');
  const m = state.model;
  if (!m) { badge.hidden = true; return; }
  badge.hidden = false;
  badge.classList.toggle('missing', !m.available);
  badge.textContent = m.available ? m.model : 'no model · keyword routing';
  badge.title = m.available
    ? `${m.model} at ${m.endpoint}`
    : `${m.reason}\nRouting is keyword scoring against your files — not a model.`;
}

/* ------------------------------------------------------------------ cards --- */
function kv(label, value) {
  const row = el('div', 'kv');
  row.append(el('span', 'kv-k', label), el('span', 'kv-v', String(value)));
  return row;
}

function pill(node, text, cls) {
  if (!text) return;
  node.append(el('div', cls || 'qualifier', text));
}

function fileRow(box, item, label) {
  const row = el('div', 'result');
  row.append(el('div', 't', label));
  if (item.excerpt) row.append(el('div', 'e', item.excerpt));
  if (item.file) row.append(el('div', 'e', item.file));
  if (item.id) row.onclick = () => focusNode(item.id);
  // A qualifier never gets separated from the number it qualifies.
  if (item.qualifier) row.append(el('div', 'qualifier', item.qualifier));
  if (item.flag) row.append(el('div', 'qualifier flag', `⚠ ${item.flag}`));
  box.append(row);
}

function renderCard(question, answer) {
  const box = $('#inspector');
  box.replaceChildren();
  const card = answer.card || {};

  const head = el('div', 'meta');
  head.textContent = `${answer.tool || 'conversation'} · ${answer.engine}`
    + (answer.routed_by ? ` · routed by ${answer.routed_by}` : '');
  box.append(head);

  if (answer.engine === 'fallback') {
    pill(box, 'No model is loaded. This was decided by scoring your question '
             + 'against your files — keyword matching, not a model.');
  }
  for (const w of answer.warnings || []) pill(box, `⚠ ${w}`);

  if (card.hits) {
    box.append(el('span', 'label', `${card.hits.length} files`));
    card.hits.forEach((h) => fileRow(box, h, h.title));
  } else if (card.messages) {
    box.append(el('span', 'label', `${card.unread} unread of ${card.total}`));
    card.messages.forEach((m) => {
      const row = el('div', 'result');
      row.append(el('div', 't', `${m.from} — ${m.subject}`));
      row.append(el('div', 'e', m.excerpt));
      row.append(el('div', 'e', `${m.received} · ${m.unread ? 'unread' : 'read'} · ${m.known_note}`));
      if (m.flag) row.append(el('div', 'qualifier flag', `⚠ ${m.flag}`));
      row.onclick = () => focusNode(m.id);
      box.append(row);
    });
    pill(box, card.note, 'meta');
  } else if (card.items) {
    box.append(el('span', 'label', 'Plan'));
    card.items.forEach((it, i) => fileRow(box, it, `${i + 1}. ${it.what} — ${it.why}`));
    pill(box, card.rule, 'meta');
  } else if (card.due || card.slipped || card.closing) {
    for (const [key, label] of [['slipped', 'Slipped'], ['due', 'Due'],
                                ['closing', 'Closing'], ['unread', 'Unread']]) {
      const rows = card[key] || [];
      if (!rows.length) continue;
      box.append(el('span', 'label', `${label} (${rows.length})`));
      rows.forEach((r) => fileRow(box, r,
        r.title ? `${r.title} — ${r.when || ''}` : `${r.from} — ${r.subject}`));
    }
    pill(box, card.note, 'meta');
  } else if (card.status === 'written') {
    box.append(el('span', 'label', 'Written'));
    box.append(kv('file', card.file), kv('date', card.date));
    box.append(el('div', 'excerpt', card.fact));
    pill(box, card.note, 'meta');
  } else if (card.status === 'disabled' || card.status === 'not implemented') {
    box.append(el('span', 'label', 'Web research'));
    box.append(kv('status', card.status), kv('reason', card.reason || card.note || ''));
    pill(box, card.how || card.note);
    (card.local_context || []).forEach((c) => fileRow(box, c, c.title));
  } else if (Object.keys(card).length) {
    box.append(el('div', 'excerpt', JSON.stringify(card, null, 2)));
  } else {
    box.append(el('p', 'hint', answer.spoken));
  }
}

async function showMemory() {
  setReactor('thinking');
  try {
    const { remembered } = await (await fetch('/api/memory')).json();
    const box = $('#inspector');
    box.replaceChildren();
    box.append(el('div', 'meta', 'memory · the only place anything is written'));
    if (!remembered.length) {
      box.append(el('p', 'hint', 'Nothing remembered yet.'));
    } else {
      remembered.forEach((m) => {
        const row = el('div', 'result');
        row.append(el('div', 't', m.fact), el('div', 'e', `${m.date} · ${m.file}`));
        box.append(row);
      });
    }
    toast('memory', `${remembered.length} remembered.`);
  } finally {
    setReactor('idle');
  }
}


/* ==========================================================================
   Voice. Recording with MediaRecorder, transcription server-side via Scribe.
   Never the Web Speech API: Chrome-only, ships audio to Google, and in Brave
   it is a stub that fails silently — you talk and nothing happens.
   ========================================================================== */

// Tune the turn-taking here. These are the only numbers that decide when
// JARVIS thinks you have stopped talking.
const SILENCE_MS       = 900;   // quiet for this long ends your turn
const SILENCE_LEVEL    = 0.045; // RMS below this counts as quiet
const LEVEL_TICK_MS    = 50;    // setInterval, deliberately not rAF
const MIN_UTTERANCE_MS = 400;   // shorter than this is a cough, not a turn
const MAX_UTTERANCE_MS = 30000; // hard stop, so nothing records for ever

const voice = {
  on: false,          // the mic session is open
  deaf: false,        // true while JARVIS is speaking — see below
  muted: false,       // replies are not spoken
  stream: null,
  ctx: null,
  analyser: null,
  buffer: null,
  recorder: null,
  chunks: [],
  timer: null,
  startedAt: 0,
  lastLoudAt: 0,
  heardSpeech: false,
  audio: null,
  status: null,
};

function caption(text, cls) {
  const node = $('#caption');
  node.textContent = text || '';
  node.className = cls ? `caption ${cls}` : 'caption';
  node.hidden = !text;
}

function pickMime() {
  const wanted = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg'];
  for (const m of wanted) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(m)) return m;
  }
  return '';
}

async function startVoice() {
  if (voice.on) return stopVoice('you stopped it');

  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    toast('no microphone support',
          'This browser cannot record audio. Voice needs MediaRecorder.',
          { warn: true, sticky: true });
    setReactor('error');
    return;
  }

  try {
    voice.stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true },
    });
  } catch (err) {
    // A blocked microphone that produces no error is the single most
    // confusing failure in this whole build. So: say it, loudly, on screen.
    const why = err?.name === 'NotAllowedError'
      ? 'The browser blocked the microphone. Click the padlock in the address bar and allow it.'
      : err?.name === 'NotFoundError'
        ? 'No microphone was found on this machine.'
        : `The microphone could not start: ${err?.name || err}`;
    toast('microphone blocked', why, { warn: true, sticky: true });
    caption(why, 'error');
    setReactor('error');
    return;
  }

  voice.ctx = new (window.AudioContext || window.webkitAudioContext)();
  const source = voice.ctx.createMediaStreamSource(voice.stream);
  voice.analyser = voice.ctx.createAnalyser();
  voice.analyser.fftSize = 1024;
  voice.buffer = new Uint8Array(voice.analyser.fftSize);
  source.connect(voice.analyser);   // analyser only — never back to the speakers

  voice.on = true;
  $('#btn-mic').classList.add('live');
  beginTurn();

  // setInterval, not requestAnimationFrame: rAF stops dead in a backgrounded
  // tab, and a mic that goes silently deaf is exactly what we are avoiding.
  voice.timer = setInterval(levelTick, LEVEL_TICK_MS);
}

function beginTurn() {
  if (!voice.on || !voice.stream) return;
  const mime = pickMime();
  try {
    voice.recorder = mime
      ? new MediaRecorder(voice.stream, { mimeType: mime })
      : new MediaRecorder(voice.stream);
  } catch (err) {
    toast('recorder failed', String(err), { warn: true, sticky: true });
    return stopVoice('recorder failed');
  }
  voice.chunks = [];
  voice.recorder.ondataavailable = (ev) => {
    if (ev.data && ev.data.size) voice.chunks.push(ev.data);
  };
  voice.recorder.onstop = () => submitTurn();
  voice.recorder.start();
  voice.startedAt = performance.now();
  voice.lastLoudAt = performance.now();
  voice.heardSpeech = false;
  setReactor('listening');
  caption('listening…');
}

function levelTick() {
  if (!voice.analyser) return;
  voice.analyser.getByteTimeDomainData(voice.buffer);
  let sum = 0;
  for (let i = 0; i < voice.buffer.length; i++) {
    const v = (voice.buffer[i] - 128) / 128;
    sum += v * v;
  }
  const rms = Math.sqrt(sum / voice.buffer.length);

  // The bars are the real microphone level, not an animation.
  if (!voice.deaf) setReactor(reactorState === 'listening' ? 'listening' : reactorState,
                              Math.min(1, rms * 6));

  // Deaf while speaking: otherwise it transcribes its own voice through the
  // speakers and talks to itself for ever.
  if (voice.deaf || !voice.recorder || voice.recorder.state !== 'recording') return;

  const now = performance.now();
  if (rms >= SILENCE_LEVEL) {
    voice.lastLoudAt = now;
    if (!voice.heardSpeech) { voice.heardSpeech = true; caption('hearing you…', 'live'); }
  }

  const elapsed = now - voice.startedAt;
  const quietFor = now - voice.lastLoudAt;
  if (voice.heardSpeech && elapsed > MIN_UTTERANCE_MS && quietFor > SILENCE_MS) {
    voice.recorder.stop();
  } else if (elapsed > MAX_UTTERANCE_MS) {
    caption('that was long — sending what I have', 'live');
    voice.recorder.stop();
  }
}

async function submitTurn() {
  const blob = new Blob(voice.chunks, { type: voice.chunks[0]?.type || 'audio/webm' });
  const seconds = (performance.now() - voice.startedAt) / 1000;
  voice.chunks = [];
  if (!voice.heardSpeech || blob.size < 1200) { if (voice.on) beginTurn(); return; }

  setReactor('thinking');
  caption('transcribing…');
  try {
    const res = await fetch('/api/listen', {
      method: 'POST',
      headers: { 'Content-Type': blob.type, 'X-Audio-Seconds': seconds.toFixed(1) },
      body: blob,
    });
    const payload = await res.json();
    if (!res.ok || payload.error) {
      caption(payload.error || 'Transcription failed.', 'error');
      toast('transcription failed', payload.error || `HTTP ${res.status}`,
            { warn: true, sticky: true });
      setReactor('error');
      if (voice.on && !payload.fatal) beginTurn();
      else stopVoice('transcription unavailable');
      return;
    }
    const text = (payload.text || '').trim();
    if (!text) { caption('did not catch that', 'live'); if (voice.on) beginTurn(); return; }

    caption(`“${text}”`);
    const answer = await askAndAnswer(text);
    if (answer && !voice.muted) await speakAloud(answer.spoken);
  } catch (err) {
    caption(`transcription failed: ${err}`, 'error');
    toast('transcription failed', String(err), { warn: true, sticky: true });
    setReactor('error');
  } finally {
    if (voice.on && reactorState !== 'error') beginTurn();
  }
}

async function speakAloud(text) {
  if (!text) return;
  voice.deaf = true;                       // stop listening before a sound plays
  if (voice.recorder && voice.recorder.state === 'recording') {
    voice.recorder.onstop = null;          // this stop is not a turn
    voice.recorder.stop();
  }
  setReactor('speaking');
  try {
    const res = await fetch('/api/speak', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      caption(err.error || 'Speech failed.', 'error');
      toast('speech failed', err.error || `HTTP ${res.status}`,
            { warn: true, sticky: true });
      return;
    }
    const url = URL.createObjectURL(await res.blob());
    await new Promise((resolve) => {
      voice.audio = new Audio(url);
      voice.audio.onended = voice.audio.onerror = () => {
        URL.revokeObjectURL(url);
        resolve();
      };
      voice.audio.play().catch(() => resolve());
    });
  } finally {
    voice.audio = null;
    voice.deaf = false;                    // ears back on, only now
    setReactor(voice.on ? 'listening' : 'idle');
  }
}

/* Barge-in is an explicit action: the mic button, Space, or Esc. */
function bargeIn() {
  if (voice.audio) {
    voice.audio.pause();
    voice.audio.onended?.();
    voice.audio = null;
  }
  voice.deaf = false;
  caption('go on', 'live');
  if (voice.on) beginTurn(); else setReactor('idle');
}

function stopVoice(why) {
  voice.on = false;
  clearInterval(voice.timer);
  voice.timer = null;
  if (voice.recorder && voice.recorder.state !== 'inactive') {
    voice.recorder.onstop = null;
    voice.recorder.stop();
  }
  voice.stream?.getTracks().forEach((t) => t.stop());
  voice.ctx?.close().catch(() => {});
  voice.stream = voice.ctx = voice.analyser = voice.recorder = null;
  $('#btn-mic').classList.remove('live');
  caption(why ? `microphone off — ${why}` : '');
  setReactor(reactorState === 'error' ? 'error' : 'idle');
}

/* One place that asks and renders, shared by typing and by speaking. */
async function askAndAnswer(text) {
  setReactor('thinking');
  try {
    const res = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    const answer = await res.json();
    state.model = answer.model;
    renderModelBadge();
    say(answer);
    renderCard(text, answer);
    return answer;
  } catch (err) {
    toast('unreachable', `The server did not answer: ${err}`,
          { warn: true, sticky: true });
    setReactor('error');
    return null;
  }
}

$('#btn-mic').onclick = () => (voice.audio ? bargeIn() : startVoice());
$('#btn-mute').onclick = (ev) => {
  voice.muted = !voice.muted;
  ev.currentTarget.classList.toggle('live', voice.muted);
  ev.currentTarget.textContent = voice.muted ? '🔇' : '🔊';
  toast('voice', voice.muted ? 'Replies stay on screen.' : 'Replies spoken again.');
};

window.addEventListener('keydown', (ev) => {
  if (ev.target.tagName === 'INPUT') return;
  if (ev.code === 'Space') { ev.preventDefault(); bargeIn(); }
  if (ev.key === 'Escape' && voice.audio) bargeIn();
});

async function loadVoiceStatus() {
  try {
    voice.status = await (await fetch('/api/voice')).json();
  } catch { return; }
  const ok = voice.status.tts && voice.status.stt;
  $('#btn-mic').disabled = !ok;
  $('#btn-mute').disabled = !ok;
  if (!ok) {
    $('#btn-mic').title = voice.status.reason || 'Voice unavailable';
    // Never let voice look available when it is not.
    toast('voice unavailable', voice.status.reason, { warn: true, sticky: true });
  } else {
    $('#btn-mic').title = `Talk — ${voice.status.voice_name || voice.status.voice}`;
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
    state.model = status.model;
    renderModelBadge();
    loadVoiceStatus();

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
