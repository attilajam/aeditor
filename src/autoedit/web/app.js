import WaveSurfer from '/assets/vendor/wavesurfer.esm.js';

const $ = (id) => document.getElementById(id);
const el = {
  title: $('project-title'), save: $('save-state'), video: $('preview-video'), emptyVideo: $('video-empty'),
  videoStage: $('video-stage'), previewState: $('preview-state'), stalePill: $('stale-pill'),
  current: $('current-time'), total: $('total-time'), videoTime: $('video-time'), stageMeta: $('stage-meta'),
  play: $('play-button'), loop: $('loop-button'), add: $('add-marker-button'), jump: $('jump-selection-button'),
  waveScroll: $('wave-scroll'), waveInner: $('wave-inner'), waveform: $('waveform'), markers: $('markers-layer'), quiet: $('quiet-layer'),
  playhead: $('playhead'), timeRuler: $('time-ruler'), zoom: $('zoom-range'), rangeMode: $('range-mode-button'), snap: $('snap-toggle'), hint: $('tool-hint'),
  selection: $('selection-summary'), fps: $('frame-rate'), cutCount: $('cut-count'), rangeForm: $('range-form'),
  rangeStart: $('range-start'), rangeEnd: $('range-end'), ranges: $('range-list'), rangeCount: $('range-count'),
  markerList: $('marker-list'), markerCount: $('marker-count'), undo: $('undo-button'), redo: $('redo-button'),
  exportsPanel: $('exports-panel'), exportsList: $('exports-list'), exportCount: $('export-count'),
  reset: $('reset-button'), preview: $('preview-button'), export: $('export-button'), summary: $('edit-summary'),
  notice: $('notice'), noticeText: $('notice-text'), audio: $('audio-master'),
  noticeLinks: $('notice-links'), exportDialog: $('export-dialog'), exportForm: $('export-form'),
  exportFilename: $('export-filename'), exportOverwrite: $('export-overwrite'),
};

let state = null;
let ws = null;
let selected = new Set();
let zoomFactor = 1;
let rangeMode = false;
let drag = null;
let drawingRange = null;
let suppressClick = false;
let undoStack = [];
let redoStack = [];
let saveTimer = null;
let saving = false;
let saveAgain = false;
let dirty = false;
let lastSavedFingerprint = '';
let polling = false;
let loopBounds = null;
let animation = 0;
let noticeTimer = 0;
let jobBusy = false;

const cloneEdit = () => ({
  markers: state.markers.map(m => ({...m})),
  no_cut_ranges: (state.no_cut_ranges || []).map(r => [...r]),
});
const fingerprint = (data = cloneEdit()) => JSON.stringify({
  markers: data.markers.map(m => [m.id, m.frame, m.enabled]).sort((a, b) => String(a[0]).localeCompare(String(b[0]))),
  ranges: data.no_cut_ranges.map(r => [...r]).sort((a, b) => a[0] - b[0] || a[1] - b[1]),
});
const fps = () => Number(state?.fps) || 30;
const totalFrames = () => Math.max(1, Number(state?.total_frames) || Math.round((state?.duration || 0) * fps()));
const duration = () => Number(state?.duration) || totalFrames() / fps();
const clampFrame = (n) => Math.max(1, Math.min(totalFrames() - 1, Math.round(n)));
const clampRangeFrame = (n) => Math.max(0, Math.min(totalFrames(), Math.round(n)));
const timeFor = (frame) => frame / fps();
const formatTime = (time, millis = true) => {
  const totalMs = Math.round(Math.max(0, Number(time) || 0) * 1000);
  const mins = Math.floor(totalMs / 60000);
  const secs = Math.floor(totalMs / 1000) % 60;
  const ms = totalMs % 1000;
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}${millis ? `.${String(ms).padStart(3, '0')}` : ''}`;
};
const cutMarkers = () => state.markers.filter(m => m.enabled && !state.no_cut_ranges.some(([a, b]) => a <= m.frame && m.frame < b)).sort((a, b) => a.frame - b.frame || String(a.id).localeCompare(String(b.id)));
const selectedMarkers = () => state.markers.filter(m => selected.has(m.id));

function setSaveState(label, mode = 'saved') {
  const dot = el.save.querySelector('i');
  const text = el.save.querySelector('span');
  text.textContent = label;
  el.save.dataset.state = mode;
  dot.className = '';
}
function showNotice(message, timeout = 0) {
  el.noticeText.textContent = message;
  el.noticeLinks.replaceChildren();
  el.notice.hidden = false;
  clearTimeout(noticeTimer);
  if (timeout) noticeTimer = setTimeout(() => { el.notice.hidden = true; }, timeout);
}
function clearNotice() { el.notice.hidden = true; clearTimeout(noticeTimer); }
function stalePreview() {
  return !state || !state.video_url || state.preview_revision !== state.revision || dirty;
}
function renderPreviewState() {
  const stale = stalePreview();
  el.stalePill.hidden = !state?.video_url || !stale;
  el.emptyVideo.hidden = Boolean(state?.video_url);
  el.video.hidden = !state?.video_url;
  const activeJob = state?.job && ['queued', 'running'].includes(state.job.status);
  el.previewState.textContent = activeJob ? `${state.job.kind === 'export' ? 'Export' : 'Preview'} ${state.job.status}…` : !state?.video_url ? 'Not rendered' : stale ? 'Needs update' : 'Up to date';
  el.previewState.className = `preview-badge ${activeJob ? 'is-working' : stale ? 'is-stale' : 'is-fresh'}`;
  el.preview.textContent = state?.job?.kind === 'preview' && activeJob ? 'Rendering preview…' : 'Update preview ↗';
  el.preview.disabled = jobBusy || (state?.job?.kind === 'preview' && ['queued', 'running'].includes(state.job.status));
  el.export.disabled = jobBusy || (state?.job?.kind === 'export' && ['queued', 'running'].includes(state.job.status));
  if (state?.video_url && el.video.dataset.url !== state.video_url) {
    const wasPlaying = Boolean(ws && !ws.getMediaElement().paused);
    const currentTime = ws ? ws.getCurrentTime() : 0;
    el.video.dataset.url = state.video_url;
    el.video.src = state.video_url;
    el.video.muted = false;
    el.video.load();
    if (ws) initWaveSurfer(currentTime, wasPlaying);
  } else if (!state?.video_url && ws?.getMediaElement() === el.video) {
    const wasPlaying = !el.video.paused;
    const currentTime = ws.getCurrentTime();
    el.video.pause(); el.video.muted = true;
    initWaveSurfer(currentTime, wasPlaying);
  }
}
function updateTransport() {
  const master = ws?.getMediaElement() || el.audio;
  const t = ws ? ws.getCurrentTime() : (Number(master.currentTime) || 0);
  const tClamped = Math.min(duration(), t);
  el.current.textContent = formatTime(tClamped);
  el.total.textContent = formatTime(duration());
  el.videoTime.textContent = formatTime(tClamped);
  el.play.textContent = master.paused ? '▶' : 'Ⅱ';
  el.play.setAttribute('aria-label', master.paused ? 'Play' : 'Pause');
  el.playhead.style.left = `${duration() ? (tClamped / duration()) * 100 : 0}%`;
  if (loopBounds && tClamped >= loopBounds[1]) seekTo(loopBounds[0], true);
}
function updateSummary() {
  const kept = cutMarkers().length;
  const sel = selectedMarkers();
  el.fps.textContent = `${fps()} fps`;
  el.cutCount.textContent = `${kept} kept ${kept === 1 ? 'cut' : 'cuts'}`;
  el.markerCount.textContent = String(state.markers.length);
  el.rangeCount.textContent = String(state.no_cut_ranges.length);
  el.selection.textContent = sel.length === 0 ? 'No cut selected' : sel.length === 1 ? `Selected · ${formatTime(timeFor(sel[0].frame))} · frame ${sel[0].frame}` : `${sel.length} cuts selected`;
  el.stageMeta.textContent = `${state.output?.width || '—'} × ${state.output?.height || '—'}  ·  ${formatTime(duration(), false)}`;
  el.summary.textContent = dirty ? 'Unsaved changes · saving automatically' : `Changes saved${state.saved_at ? ` · ${formatSavedAt(state.saved_at)}` : ''}`;
  el.undo.disabled = undoStack.length === 0;
  el.redo.disabled = redoStack.length === 0;
  document.querySelectorAll('.nudge, .group-enable, .group-disable').forEach(button => { button.disabled = selected.size === 0; });
  renderPreviewState();
}
function formatSavedAt(value) {
  const d = new Date(value);
  return Number.isNaN(d.valueOf()) ? 'saved' : `saved ${d.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'})}`;
}
function markDirty() {
  dirty = fingerprint() !== lastSavedFingerprint;
  updateSummary();
  if (!dirty) { setSaveState('Saved'); return; }
  setSaveState('Saving…', 'saving');
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => saveNow(), 550);
}
function snapshotForUndo() {
  undoStack.push(cloneEdit());
  if (undoStack.length > 100) undoStack.shift();
  redoStack = [];
  updateSummary();
}
function applySnapshot(snapshot, targetStack) {
  targetStack.push(cloneEdit());
  state.markers = snapshot.markers.map(m => ({...m}));
  state.no_cut_ranges = snapshot.no_cut_ranges.map(r => [...r]);
  selected = new Set([...selected].filter(id => state.markers.some(m => m.id === id)));
  renderAll(); markDirty();
}
function undo() { if (undoStack.length) applySnapshot(undoStack.pop(), redoStack); }
function redo() { if (redoStack.length) applySnapshot(redoStack.pop(), undoStack); }
function nearestOnset(frame) {
  const times = state.onset_times || [];
  const t = timeFor(frame);
  let best = null;
  for (const onset of times) {
    const delta = Math.abs(onset - t);
    if (!best || delta < best.delta) best = {frame: Math.round(onset * fps()), delta};
  }
  return best && best.delta <= 0.12 ? clampFrame(best.frame) : frame;
}
function setMarkerFrames(ids, deltaOrFrame, absolute = false, snap = false) {
  const list = state.markers.filter(m => ids.has(m.id));
  if (!list.length) return;
  const target = absolute ? clampFrame(deltaOrFrame) : null;
  const base = list[0].frame;
  const delta = absolute ? target - base : deltaOrFrame;
  if (!delta) return;
  const proposed = new Map(list.map(m => [m.id, clampFrame(m.frame + delta)]));
  if (snap && list.length === 1) proposed.set(list[0].id, nearestOnset(proposed.get(list[0].id)));
  if (list.some(m => m.frame + delta < 1 || m.frame + delta >= totalFrames())) {
    showNotice('That move would take a cut outside the video frame range.', 3500); return;
  }
  const enabledFrames = new Set();
  for (const m of state.markers) {
    if (!m.enabled) continue;
    const frame = proposed.has(m.id) ? proposed.get(m.id) : m.frame;
    if (enabledFrames.has(frame)) { showNotice(`That move would place two kept cuts on frame ${frame}.`, 4500); return; }
    enabledFrames.add(frame);
  }
  snapshotForUndo();
  for (const m of list) {
    m.frame = proposed.get(m.id);
  }
  renderAll(); markDirty();
}
function toggleMarker(id) {
  const m = state.markers.find(x => x.id === id);
  if (!m) return;
  if (!m.enabled && state.markers.some(other => other.id !== id && other.enabled && other.frame === m.frame)) {
    showNotice(`Frame ${m.frame} already has an enabled cut. Move or skip that marker before keeping this one.`, 5000);
    return;
  }
  snapshotForUndo();
  m.enabled = !m.enabled;
  renderAll(); markDirty();
}
function setGroupEnabled(enabled) {
  if (!selected.size) { showNotice('Select one or more cut points first.', 3000); return; }
  const ids = selected;
  let targets = state.markers.filter(m => ids.has(m.id) && m.enabled !== enabled);
  if (enabled) {
    const occupied = new Set(state.markers.filter(m => m.enabled && !ids.has(m.id)).map(m => m.frame));
    const seen = new Set(state.markers.filter(m => m.enabled && ids.has(m.id)).map(m => m.frame));
    const before = targets.length;
    targets = targets.filter(m => {
      if (occupied.has(m.frame) || seen.has(m.frame)) return false;
      seen.add(m.frame); return true;
    });
    if (targets.length < before) showNotice('Some selected cuts share a frame with an enabled cut, so they could not be kept.', 4500);
  }
  if (!targets.length) return;
  snapshotForUndo(); targets.forEach(m => { m.enabled = enabled; }); renderAll(); markDirty();
}
function keepOnlyKind(kind) {
  const targets = state.markers.filter(m => m.kind === kind);
  if (!targets.length) { showNotice(`No ${labelKind(kind)} markers in this edit.`, 3000); return; }
  const seen = new Set(), ids = new Set();
  for (const m of targets) {
    if (seen.has(m.frame)) continue;
    seen.add(m.frame); ids.add(m.id);
  }
  if (state.markers.every(m => m.enabled === ids.has(m.id))) { showNotice(`Already keeping only ${labelKind(kind)} cuts.`, 2500); return; }
  snapshotForUndo();
  for (const m of state.markers) m.enabled = ids.has(m.id);
  renderAll(); markDirty();
}
function renderQuiet() {
  el.quiet.replaceChildren();
  (state.quiet_intervals || []).forEach(([start, end]) => {
    const block = document.createElement('div');
    block.className = 'quiet-span';
    block.style.left = `${Math.max(0, start / duration()) * 100}%`;
    block.style.width = `${Math.max(0, Math.min(duration(), end) - Math.max(0, start)) / duration() * 100}%`;
    el.quiet.append(block);
  });
  (state.no_cut_ranges || []).forEach((range, index) => {
    const block = document.createElement('div');
    block.className = 'saved-range-overlay';
    block.style.left = `${range[0] / totalFrames() * 100}%`;
    block.style.width = `${Math.max(1, range[1] - range[0]) / totalFrames() * 100}%`;
    block.dataset.rangeIndex = String(index);
    el.quiet.append(block);
  });
}
function renderTimeRuler() {
  if (!state) return;
  el.timeRuler.replaceChildren();
  const pxPerSecond = el.waveInner.clientWidth / duration();
  const wanted = 90 / Math.max(1, pxPerSecond);
  const power = Math.pow(10, Math.floor(Math.log10(Math.max(.001, wanted))));
  const step = [1, 2, 5, 10].map(x => x * power).find(x => x >= wanted) || 10 * power;
  for (let time = 0; time <= duration(); time += step) {
    const tick = document.createElement('div'); tick.className = 'time-tick';
    tick.style.left = `${time / duration() * 100}%`;
    const label = document.createElement('span'); label.textContent = formatTime(time, step < 1);
    tick.append(label); el.timeRuler.append(tick);
  }
}
function makeMarkerButton(m) {
  const button = document.createElement('button');
  const mutedByRange = state.no_cut_ranges.some(([a, b]) => a <= m.frame && m.frame < b);
  const effective = m.enabled && !mutedByRange;
  button.type = 'button';
  button.className = `marker ${effective ? 'is-enabled' : 'is-disabled'} ${mutedByRange && m.enabled ? 'in-no-cut-range' : ''} ${selected.has(m.id) ? 'is-selected' : ''} kind-${safeClass(m.kind)}`;
  button.style.left = `${(m.frame / totalFrames()) * 100}%`;
  button.dataset.id = m.id;
  button.setAttribute('role', 'button');
  button.setAttribute('aria-pressed', String(m.enabled));
  button.setAttribute('aria-label', `${mutedByRange && m.enabled ? 'Kept but muted by no-cuts range' : m.enabled ? 'Kept' : 'Skipped'} ${labelKind(m.kind)} at ${formatTime(timeFor(m.frame))}, frame ${m.frame}. Click to ${m.enabled ? 'skip' : 'keep'}; shift-click to multi-select.`);
  const stem = document.createElement('span'); stem.className = 'marker-stem';
  const dot = document.createElement('span'); dot.className = 'marker-dot';
  const kind = document.createElement('span'); kind.className = 'marker-kind'; kind.textContent = kindShort(m.kind);
  button.append(stem, dot, kind);
  button.addEventListener('pointerdown', (event) => {
    if (rangeMode || event.shiftKey) return;
    if (!selected.has(m.id)) {
      selected = new Set([m.id]);
      updateSelectionUI();
    }
    drag = {id: m.id, ids: new Set(selected.has(m.id) ? selected : [m.id]), originX: event.clientX, originFrame: m.frame, originals: new Map(state.markers.filter(item => selected.has(item.id) || item.id === m.id).map(item => [item.id, item.frame])), moved: false, savedUndo: false};
    button.setPointerCapture?.(event.pointerId);
  });
  button.addEventListener('click', (event) => {
    if (suppressClick) { suppressClick = false; event.preventDefault(); return; }
    if (event.shiftKey) { selectMarker(m.id, true); return; }
    if (!selected.has(m.id)) selectMarker(m.id, false);
    toggleMarker(m.id);
  });
  button.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') { event.preventDefault(); selectMarker(m.id, event.shiftKey); if (!event.shiftKey) toggleMarker(m.id); }
  });
  return button;
}
function safeClass(value) { return String(value || 'beat').replace(/[^a-z0-9_-]/gi, '').toLowerCase() || 'beat'; }
function labelKind(kind) { return ({beat: 'beat', kick: 'kick', onset: 'onset', manual: 'manual cut', automatic: 'automatic cut'})[kind] || 'cut'; }
function kindShort(kind) { return ({beat: 'B', kick: 'K', onset: 'O', manual: 'M', automatic: 'A'})[kind] || 'C'; }
function renderMarkerList() {
  el.markerList.replaceChildren();
  const sorted = [...state.markers].sort((a,b) => a.frame - b.frame);
  for (const m of sorted) {
    const row = document.createElement('div');
    const mutedByRange = state.no_cut_ranges.some(([a, b]) => a <= m.frame && m.frame < b);
    row.className = `marker-row ${selected.has(m.id) ? 'is-selected' : ''} ${m.enabled ? '' : 'is-disabled'} ${mutedByRange && m.enabled ? 'in-no-cut-range' : ''}`;
    row.setAttribute('role', 'option'); row.setAttribute('aria-selected', String(selected.has(m.id)));
    const select = document.createElement('button'); select.type = 'button'; select.className = 'marker-row-select';
    const swatch = document.createElement('i'); swatch.className = `row-swatch kind-${safeClass(m.kind)}`;
    const text = document.createElement('span'); text.className = 'row-main';
    const title = document.createElement('strong'); title.textContent = `${labelKind(m.kind)}${mutedByRange && m.enabled ? ' · muted by range' : ''}`;
    const details = document.createElement('small'); details.textContent = `Frame ${m.frame} · ${formatTime(timeFor(m.frame))}`;
    text.append(title, details); select.append(swatch, text);
    select.addEventListener('click', event => selectMarker(m.id, event.shiftKey));
    const toggle = document.createElement('button'); toggle.type = 'button'; toggle.className = `row-toggle ${m.enabled ? 'on' : ''}`;
    toggle.textContent = m.enabled ? 'Kept' : 'Skipped'; toggle.setAttribute('aria-label', `${m.enabled ? 'Skip' : 'Keep'} ${labelKind(m.kind)} at frame ${m.frame}`);
    toggle.addEventListener('click', () => { selectMarker(m.id, false); toggleMarker(m.id); });
    row.append(select, toggle); el.markerList.append(row);
  }
}
function renderRanges() {
  el.ranges.replaceChildren();
  if (!state.no_cut_ranges.length) {
    const empty = document.createElement('div'); empty.className = 'empty-list'; empty.textContent = 'No ranges added'; el.ranges.append(empty); return;
  }
  state.no_cut_ranges.forEach(([start, end], index) => {
    const row = document.createElement('div'); row.className = 'range-row';
    const label = document.createElement('span'); label.textContent = `Frames ${start}–${end} · end exclusive`;
    const time = document.createElement('small'); time.textContent = `${formatTime(timeFor(start))} – ${formatTime(timeFor(end))}`;
    const info = document.createElement('span'); info.className = 'range-info'; info.append(label, time);
    const remove = document.createElement('button'); remove.type = 'button'; remove.className = 'range-remove'; remove.textContent = 'Remove';
    remove.addEventListener('click', () => { snapshotForUndo(); state.no_cut_ranges.splice(index, 1); renderAll(); markDirty(); });
    row.append(info, remove); el.ranges.append(row);
  });
}
function renderExports() {
  const exports = [...(state.exports || [])].slice().reverse();
  el.exportCount.textContent = String(exports.length);
  el.exportsPanel.hidden = exports.length === 0;
  el.exportsList.replaceChildren();
  for (const item of exports.slice(0, 5)) {
    const row = document.createElement('div'); row.className = 'export-row';
    const title = document.createElement('span'); title.textContent = item.filename || 'Reviewed video';
    const revision = document.createElement('small'); revision.textContent = `Revision ${item.revision}`; title.append(revision);
    const links = document.createElement('div'); links.className = 'export-links';
    for (const [label, href] of [['Download video', item.download_url], ['Download plan', item.plan_url], ['Download validation', item.validation_url], ['Download report', item.report_url]]) {
      if (!href) continue;
      const link = document.createElement('a'); link.href = href; link.textContent = label; links.append(link);
    }
    row.append(title, links); el.exportsList.append(row);
  }
}
function renderAll() {
  if (!state) return;
  renderQuiet();
  renderTimeRuler();
  el.markers.replaceChildren(...state.markers.map(makeMarkerButton));
  renderMarkerList(); updateSummary(); updateTransport();
  renderExports();
}
function selectMarker(id, additive = false) {
  if (additive) {
    if (selected.has(id)) selected.delete(id); else selected.add(id);
  } else selected = new Set([id]);
  updateSelectionUI();
}
function updateSelectionUI() {
  el.markers.querySelectorAll('.marker').forEach(button => {
    const active = selected.has(button.dataset.id);
    button.classList.toggle('is-selected', active);
    button.setAttribute('aria-selected', String(active));
  });
  renderMarkerList(); updateSummary();
}
function updateZoom() {
  if (!ws || !state || !ws.getDecodedData()) return;
  zoomFactor = Number(el.zoom.value) || 1;
  const viewport = el.waveScroll.clientWidth || 900;
  const minPxPerSec = Math.max(28, (viewport / duration()) * zoomFactor);
  const width = Math.max(viewport, Math.ceil(duration() * minPxPerSec));
  el.waveInner.style.width = `${width}px`;
  el.waveform.style.width = `${width}px`;
  ws.zoom(minPxPerSec);
  requestAnimationFrame(() => { renderQuiet(); updateTransport(); });
  requestAnimationFrame(renderTimeRuler);
}
function initWaveSurfer(startTime = 0, resume = false) {
  if (ws) { ws.destroy(); ws = null; }
  el.audio.src = state.audio_url;
  el.audio.load();
  const media = state.video_url ? el.video : el.audio;
  const url = state.video_url ? (el.video.currentSrc || el.video.src) : state.audio_url;
  if (media === el.video) { el.video.muted = false; el.audio.pause(); }
  else { el.video.pause(); el.video.muted = true; }
  ws = WaveSurfer.create({
    container: el.waveform,
    media,
    url,
    peaks: [state.peaks || []],
    duration: duration(),
    height: 132,
    waveColor: '#405b67',
    progressColor: '#20c9b2',
    cursorColor: 'transparent',
    barWidth: 2,
    barGap: 2,
    barRadius: 2,
    normalize: false,
    interact: true,
    dragToSeek: false,
    autoScroll: false,
    autoCenter: false,
  });
  ws.on('ready', () => {
    updateZoom();
    const restorePosition = () => {
      if (startTime > 0) ws?.setTime(Math.min(startTime, duration()));
      updateTransport();
      if (resume) ws?.play().catch(() => {});
    };
    if (media === el.video && media.readyState < 1) media.addEventListener('loadedmetadata', restorePosition, {once: true});
    else restorePosition();
    renderAll();
  });
  ws.on('timeupdate', updateTransport);
  ws.on('interaction', updateTransport);
  ws.on('play', updateTransport);
  ws.on('pause', updateTransport);
  ws.on('finish', updateTransport);
  ws.on('error', error => showNotice(`Audio waveform could not load: ${error?.message || 'unknown error'}`));
}
function seekTo(time, keepLoop = false) {
  if (!keepLoop) { loopBounds = null; el.loop.classList.remove('is-active'); }
  const t = Math.max(0, Math.min(duration(), time));
  if (ws) ws.setTime(t); else el.audio.currentTime = t;
  updateTransport();
}
async function togglePlay() {
  if (!ws) return;
  try { if (ws.getMediaElement().paused) await ws.play(); else ws.pause(); }
  catch (error) { showNotice(`Playback could not start: ${error.message}`, 5000); }
  updateTransport();
}
function selectedCutBounds() {
  const active = selectedMarkers().find(m => m.enabled && !state.no_cut_ranges.some(([a, b]) => a <= m.frame && m.frame < b));
  if (!active) return null;
  const center = timeFor(active.frame);
  return [Math.max(0, center - 1), Math.min(duration(), center + 1)];
}
function auditionSelected() {
  const bounds = selectedCutBounds();
  if (!bounds) { showNotice('Select a cut point first to audition its section.', 3500); return; }
  if (loopBounds && loopBounds[0] === bounds[0] && loopBounds[1] === bounds[1]) {
    loopBounds = null; el.loop.classList.remove('is-active'); return;
  }
  loopBounds = bounds;
  seekTo(bounds[0], true);
  if (ws?.getMediaElement().paused) ws.play().catch(error => showNotice(`Playback could not start: ${error.message}`, 5000));
  el.loop.classList.add('is-active');
}
function addManualCut() {
  const frame = Math.round((ws?.getCurrentTime() || 0) * fps());
  if (frame <= 0 || frame >= totalFrames()) { showNotice('Move the playhead inside the video to add a cut.', 3000); return; }
  const existing = state.markers.find(m => m.enabled && m.frame === frame);
  if (existing) { selectMarker(existing.id); showNotice('There is already a kept cut at the playhead.', 3000); return; }
  const m = {id: `manual-${Date.now()}-${Math.random().toString(16).slice(2,7)}`, kind: 'manual', detected_time: null, original_frame: frame, frame, enabled: true};
  snapshotForUndo(); state.markers.push(m); selected = new Set([m.id]); renderAll(); markDirty();
}
function addRange(start, end) {
  start = clampRangeFrame(start); end = clampRangeFrame(end);
  if (end < start) [start, end] = [end, start];
  if (end - start < 1) { showNotice('Choose a range of at least one frame.', 3200); return; }
  snapshotForUndo();
  state.no_cut_ranges.push([start, end]);
  renderAll(); markDirty();
}
function removeRange(index) { snapshotForUndo(); state.no_cut_ranges.splice(index, 1); renderAll(); markDirty(); }

async function request(path, body, method = 'POST') {
  const response = await fetch(path, {method, headers: {'Content-Type': 'application/json', 'X-Autoedit-Token': state.csrf_token}, body: JSON.stringify(body)});
  let data;
  try { data = await response.json(); } catch { data = {}; }
  if (!response.ok) {
    const error = new Error(data.error || `Request failed (${response.status})`);
    error.status = response.status; error.details = data.details; throw error;
  }
  return data;
}
function adoptServerState(incoming, preserveEdits = false) {
  if (!incoming) return;
  const edits = preserveEdits ? cloneEdit() : null;
  state = {...state, ...incoming};
  if (edits) { state.markers = edits.markers; state.no_cut_ranges = edits.no_cut_ranges; }
  if (!preserveEdits) {
    lastSavedFingerprint = fingerprint(); dirty = false;
  }
  renderAll();
}
async function saveNow() {
  clearTimeout(saveTimer);
  if (!state || !dirty) return;
  if (saving) { saveAgain = true; return; }
  saving = true;
  setSaveState('Saving…', 'saving');
  const snapshot = cloneEdit();
  const sentFingerprint = fingerprint(snapshot);
  try {
    const result = await request('/api/session', {revision: state.revision, markers: snapshot.markers, no_cut_ranges: snapshot.no_cut_ranges});
    const editsChanged = fingerprint() !== sentFingerprint;
    state = {...state, ...result, markers: editsChanged ? state.markers : result.markers, no_cut_ranges: editsChanged ? state.no_cut_ranges : result.no_cut_ranges};
    lastSavedFingerprint = sentFingerprint;
    dirty = fingerprint() !== lastSavedFingerprint;
    setSaveState(dirty ? 'Saving…' : 'Saved', dirty ? 'saving' : 'saved');
    clearNotice(); renderAll();
  } catch (error) {
    if (error.status === 409) showNotice(`This edit changed elsewhere. Reload the page to review the latest saved version. ${error.message}`);
    else showNotice(`Could not save edits. ${error.message}`, 0);
    setSaveState('Save failed', 'error');
  } finally {
    saving = false;
    if (saveAgain || dirty && !el.notice.hidden) { saveAgain = false; }
    else if (dirty) { saveTimer = setTimeout(saveNow, 500); }
    else updateSummary();
  }
}
async function flushSave() {
  clearTimeout(saveTimer);
  if (dirty) await saveNow();
  while (saving) await new Promise(resolve => setTimeout(resolve, 60));
  if (dirty) throw new Error('Edits are still unsaved. Resolve the save error before continuing.');
}
async function sendPreview() {
  try {
    clearNotice(); await flushSave();
    const result = await request('/api/preview', {revision: state.revision});
    state.job = result.job; jobBusy = true; renderAll(); pollJob();
  } catch (error) { showNotice(error.message); }
}
async function sendExport() {
  try {
    clearNotice(); await flushSave();
    const result = await request('/api/export', {revision: state.revision, filename: el.exportFilename.value.trim(), overwrite: el.exportOverwrite.checked});
    state.job = result.job; jobBusy = true; renderAll(); pollJob();
  } catch (error) { showNotice(error.message); }
}
function addNoticeLink(label, href) {
  if (!href) return;
  const link = document.createElement('a'); link.href = href; link.textContent = label;
  if (label.startsWith('Open')) { link.target = '_blank'; link.rel = 'noopener'; }
  el.noticeLinks.append(link);
}
async function pollJob() {
  if (polling) return;
  polling = true;
  try {
    while (state?.job && ['queued', 'running'].includes(state.job.status)) {
      await new Promise(resolve => setTimeout(resolve, 1000));
      const response = await fetch('/api/state', {cache: 'no-store'});
      if (!response.ok) throw new Error(`Status check failed (${response.status})`);
      const incoming = await response.json();
      const previous = state.job;
      adoptServerState(incoming, dirty);
      if (state.job?.status !== previous?.status || state.job?.message !== previous?.message) {
        const label = state.job?.kind === 'export' ? 'Export' : 'Preview';
        el.summary.textContent = state.job?.message || `${label} ${state.job?.status || 'in progress'}…`;
      }
    }
    if (state?.job?.status === 'failed') showNotice(`${state.job.kind === 'export' ? 'Export' : 'Preview'} failed: ${state.job.error || state.job.message || 'Unknown error'}`);
    else if (state?.job?.status === 'completed') {
      if (state.job.kind === 'export' && state.job.download_url) {
        showNotice(`Export complete · ${state.job.filename || state.job.output?.split(/[\\/]/).pop() || 'reviewed video'}`, 0);
        addNoticeLink('Download video', state.job.download_url);
        addNoticeLink('Download plan', state.job.plan_url);
        addNoticeLink('Download validation', state.job.validation_url);
        addNoticeLink('Download report', state.job.report_url);
      }
      else showNotice('Preview updated.', 4000);
    }
    jobBusy = false; renderAll();
  } catch (error) { showNotice(`Could not check render status: ${error.message}`); jobBusy = false; }
  finally { polling = false; }
}
async function resetEdits() {
  if (!confirm('Reset all review edits to the original automatic edit?')) return;
  try {
    clearNotice(); clearTimeout(saveTimer);
    const result = await request('/api/reset', {revision: state.revision});
    state = result; selected.clear(); undoStack = []; redoStack = []; dirty = false;
    lastSavedFingerprint = fingerprint(); renderAll(); setSaveState('Saved');
  } catch (error) { showNotice(`Could not reset edits. ${error.message}`); }
}
async function init() {
  try {
    const response = await fetch('/api/state', {cache: 'no-store'});
    if (!response.ok) throw new Error(`Could not load editor (${response.status})`);
    state = await response.json();
    if (!state.markers || !state.audio_url) throw new Error('The server returned incomplete edit data.');
    el.title.textContent = state.title || state.plan_name || 'Untitled edit';
    el.exportFilename.value = state.export_filename || `${state.title || 'reviewed-edit'}.mp4`;
    selected.clear();
    lastSavedFingerprint = fingerprint(); dirty = false;
    setSaveState('Saved');
    renderPreviewState();
    initWaveSurfer();
    renderAll();
    if (state.job && ['queued', 'running'].includes(state.job.status)) { jobBusy = true; pollJob(); }
    requestAnimationFrame(updateZoom);
    window.addEventListener('resize', updateZoom);
  } catch (error) {
    el.title.textContent = 'Could not load edit'; setSaveState('Connection error', 'error');
    showNotice(error.message);
  }
}

window.autoeditReview = {
  getState: () => state ? ({revision: state.revision, markers: state.markers.map(m => ({id: m.id, frame: m.frame, enabled: m.enabled, kind: m.kind})), no_cut_ranges: state.no_cut_ranges.map(r => [...r]), cut_frames: state.cut_frames, preview_revision: state.preview_revision, job: state.job}) : null,
  getSelectedIds: () => [...selected],
  play: togglePlay,
};

// Marker drag gestures move on the integer output frame grid. Hold Alt/Option to bypass onset snap.
window.addEventListener('pointermove', event => {
  if (drag) {
    const movement = Math.abs(event.clientX - drag.originX);
    if (movement > 3) drag.moved = true;
    if (drag.moved) {
      const rect = el.waveInner.getBoundingClientRect();
      const delta = Math.round((event.clientX - drag.originX) / rect.width * totalFrames());
      const proposed = new Map();
      for (const m of state.markers) {
        if (!drag.ids.has(m.id)) continue;
        let frame = drag.originals.get(m.id) + delta;
        if (frame < 1 || frame >= totalFrames()) {
          if (!drag.blockedOnce) showNotice('That move would take a cut outside the video frame range.', 2500);
          drag.blockedOnce = true; return;
        }
        if (drag.ids.size === 1 && el.snap.checked && !event.altKey) frame = nearestOnset(frame);
        proposed.set(m.id, frame);
      }
      const proposedFrames = new Set();
      for (const m of state.markers) {
        if (!m.enabled) continue;
        const frame = proposed.has(m.id) ? proposed.get(m.id) : m.frame;
        if (proposedFrames.has(frame)) {
          if (!drag.blockedOnce) showNotice(`That move would place two kept cuts on frame ${frame}.`, 3500);
          drag.blockedOnce = true; return;
        }
        proposedFrames.add(frame);
      }
      if (!drag.savedUndo) { snapshotForUndo(); drag.savedUndo = true; }
      state.markers.forEach(m => { if (proposed.has(m.id)) m.frame = proposed.get(m.id); });
      drag.blockedOnce = false;
      renderAll(); markDirty();
    }
  }
  if (drawingRange) {
    const rect = el.waveInner.getBoundingClientRect();
    drawingRange.end = clampRangeFrame(((event.clientX - rect.left) / rect.width) * totalFrames());
    drawingRange.element.style.left = `${Math.min(drawingRange.start, drawingRange.end) / totalFrames() * 100}%`;
    drawingRange.element.style.width = `${Math.max(1, Math.abs(drawingRange.end - drawingRange.start)) / totalFrames() * 100}%`;
  }
});
window.addEventListener('pointerup', () => {
  if (drag) {
    if (drag.moved) { suppressClick = true; setTimeout(() => { suppressClick = false; }, 0); }
    drag = null;
  }
  if (drawingRange) {
    const {start, end, element} = drawingRange;
    element.remove(); drawingRange = null;
    if (Math.abs(end - start) >= 1) addRange(start, end);
  }
});
el.waveInner.addEventListener('pointerdown', event => {
  if (!rangeMode || event.target.closest('.marker')) return;
  event.preventDefault();
  const rect = el.waveInner.getBoundingClientRect();
  const start = clampRangeFrame(((event.clientX - rect.left) / rect.width) * totalFrames());
  const shade = document.createElement('div'); shade.className = 'range-drawing'; shade.style.left = `${start / totalFrames() * 100}%`; shade.style.width = '1px';
  el.waveInner.append(shade); drawingRange = {start, end: start, element: shade};
});

el.play.addEventListener('click', togglePlay);
// This overlay survives Wavesurfer reloads, so bind its seek handler once.
el.waveInner.addEventListener('click', event => {
  if (!state || event.target.closest('.marker') || rangeMode || suppressClick || drawingRange) return;
  const rect = el.waveInner.getBoundingClientRect();
  seekTo(((event.clientX - rect.left) / rect.width) * duration());
});
el.videoStage.addEventListener('click', event => { if (!event.target.closest('.video-overlay')) togglePlay(); });
el.loop.addEventListener('click', auditionSelected);
el.add.addEventListener('click', addManualCut);
el.jump.addEventListener('click', () => { const bounds = selectedCutBounds(); if (bounds) seekTo(bounds[0]); else showNotice('Select a cut point first.', 3000); });
el.zoom.addEventListener('input', updateZoom);
$('zoom-in').addEventListener('click', () => { el.zoom.value = String(Math.min(8, Number(el.zoom.value) + 1)); updateZoom(); });
$('zoom-out').addEventListener('click', () => { el.zoom.value = String(Math.max(1, Number(el.zoom.value) - 1)); updateZoom(); });
el.rangeMode.addEventListener('click', () => {
  rangeMode = !rangeMode; el.rangeMode.classList.toggle('is-active', rangeMode); el.rangeMode.setAttribute('aria-pressed', String(rangeMode));
  el.waveInner.classList.toggle('range-mode', rangeMode);
  el.hint.textContent = rangeMode ? 'Drag across the waveform to mark a no-cuts range' : 'Click a beat to keep or skip it · drag to fine-tune';
});
document.querySelectorAll('.nudge').forEach(button => button.addEventListener('click', () => {
  if (!selected.size) { showNotice('Select one or more cut points first.', 3000); return; }
  setMarkerFrames(selected, Number(button.dataset.nudge));
}));
$('keep-selected').addEventListener('click', () => setGroupEnabled(true));
$('skip-selected').addEventListener('click', () => setGroupEnabled(false));
$('keep-kicks').addEventListener('click', () => keepOnlyKind('kick'));
$('keep-beats').addEventListener('click', () => keepOnlyKind('beat'));
$('range-form').addEventListener('submit', event => {
  event.preventDefault(); addRange(Number(el.rangeStart.value), Number(el.rangeEnd.value));
});
el.undo.addEventListener('click', undo); el.redo.addEventListener('click', redo);
el.reset.addEventListener('click', resetEdits); el.preview.addEventListener('click', sendPreview);
el.export.addEventListener('click', () => { el.exportFilename.value = state?.export_filename || el.exportFilename.value; el.exportDialog.showModal(); });
el.exportForm.addEventListener('submit', event => { event.preventDefault(); if (!el.exportForm.reportValidity()) return; el.exportDialog.close(); sendExport(); });
$('export-cancel').addEventListener('click', () => el.exportDialog.close());
$('notice-dismiss').addEventListener('click', clearNotice);
document.addEventListener('keydown', event => {
  const tag = event.target?.tagName?.toLowerCase();
  const editing = ['input', 'textarea', 'select'].includes(tag) || event.target?.isContentEditable;
  const modifier = event.ctrlKey || event.metaKey;
  if (editing) return;
  if (modifier && event.key.toLowerCase() === 'z') { event.preventDefault(); event.shiftKey ? redo() : undo(); return; }
  if (modifier && event.key.toLowerCase() === 'y') { event.preventDefault(); redo(); return; }
  if (modifier && event.key.toLowerCase() === 'a') { event.preventDefault(); selected = new Set(state.markers.map(m => m.id)); renderAll(); return; }
  if (event.code === 'Space') { event.preventDefault(); togglePlay(); }
  if (event.key === 'ArrowLeft') { event.preventDefault(); if (selected.size) setMarkerFrames(selected, -1); else seekTo((ws?.getCurrentTime() || 0) - 1 / fps()); }
  if (event.key === 'ArrowRight') { event.preventDefault(); if (selected.size) setMarkerFrames(selected, 1); else seekTo((ws?.getCurrentTime() || 0) + 1 / fps()); }
  if (event.key.toLowerCase() === 'k') setGroupEnabled(true);
  if (event.key.toLowerCase() === 's') setGroupEnabled(false);
  if (event.key === 'Escape' && rangeMode) el.rangeMode.click();
});

init();
