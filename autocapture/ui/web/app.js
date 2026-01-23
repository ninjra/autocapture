const tabs = document.querySelectorAll('.nav-tabs button');
const sections = document.querySelectorAll('.tab');
const urlParams = new URLSearchParams(window.location.search);
const unlockToken = urlParams.get('unlock');
let sessionToken = unlockToken || sessionStorage.getItem('autocaptureUnlock') || '';
let unlockPromise = null;
let unlockBlockedUntil = 0;
let unlockWarning = null;
const perfProfileSelect = document.getElementById('perfProfileSelect');
const applyPerfProfile = document.getElementById('applyPerfProfile');
const perfLogComponent = document.getElementById('perfLogComponent');
const refreshPerfLog = document.getElementById('refreshPerfLog');

function apiHeaders() {
  const headers = { 'Content-Type': 'application/json' };
  if (sessionToken) {
    headers['Authorization'] = `Bearer ${sessionToken}`;
  }
  return headers;
}

function withUnlock(url) {
  if (!sessionToken) return url;
  const separator = url.includes('?') ? '&' : '?';
  return `${url}${separator}unlock=${encodeURIComponent(sessionToken)}`;
}

async function apiFetch(path, options = {}) {
  const { _retryUnlock, ...fetchOptions } = options;
  const headers = { ...apiHeaders(), ...(options.headers || {}) };
  let response = await fetch(path, { ...fetchOptions, headers });
  if (
    response.status === 401 &&
    !_retryUnlock &&
    !path.startsWith('/api/unlock') &&
    !path.startsWith('/api/lock')
  ) {
    try {
      await ensureUnlocked();
      const retryHeaders = { ...apiHeaders(), ...(options.headers || {}) };
      response = await fetch(path, { ...fetchOptions, headers: retryHeaders });
      if (response.status !== 401) {
        setUnlockWarning(null);
      }
    } catch (err) {
      setUnlockWarning('Unlock required');
    }
  }
  return response;
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return 'n/a';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let size = bytes;
  let idx = 0;
  while (size >= 1024 && idx < units.length - 1) {
    size /= 1024;
    idx += 1;
  }
  return `${size.toFixed(1)} ${units[idx]}`;
}

function formatNumber(value, digits = 1) {
  if (!Number.isFinite(value)) return 'n/a';
  return value.toFixed(digits);
}

function formatRate(value, total = null) {
  if (!Number.isFinite(value)) {
    if (Number.isFinite(total)) return 'warming up';
    return 'n/a';
  }
  return `${value.toFixed(2)}/min`;
}

function formatMs(value) {
  if (!Number.isFinite(value)) return 'n/a';
  if (value >= 1000) return `${(value / 1000).toFixed(2)} s`;
  return `${value.toFixed(0)} ms`;
}

function formatPercent(value) {
  if (!Number.isFinite(value)) return 'n/a';
  return `${value.toFixed(1)}%`;
}

function setSessionToken(token) {
  sessionToken = token || '';
  if (sessionToken) {
    sessionStorage.setItem('autocaptureUnlock', sessionToken);
  } else {
    sessionStorage.removeItem('autocaptureUnlock');
  }
}

async function ensureUnlocked() {
  const now = Date.now();
  if (unlockBlockedUntil && now < unlockBlockedUntil) {
    throw new Error('unlock blocked');
  }
  if (unlockPromise) return unlockPromise;
  unlockPromise = (async () => {
    const response = await fetch('/api/unlock', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
    if (!response.ok) {
      throw new Error('unlock failed');
    }
    const payload = await response.json();
    if (!payload || !payload.token) {
      throw new Error('unlock missing token');
    }
    setSessionToken(payload.token);
    setUnlockWarning(null);
    return payload.token;
  })()
    .catch((err) => {
      unlockBlockedUntil = Date.now() + 60000;
      throw err;
    })
    .finally(() => {
      unlockPromise = null;
    });
  return unlockPromise;
}

function setUnlockWarning(message) {
  unlockWarning = message;
  renderStatusBar(null);
}

function normalizeCpuPercent(proc) {
  if (!proc) return null;
  let value = Number.isFinite(proc.cpu_percent_total)
    ? proc.cpu_percent_total
    : proc.cpu_percent;
  if (!Number.isFinite(value)) return null;
  const count = Number.isFinite(proc.cpu_count)
    ? proc.cpu_count
    : (Number.isFinite(navigator.hardwareConcurrency) ? navigator.hardwareConcurrency : null);
  if (value > 100 && count && count > 0) {
    value /= count;
  }
  return value;
}

function createStateStore(onUpdate, onError) {
  let delay = 2000;
  let timer = null;
  let stopped = false;

  async function poll() {
    if (stopped) return;
    if (document.hidden) {
      schedule(delay);
      return;
    }
    try {
      const response = await apiFetch('/api/state', { headers: {} });
      if (!response.ok) throw new Error('state fetch failed');
      const data = await response.json();
      delay = 2000;
      if (onUpdate) onUpdate(data);
    } catch (err) {
      delay = Math.min(delay * 2, 30000);
      if (onError) onError(err);
    }
    schedule(delay);
  }

  function schedule(ms) {
    clearTimeout(timer);
    timer = setTimeout(poll, ms);
  }

  function start() {
    stopped = false;
    poll();
  }

  function stop() {
    stopped = true;
    clearTimeout(timer);
  }

  document.addEventListener('visibilitychange', () => {
    if (!document.hidden && !stopped) {
      delay = 2000;
      poll();
    }
  });

  return { start, stop };
}

function renderStatusBar(state) {
  const container = document.getElementById('statusBar');
  if (!container) return;
  container.textContent = '';
  if (unlockWarning) {
    const chip = document.createElement('div');
    chip.className = 'status-chip blocked';
    chip.textContent = unlockWarning;
    chip.addEventListener('click', async () => {
      try {
        await ensureUnlocked();
        setUnlockWarning(null);
      } catch (err) {
        setUnlockWarning('Unlock required');
      }
    });
    container.appendChild(chip);
    if (!state) return;
  }
  if (!state) return;
  const overall = state.health ? state.health.overall : 'unknown';
  const chip = document.createElement('div');
  chip.className = `status-chip ${overall}`;
  chip.innerHTML = `<strong>${overall.toUpperCase()}</strong> State`;
  container.appendChild(chip);

  if (state.health && Array.isArray(state.health.issues) && state.health.issues.length > 0) {
    const issues = document.createElement('div');
    issues.className = 'status-chip';
    issues.textContent = `Issues: ${state.health.issues.length}`;
    container.appendChild(issues);
  }

  const mode = document.createElement('div');
  mode.className = 'status-chip';
  mode.textContent = `Mode: ${state.app?.mode || '--'}`;
  container.appendChild(mode);

  if (state.privacy) {
    const privacy = document.createElement('div');
    privacy.className = 'status-chip';
    privacy.textContent = state.privacy.paused ? 'Capture paused' : 'Capture active';
    container.appendChild(privacy);
  }

  if (state.lock && state.lock.required) {
    const lock = document.createElement('div');
    lock.className = `status-chip ${state.lock.unlocked ? 'ok' : 'blocked'}`;
    lock.textContent = state.lock.unlocked ? 'Unlocked' : 'Unlock required';
    container.appendChild(lock);
  }

  if (state.storage) {
    const storage = document.createElement('div');
    storage.className = 'status-chip';
    storage.textContent = `Storage: ${formatBytes(state.storage.media_usage_bytes)}`;
    container.appendChild(storage);
  }

  if (state.queues) {
    const queues = document.createElement('div');
    queues.className = 'status-chip';
    queues.textContent = `OCR ${state.queues.ocr_pending}/${state.queues.ocr_processing}`;
    container.appendChild(queues);
  }

  if (Array.isArray(state.components)) {
    const stale = state.components.filter((c) => c.stale).length;
    const comp = document.createElement('div');
    comp.className = 'status-chip';
    comp.textContent = stale ? `Components stale: ${stale}` : 'Components healthy';
    container.appendChild(comp);
  }
}

function renderPerfPanel(state) {
  const panel = document.getElementById('perfPanel');
  if (!panel) return;
  panel.textContent = '';
  if (!state || !Array.isArray(state.components)) {
    panel.textContent = 'Performance data unavailable.';
    return;
  }
  const runtime = state.components.find((c) => c.component === 'runtime');
  const api = state.components.find((c) => c.component === 'api');
  const runtimePerf = runtime && runtime.signals ? runtime.signals.perf : null;
  const apiPerf = api && api.signals ? api.signals.perf : null;
  if (!runtimePerf && !apiPerf) {
    panel.textContent = 'Performance data unavailable.';
    return;
  }

  const addCard = (title, rows) => {
    const card = document.createElement('div');
    card.className = 'perf-card';
    const heading = document.createElement('h3');
    heading.textContent = title;
    card.appendChild(heading);
    rows.forEach(([label, value]) => {
      const row = document.createElement('div');
      row.className = 'perf-metric';
      const name = document.createElement('div');
      name.textContent = label;
      const val = document.createElement('span');
      val.textContent = value;
      row.appendChild(name);
      row.appendChild(val);
      card.appendChild(row);
    });
    panel.appendChild(card);
  };

  if (runtimePerf) {
    const proc = runtimePerf.process || {};
    const profile = runtimePerf.profile || {};
    const profileLabel = profile.active
      ? `${profile.active}${profile.override ? ' (override)' : ''}`
      : 'n/a';
    const cpuPercent = normalizeCpuPercent(proc);
    addCard('Runtime', [
      ['CPU', formatPercent(cpuPercent)],
      ['Memory', proc.rss_mb ? `${formatNumber(proc.rss_mb)} MB` : 'n/a'],
      ['Profile', profileLabel],
      ['Mode', profile.mode || 'n/a'],
    ]);
  }

  if (runtimePerf && runtimePerf.gpu && runtimePerf.gpu.available) {
    const gpu = runtimePerf.gpu;
    addCard('GPU', [
      ['Utilization', formatPercent(gpu.utilization_percent)],
      ['Memory', gpu.memory_used_mb ? `${formatNumber(gpu.memory_used_mb)} MB` : 'n/a'],
    ]);
  }

  if (runtimePerf && runtimePerf.captures) {
    const captures = runtimePerf.captures || {};
    addCard('Captures', [
      ['Rate', formatRate(captures.per_min, captures.total)],
      ['Total', formatNumber(captures.total, 0)],
      ['Dropped', formatNumber(captures.dropped, 0)],
      ['Backpressure', formatNumber(captures.skipped_backpressure, 0)],
    ]);
  }

  if (runtimePerf && runtimePerf.queues) {
    const queues = runtimePerf.queues || {};
    addCard('Queues', [
      ['OCR backlog', formatNumber(queues.ocr_backlog, 0)],
      ['Embed backlog', formatNumber(queues.embedding_backlog, 0)],
      ['ROI depth', formatNumber(queues.roi_queue_depth, 0)],
      ['Enrich backlog', formatNumber(queues.enrichment_backlog, 0)],
    ]);
  }

  if (runtimePerf && runtimePerf.latency_ms) {
    const lat = runtimePerf.latency_ms || {};
    const ocr = lat.ocr || {};
    const embed = lat.embedding || {};
    const retrieval = lat.retrieval || {};
    addCard('Latency (p50/p95)', [
      ['OCR', `${formatMs(ocr.p50)} / ${formatMs(ocr.p95)}`],
      ['Embed', `${formatMs(embed.p50)} / ${formatMs(embed.p95)}`],
      ['Retrieval', `${formatMs(retrieval.p50)} / ${formatMs(retrieval.p95)}`],
    ]);
  }

  if (apiPerf) {
    const proc = apiPerf.process || {};
    addCard('API', [
      ['CPU', formatPercent(proc.cpu_percent)],
      ['Memory', proc.rss_mb ? `${formatNumber(proc.rss_mb)} MB` : 'n/a'],
      ['Uptime', apiPerf.uptime_s ? `${formatNumber(apiPerf.uptime_s, 0)} s` : 'n/a'],
    ]);
  }

  if (perfProfileSelect && runtimePerf && runtimePerf.profile) {
    const override = runtimePerf.profile.override || 'auto';
    perfProfileSelect.value = override;
  }
}

async function loadPerfLog() {
  const target = document.getElementById('perfLogEntries');
  const status = document.getElementById('perfLogStatus');
  if (!target) return;
  target.textContent = '';
  if (status) status.textContent = 'Loading...';
  const component = perfLogComponent ? perfLogComponent.value : 'runtime';
  try {
    const response = await apiFetch(
      `/api/perf/log?component=${encodeURIComponent(component)}`
    );
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.detail || 'Perf log fetch failed');
    }
    const data = await response.json();
    const entries = Array.isArray(data.entries) ? data.entries : [];
    if (status) {
      status.textContent = entries.length
        ? `Showing ${entries.length} entries.`
        : 'No entries yet.';
    }
    entries.slice().reverse().forEach((entry) => {
      const parsed = entry.parsed || {};
      const time = parsed.time_utc || 'unknown time';
      const card = document.createElement('div');
      card.className = 'perf-log-entry';
      const title = document.createElement('h4');
      title.textContent = time;
      card.appendChild(title);
      const summary = document.createElement('div');
      summary.className = 'perf-metric';
      const captures = parsed.captures || {};
    const proc = parsed.process || {};
    const cpuPercent = normalizeCpuPercent(proc);
    summary.innerHTML = `<div>CPU</div><span>${formatPercent(
        cpuPercent
      )}</span>`;
      card.appendChild(summary);
      const captureRow = document.createElement('div');
      captureRow.className = 'perf-metric';
    captureRow.innerHTML = `<div>Capture rate</div><span>${formatRate(
        captures.per_min,
        captures.total
      )}</span>`;
      card.appendChild(captureRow);
      const queues = parsed.queues || {};
      const queueRow = document.createElement('div');
      queueRow.className = 'perf-metric';
      queueRow.innerHTML = `<div>OCR backlog</div><span>${formatNumber(
        queues.ocr_backlog,
        0
      )}</span>`;
      card.appendChild(queueRow);
      const latency = parsed.latency_ms || {};
      const ocr = latency.ocr || {};
      const latencyRow = document.createElement('div');
      latencyRow.className = 'perf-metric';
      latencyRow.innerHTML = `<div>OCR p95</div><span>${formatMs(
        ocr.p95
      )}</span>`;
      card.appendChild(latencyRow);
      if (!entry.parsed) {
        const raw = document.createElement('pre');
        raw.textContent = entry.raw || '';
        card.appendChild(raw);
      }
      target.appendChild(card);
    });
  } catch (err) {
    if (status) status.textContent = 'Perf log unavailable.';
  }
}

function renderBanner(banner) {
  const bannerEl = document.getElementById('answerBanner');
  if (!bannerEl) return;
  if (!banner || banner.level === 'none') {
    bannerEl.className = 'banner';
    bannerEl.textContent = '';
    return;
  }
  bannerEl.className = `banner visible ${banner.level}`;
  const title = banner.title ? `${banner.title} ` : '';
  const message = banner.message || '';
  const reasons = Array.isArray(banner.reasons) && banner.reasons.length
    ? ` (${banner.reasons.join(', ')})`
    : '';
  bannerEl.textContent = `${title}${message}${reasons}`;
}

function getPath(obj, path) {
  if (!path) return undefined;
  return path.split('.').reduce((acc, key) => (acc ? acc[key] : undefined), obj);
}

function setPath(obj, path, value) {
  const keys = path.split('.');
  let current = obj;
  keys.forEach((key, idx) => {
    if (idx === keys.length - 1) {
      current[key] = value;
    } else {
      if (!current[key] || typeof current[key] !== 'object') {
        current[key] = {};
      }
      current = current[key];
    }
  });
}

function deepClone(obj) {
  return JSON.parse(JSON.stringify(obj || {}));
}

function buildCandidate(base, draft) {
  if (Array.isArray(base) || Array.isArray(draft)) {
    return JSON.stringify(base) === JSON.stringify(draft) ? undefined : draft;
  }
  if (typeof base !== 'object' || typeof draft !== 'object' || !base || !draft) {
    return base === draft ? undefined : draft;
  }
  const out = {};
  Object.keys(draft).forEach((key) => {
    const candidate = buildCandidate(base[key], draft[key]);
    if (candidate !== undefined) {
      out[key] = candidate;
    }
  });
  return Object.keys(out).length ? out : undefined;
}

async function populateSelect(kind, selectEl) {
  if (!selectEl) return;
  let data;
  try {
    const response = await apiFetch(`/api/plugins/extensions?kind=${encodeURIComponent(kind)}`);
    if (!response.ok) return;
    data = await response.json();
  } catch (err) {
    return;
  }
  const extensions = Array.isArray(data.extensions) ? data.extensions : [];
  selectEl.textContent = '';
  if (extensions.length === 0) {
    const fallback = document.createElement('option');
    fallback.value = 'ollama';
    fallback.textContent = 'ollama (fallback)';
    selectEl.appendChild(fallback);
    return;
  }
  extensions.forEach((ext) => {
    const option = document.createElement('option');
    option.value = ext.id;
    option.textContent = ext.name || ext.id;
    selectEl.appendChild(option);
  });
}

function ensureOption(selectEl, value) {
  if (!selectEl || !value) return;
  const options = Array.from(selectEl.options || []);
  if (options.some((opt) => opt.value === value)) return;
  const option = document.createElement('option');
  option.value = value;
  option.textContent = `${value} (unavailable)`;
  option.dataset.missing = 'true';
  selectEl.appendChild(option);
  selectEl.value = value;
}

async function loadRoutingOptions() {
  await populateSelect('llm.provider', modelSelect);
}

function appendMessage(author, text) {
  const div = document.createElement('div');
  div.className = 'msg';
  const strong = document.createElement('strong');
  strong.textContent = author;
  const span = document.createElement('span');
  span.textContent = text;
  div.appendChild(strong);
  div.appendChild(span);
  thread.appendChild(div);
  thread.scrollTop = thread.scrollHeight;
}

function appendAssistantMessage(answer, citations, contextPack, meta = {}) {
  const div = document.createElement('div');
  div.className = 'msg';
  const strong = document.createElement('strong');
  strong.textContent = 'Assistant';
  const span = document.createElement('span');
  span.textContent = answer;
  div.appendChild(strong);
  div.appendChild(span);
  if (meta && meta.mode) {
    const badge = document.createElement('small');
    badge.className = 'mode-badge';
    badge.textContent = `Mode: ${meta.mode}`;
    div.appendChild(badge);
  }
  if (meta && meta.evidence_summary) {
    const summary = meta.evidence_summary;
    const summaryEl = document.createElement('small');
    summaryEl.className = 'citations';
    summaryEl.textContent = `Evidence: ${summary.total} (citable ${summary.citable})`;
    div.appendChild(summaryEl);
  }
  if (Array.isArray(citations) && citations.length > 0) {
    const citationText = document.createElement('small');
    citationText.className = 'citations';
    citationText.textContent = `Citations: ${citations.join(', ')}`;
    div.appendChild(citationText);
    if (contextPack && Array.isArray(contextPack.evidence)) {
      const actions = document.createElement('div');
      actions.className = 'citation-actions';
      citations.forEach((cite) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.textContent = `Overlay ${cite}`;
        btn.addEventListener('click', () => {
          showCitationOverlay(cite, contextPack);
        });
        actions.appendChild(btn);
      });
      div.appendChild(actions);
    }
  }
  if (meta && Array.isArray(meta.hints) && meta.hints.length > 0) {
    const hints = document.createElement('div');
    hints.className = 'hints';
    hints.textContent = `Hints: ${meta.hints.map((h) => h.label || '').join(', ')}`;
    div.appendChild(hints);
  }
  if (meta && Array.isArray(meta.actions) && meta.actions.length > 0) {
    const actions = document.createElement('div');
    actions.className = 'actions';
    meta.actions.forEach((action) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.textContent = action.label || 'Action';
      btn.addEventListener('click', () => {
        if (action.type === 'time_range' && action.value) {
          timeRange.value = action.value;
        } else if (action.type === 'refine_query') {
          chatInput.value = action.value || '';
          chatInput.focus();
        }
      });
      actions.appendChild(btn);
    });
    div.appendChild(actions);
  }
  thread.appendChild(div);
  thread.scrollTop = thread.scrollHeight;
}

async function showCitationOverlay(citationId, contextPack) {
  const evidenceList = Array.isArray(contextPack.evidence) ? contextPack.evidence : [];
  const item = evidenceList.find((entry) => entry && entry.id === citationId);
  if (!item || !item.meta) {
    appendAssistantMessage('No overlay available for that citation.', []);
    return;
  }
  const eventId = item.meta.event_id;
  const spans = Array.isArray(item.meta.spans) ? item.meta.spans : [];
  const spanIds = spans.map((span) => span.span_id).filter((id) => id);
  const bboxes = [];
  let useNorm = false;
  spans.forEach((span) => {
    if (Array.isArray(span.bbox_norm) && span.bbox_norm.length >= 4) {
      bboxes.push(span.bbox_norm);
      useNorm = true;
      return;
    }
    if (Array.isArray(span.bbox) && span.bbox.length >= 4) {
      bboxes.push(span.bbox);
    }
  });
  if (!eventId || bboxes.length === 0) {
    appendAssistantMessage('No overlay regions available for that citation.', []);
    return;
  }
  if (spanIds.length > 0) {
    try {
      const validate = await apiFetch('/api/citations/validate', {
        method: 'POST',
        body: JSON.stringify({ span_ids: spanIds })
      });
      if (!validate.ok) {
        appendAssistantMessage('Citation validation failed.', []);
        return;
      }
      const payload = await validate.json();
      if (payload.invalid_span_ids && Object.keys(payload.invalid_span_ids).length > 0) {
        appendAssistantMessage('Citation is no longer valid.', []);
        return;
      }
    } catch (err) {
      appendAssistantMessage('Citation validation failed.', []);
      return;
    }
  }
  let response;
  try {
    response = await apiFetch('/api/citations/overlay', {
      method: 'POST',
      body: JSON.stringify({
        event_id: eventId,
        bboxes,
        bbox_format: useNorm ? 'norm' : 'px'
      })
    });
  } catch (err) {
    appendAssistantMessage('Failed to fetch overlay from the API.', []);
    return;
  }
  if (!response.ok) {
    appendAssistantMessage('Overlay request failed.', []);
    return;
  }
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  window.open(url, '_blank');
}

function rangeToTuple(rangeKey) {
  const now = new Date();
  let start;
  if (rangeKey === '24h') {
    start = new Date(now.getTime() - 24 * 60 * 60 * 1000);
  } else if (rangeKey === '7d') {
    start = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
  } else {
    start = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
  }
  return [start.toISOString(), now.toISOString()];
}

const thread = document.getElementById('thread');
const chatForm = document.getElementById('chatForm');
const chatInput = document.getElementById('chatInput');
const modelSelect = document.getElementById('modelSelect');
const sanitizeToggle = document.getElementById('sanitizeToggle');
const extractiveToggle = document.getElementById('extractiveToggle');
const timeRange = document.getElementById('timeRange');
const safeModeIndicator = document.getElementById('safeModeIndicator');
const pluginSafeMode = document.getElementById('pluginSafeMode');
const pluginList = document.getElementById('pluginList');
const pluginWarnings = document.getElementById('pluginWarnings');

chatForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const query = chatInput.value.trim();
  if (!query) return;
  appendMessage('You', query);
  chatInput.value = '';

  const payload = {
    query,
    sanitize: sanitizeToggle.checked,
    extractive_only: extractiveToggle.checked,
    routing: { llm: modelSelect.value || 'ollama' },
    time_range: rangeToTuple(timeRange.value)
  };

  let response;
  let data = {};
  try {
    response = await apiFetch('/api/answer', {
      method: 'POST',
      body: JSON.stringify(payload)
    });
    try {
      data = await response.json();
    } catch (err) {
      data = {};
    }
  } catch (err) {
    appendAssistantMessage('Error: failed to reach the local API server.', []);
    return;
  }
  if (!response.ok) {
    const detail = typeof data.detail === 'string' ? data.detail : 'request failed';
    appendAssistantMessage(`Error: ${detail}.`, []);
    return;
  }
  renderBanner(data.banner || null);
  if (data.banner && (data.banner.level === 'no_evidence' || data.banner.level === 'locked')) {
    return;
  }
  appendAssistantMessage(
    data.answer || '',
    data.citations || [],
    data.used_context_pack || null,
    data || {}
  );
  if (safeModeIndicator) {
    const strategy = data.prompt_strategy;
    if (!strategy) {
      safeModeIndicator.textContent = '--';
      safeModeIndicator.classList.remove('safe');
    } else if (strategy.safe_mode_degraded) {
      safeModeIndicator.textContent = 'On';
      safeModeIndicator.classList.add('safe');
    } else {
      safeModeIndicator.textContent = 'Off';
      safeModeIndicator.classList.remove('safe');
    }
  }
});

const searchForm = document.getElementById('searchForm');
const searchInput = document.getElementById('searchInput');
const searchResults = document.getElementById('searchResults');

searchForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const query = searchInput.value.trim();
  if (!query) return;
  searchResults.textContent = '';
  let response;
  let data = {};
  try {
    response = await apiFetch('/api/retrieve', {
      method: 'POST',
      body: JSON.stringify({ query, k: 10, include_screenshots: true })
    });
    try {
      data = await response.json();
    } catch (err) {
      data = {};
    }
  } catch (err) {
    const card = document.createElement('div');
    card.className = 'result-card';
    card.textContent = 'failed to reach the local API server.';
    searchResults.appendChild(card);
    return;
  }
  if (!response.ok) {
    const detail = typeof data.detail === 'string' ? data.detail : 'request failed';
    const card = document.createElement('div');
    card.className = 'result-card';
    card.textContent = `Error: ${detail}.`;
    searchResults.appendChild(card);
    return;
  }
  if (data && data.no_evidence) {
    const card = document.createElement('div');
    card.className = 'result-card';
    card.textContent = data.message || 'No evidence found.';
    searchResults.appendChild(card);
    return;
  }
  const evidence = Array.isArray(data.evidence) ? data.evidence : [];
  evidence.forEach((item) => {
    const card = document.createElement('div');
    card.className = 'result-card';
    const strong = document.createElement('strong');
    strong.textContent = item.app || 'Unknown app';
    const paragraph = document.createElement('p');
    paragraph.textContent = item.text || '';
    const small = document.createElement('small');
    small.textContent = item.event_id || '';
    card.appendChild(strong);
    card.appendChild(paragraph);
    card.appendChild(small);
    if (item.event_id) {
      const link = document.createElement('a');
      link.href = withUnlock(`/api/screenshot/${item.event_id}`);
      link.textContent = 'View screenshot';
      link.target = '_blank';
      link.rel = 'noreferrer';
      card.appendChild(link);
    }
    searchResults.appendChild(card);
  });
});

const highlightsDay = document.getElementById('highlightsDay');
const highlightsContent = document.getElementById('highlightsContent');
const refreshHighlights = document.getElementById('refreshHighlights');

async function loadHighlightsList() {
  highlightsDay.textContent = '';
  let response;
  try {
    response = await apiFetch('/api/highlights');
  } catch (err) {
    return;
  }
  if (!response.ok) return;
  const data = await response.json();
  data.forEach((item) => {
    const option = document.createElement('option');
    option.value = item.day;
    option.textContent = item.day;
    highlightsDay.appendChild(option);
  });
  if (highlightsDay.options.length > 0) {
    await loadHighlightsDetail(highlightsDay.value);
  }
}

async function loadHighlightsDetail(day) {
  highlightsContent.textContent = '';
  if (!day) return;
  let response;
  try {
    response = await apiFetch(`/api/highlights/${day}`);
  } catch (err) {
    highlightsContent.textContent = 'Failed to load highlights.';
    return;
  }
  if (!response.ok) {
    highlightsContent.textContent = 'Highlights not available.';
    return;
  }
  const data = await response.json();
  const payload = data.data || {};
  const summary = document.createElement('p');
  summary.textContent = payload.summary || '';
  highlightsContent.appendChild(summary);

  const list = document.createElement('ul');
  (payload.highlights || []).forEach((item) => {
    const li = document.createElement('li');
    li.textContent = item;
    li.addEventListener('click', async () => {
      await apiFetch('/api/retrieve', {
        method: 'POST',
        body: JSON.stringify({ query: item, k: 5 })
      });
    });
    list.appendChild(li);
  });
  highlightsContent.appendChild(list);

  if (payload.open_loops && payload.open_loops.length) {
    const openLoops = document.createElement('div');
    openLoops.className = 'open-loops';
    const title = document.createElement('h4');
    title.textContent = 'Open loops';
    openLoops.appendChild(title);
    const openList = document.createElement('ul');
    payload.open_loops.forEach((item) => {
      const li = document.createElement('li');
      li.textContent = item;
      openList.appendChild(li);
    });
    openLoops.appendChild(openList);
    highlightsContent.appendChild(openLoops);
  }
}

refreshHighlights.addEventListener('click', async () => {
  await loadHighlightsList();
});

highlightsDay.addEventListener('change', async () => {
  await loadHighlightsDetail(highlightsDay.value);
});

let settingsSchema = null;
let settingsBase = {};
let settingsDraft = {};
let settingsTier = 'guided';
let settingsPreview = null;

const tierSelector = document.getElementById('tierSelector');
const settingsForm = document.getElementById('settingsForm');
const settingsStatus = document.getElementById('settingsStatus');
const previewSettingsBtn = document.getElementById('previewSettings');
const applySettingsBtn = document.getElementById('applySettings');
const settingsPreviewPanel = document.getElementById('settingsPreview');

function renderTierSelector() {
  if (!settingsSchema || !tierSelector) return;
  tierSelector.textContent = '';
  settingsSchema.tiers.forEach((tier) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.textContent = tier.label;
    btn.className = tier.tier_id === settingsTier ? 'active' : '';
    btn.addEventListener('click', () => {
      settingsTier = tier.tier_id;
      renderTierSelector();
      renderSettingsForm();
    });
    tierSelector.appendChild(btn);
  });
}

function renderSettingsForm() {
  if (!settingsSchema || !settingsForm) return;
  settingsForm.textContent = '';
  const tierRankMap = {};
  settingsSchema.tiers.forEach((tier) => {
    tierRankMap[tier.tier_id] = tier.rank;
  });
  const activeRank = tierRankMap[settingsTier] || 1;
  settingsSchema.sections.forEach((section) => {
    const sectionEl = document.createElement('div');
    sectionEl.className = 'preview-panel';
    const title = document.createElement('h3');
    title.textContent = section.label;
    sectionEl.appendChild(title);
    if (section.description) {
      const desc = document.createElement('p');
      desc.className = 'hint';
      desc.textContent = section.description;
      sectionEl.appendChild(desc);
    }
    section.fields.forEach((field) => {
      const fieldRank = tierRankMap[field.tier] || 1;
      if (fieldRank > activeRank) return;
      const wrapper = document.createElement('div');
      wrapper.className = 'settings-field';
      const label = document.createElement('label');
      label.textContent = field.label;
      wrapper.appendChild(label);
      if (field.description) {
        const desc = document.createElement('span');
        desc.className = 'hint';
        desc.textContent = field.description;
        wrapper.appendChild(desc);
      }
      const value = getPath(settingsDraft, field.path);
      let input;
      if (field.kind === 'bool') {
        input = document.createElement('input');
        input.type = 'checkbox';
        input.checked = Boolean(value);
        input.addEventListener('change', () => {
          setPath(settingsDraft, field.path, input.checked);
          markSettingsDirty();
        });
      } else if (field.kind === 'select') {
        input = document.createElement('select');
        if (field.options && field.options.length) {
          field.options.forEach((opt) => {
            const option = document.createElement('option');
            option.value = opt.value;
            option.textContent = opt.label;
            input.appendChild(option);
          });
        }
        input.addEventListener('change', () => {
          setPath(settingsDraft, field.path, input.value);
          markSettingsDirty();
        });
        if (field.options_source && field.options_source.startsWith('plugins:')) {
          loadOptionsSource(field.options_source, input, value);
        } else if (value) {
          input.value = value;
        }
      } else if (field.kind === 'list') {
        input = document.createElement('textarea');
        input.value = Array.isArray(value) ? value.join('\n') : '';
        input.placeholder = field.placeholder || '';
        input.addEventListener('change', () => {
          const list = input.value.split('\n').map((line) => line.trim()).filter((line) => line);
          setPath(settingsDraft, field.path, list);
          markSettingsDirty();
        });
      } else if (field.kind === 'json') {
        input = document.createElement('textarea');
        input.value = value ? JSON.stringify(value, null, 2) : '';
        input.placeholder = field.placeholder || '{ }';
        input.addEventListener('change', () => {
          try {
            const parsed = input.value ? JSON.parse(input.value) : null;
            setPath(settingsDraft, field.path, parsed);
            markSettingsDirty();
          } catch (err) {
            input.value = value ? JSON.stringify(value, null, 2) : '';
          }
        });
      } else if (field.kind === 'int' || field.kind === 'float') {
        input = document.createElement('input');
        input.type = 'number';
        input.step = field.kind === 'float' ? '0.01' : '1';
        input.value = value === undefined || value === null ? '' : value;
        input.placeholder = field.placeholder || '';
        input.addEventListener('change', () => {
          const raw = input.value.trim();
          if (!raw) {
            setPath(settingsDraft, field.path, null);
          } else {
            const parsed = field.kind === 'float' ? parseFloat(raw) : parseInt(raw, 10);
            setPath(settingsDraft, field.path, Number.isNaN(parsed) ? null : parsed);
          }
          markSettingsDirty();
        });
      } else {
        input = document.createElement('input');
        input.type = 'text';
        input.value = value === undefined || value === null ? '' : value;
        input.placeholder = field.placeholder || '';
        input.addEventListener('change', () => {
          setPath(settingsDraft, field.path, input.value);
          markSettingsDirty();
        });
      }
      wrapper.appendChild(input);
      sectionEl.appendChild(wrapper);
    });
    settingsForm.appendChild(sectionEl);
  });
}

async function loadOptionsSource(source, selectEl, currentValue) {
  const kind = source.split(':')[1];
  await populateSelect(kind, selectEl);
  if (currentValue) {
    selectEl.value = currentValue;
    ensureOption(selectEl, currentValue);
  }
}

function markSettingsDirty() {
  if (!settingsStatus) return;
  settingsStatus.textContent = 'Draft modified. Preview to continue.';
}

async function loadSettingsSchema() {
  try {
    const response = await apiFetch('/api/settings/schema');
    if (!response.ok) return;
    settingsSchema = await response.json();
  } catch (err) {
    return;
  }
  renderTierSelector();
}

async function loadSettingsEffective() {
  try {
    const response = await apiFetch('/api/settings/effective');
    if (!response.ok) return;
    const data = await response.json();
    settingsBase = data.settings || {};
    settingsDraft = deepClone(settingsBase);
    renderSettingsForm();
    if (settingsStatus) {
      settingsStatus.textContent = 'No pending changes';
    }
    if (modelSelect && settingsBase.routing && settingsBase.routing.llm) {
      modelSelect.value = settingsBase.routing.llm;
      ensureOption(modelSelect, settingsBase.routing.llm);
    }
  } catch (err) {
    return;
  }
}

function renderPreviewPanel(preview) {
  if (!settingsPreviewPanel) return;
  settingsPreviewPanel.textContent = '';
  if (!preview) return;
  const header = document.createElement('h3');
  header.textContent = `Preview ${preview.preview_id}`;
  settingsPreviewPanel.appendChild(header);
  const impacts = document.createElement('p');
  impacts.textContent = preview.impacts.length ? `Impacts: ${preview.impacts.join(', ')}` : 'Impacts: none';
  settingsPreviewPanel.appendChild(impacts);
  const warnings = document.createElement('p');
  warnings.textContent = preview.warnings.length ? `Warnings: ${preview.warnings.join(', ')}` : 'Warnings: none';
  settingsPreviewPanel.appendChild(warnings);
  const list = document.createElement('ul');
  preview.diff.forEach((entry) => {
    const li = document.createElement('li');
    const before = typeof entry.before === 'object' ? JSON.stringify(entry.before) : entry.before;
    const after = typeof entry.after === 'object' ? JSON.stringify(entry.after) : entry.after;
    li.textContent = `${entry.path}: ${before} -> ${after}`;
    list.appendChild(li);
  });
  settingsPreviewPanel.appendChild(list);
  if (settingsSchema) {
    const tier = settingsSchema.tiers.find((t) => t.tier_id === settingsTier);
    if (tier && tier.requires_confirm) {
      const confirm = document.createElement('div');
      confirm.className = 'preview-panel';
      const label = document.createElement('p');
      label.textContent = `Confirm phrase: ${tier.confirm_phrase}`;
      confirm.appendChild(label);
      settingsPreviewPanel.appendChild(confirm);
    }
  }
}

previewSettingsBtn.addEventListener('click', async () => {
  const candidate = buildCandidate(settingsBase, settingsDraft) || {};
  if (Object.keys(candidate).length === 0) {
    settingsStatus.textContent = 'No changes to preview.';
    return;
  }
  try {
    const response = await apiFetch('/api/settings/preview', {
      method: 'POST',
      body: JSON.stringify({ candidate, tier: settingsTier })
    });
    if (!response.ok) throw new Error('preview failed');
    settingsPreview = await response.json();
    settingsStatus.textContent = 'Preview ready. Apply to persist changes.';
    renderPreviewPanel(settingsPreview);
  } catch (err) {
    settingsStatus.textContent = 'Preview failed.';
  }
});

applySettingsBtn.addEventListener('click', async () => {
  if (!settingsPreview) {
    settingsStatus.textContent = 'Run preview first.';
    return;
  }
  let confirmPhrase = null;
  let confirm = true;
  if (settingsSchema) {
    const tier = settingsSchema.tiers.find((t) => t.tier_id === settingsTier);
    if (tier && tier.requires_confirm) {
      confirmPhrase = prompt(`Type confirm phrase: ${tier.confirm_phrase}`) || '';
      confirm = confirmPhrase === tier.confirm_phrase;
      if (!confirm) {
        settingsStatus.textContent = 'Confirm phrase mismatch.';
        return;
      }
    }
  }
  const candidate = buildCandidate(settingsBase, settingsDraft) || {};
  try {
    const response = await apiFetch('/api/settings/apply', {
      method: 'POST',
      body: JSON.stringify({
        candidate,
        preview_id: settingsPreview.preview_id,
        confirm,
        confirm_phrase: confirmPhrase,
        tier: settingsTier
      })
    });
    if (!response.ok) throw new Error('apply failed');
    const data = await response.json();
    settingsStatus.textContent = `Applied at ${data.applied_at_utc}`;
    settingsPreview = null;
    await loadSettingsEffective();
    renderPreviewPanel(null);
  } catch (err) {
    settingsStatus.textContent = 'Apply failed.';
  }
});

const runDoctor = document.getElementById('runDoctor');
const doctorResults = document.getElementById('doctorResults');

runDoctor.addEventListener('click', async () => {
  doctorResults.textContent = 'Running...';
  try {
    const response = await apiFetch('/api/doctor?verbose=true');
    if (!response.ok) throw new Error('doctor failed');
    const data = await response.json();
    doctorResults.textContent = '';
    data.results.forEach((check) => {
      const card = document.createElement('div');
      card.className = 'diagnostic-card';
      card.textContent = `${check.name}: ${check.ok ? 'OK' : 'FAIL'} - ${check.detail}`;
      doctorResults.appendChild(card);
    });
  } catch (err) {
    doctorResults.textContent = 'Doctor failed.';
  }
});

const deleteKind = document.getElementById('deleteKind');
const deleteStart = document.getElementById('deleteStart');
const deleteEnd = document.getElementById('deleteEnd');
const deleteProcess = document.getElementById('deleteProcess');
const deleteWindowTitle = document.getElementById('deleteWindowTitle');
const deleteConfirmPhrase = document.getElementById('deleteConfirmPhrase');
const deleteConfirmCheck = document.getElementById('deleteConfirmCheck');
const previewDelete = document.getElementById('previewDelete');
const applyDelete = document.getElementById('applyDelete');
const deletePreviewPanel = document.getElementById('deletePreview');
let deletePreviewState = null;

previewDelete.addEventListener('click', async () => {
  const kind = deleteKind.value;
  const payload = {
    start_utc: deleteStart.value || null,
    end_utc: deleteEnd.value || null,
    process: deleteProcess.value || null,
    window_title: deleteWindowTitle.value || null,
    sample_limit: 20
  };
  try {
    const response = await apiFetch(`/api/delete/${kind}/preview`, {
      method: 'POST',
      body: JSON.stringify(payload)
    });
    if (!response.ok) throw new Error('preview failed');
    const data = await response.json();
    deletePreviewState = data;
    deletePreviewPanel.textContent = `Preview ${data.preview_id} | counts ${JSON.stringify(data.counts)}`;
  } catch (err) {
    deletePreviewPanel.textContent = 'Delete preview failed.';
  }
});

applyDelete.addEventListener('click', async () => {
  if (!deletePreviewState) {
    deletePreviewPanel.textContent = 'Run preview first.';
    return;
  }
  if (!deleteConfirmCheck.checked) {
    deletePreviewPanel.textContent = 'Confirm checkbox required.';
    return;
  }
  const phrase = (deleteConfirmPhrase.value || '').trim();
  const kind = deleteKind.value;
  const payload = {
    start_utc: deleteStart.value || null,
    end_utc: deleteEnd.value || null,
    process: deleteProcess.value || null,
    window_title: deleteWindowTitle.value || null,
    sample_limit: 20,
    preview_id: deletePreviewState.preview_id,
    confirm: true,
    confirm_phrase: phrase || 'DELETE',
    expected_counts: deletePreviewState.counts
  };
  try {
    const response = await apiFetch(`/api/delete/${kind}/apply`, {
      method: 'POST',
      body: JSON.stringify(payload)
    });
    if (!response.ok) throw new Error('apply failed');
    const data = await response.json();
    deletePreviewPanel.textContent = `Deleted: ${JSON.stringify(data.counts)}`;
    deletePreviewState = null;
  } catch (err) {
    deletePreviewPanel.textContent = 'Delete apply failed.';
  }
});

const auditList = document.getElementById('auditList');
const auditDetail = document.getElementById('auditDetail');
const refreshAudit = document.getElementById('refreshAudit');

async function loadAudit() {
  if (!auditList) return;
  auditList.textContent = '';
  try {
    const response = await apiFetch('/api/audit/requests?limit=30');
    if (!response.ok) throw new Error('audit failed');
    const data = await response.json();
    data.requests.forEach((item) => {
      const card = document.createElement('div');
      card.className = 'audit-card';
      card.textContent = `${item.status} · ${item.query_text}`;
      card.addEventListener('click', async () => {
        Array.from(auditList.children).forEach((child) => child.classList.remove('active'));
        card.classList.add('active');
        if (!item.answer_id) {
          auditDetail.textContent = 'No answer record for this request.';
          return;
        }
        const detailResp = await apiFetch(`/api/audit/answers/${item.answer_id}`);
        if (!detailResp.ok) {
          auditDetail.textContent = 'Failed to load answer detail.';
          return;
        }
        const detail = await detailResp.json();
        auditDetail.textContent = `Answer ${detail.answer.answer_id} · mode ${detail.answer.mode} · claims ${detail.answer.claims.length}`;
      });
      auditList.appendChild(card);
    });
  } catch (err) {
    auditList.textContent = 'Failed to load audit data.';
  }
}

refreshAudit.addEventListener('click', async () => {
  await loadAudit();
});

async function loadPlugins() {
  if (!pluginList) return;
  pluginList.textContent = '';
  if (pluginWarnings) pluginWarnings.textContent = '';
  let payload;
  try {
    const response = await apiFetch('/api/plugins/catalog');
    if (!response.ok) return;
    payload = await response.json();
  } catch (err) {
    return;
  }
  if (pluginSafeMode) {
    if (payload.safe_mode) {
      pluginSafeMode.textContent = 'On';
      pluginSafeMode.classList.add('safe');
    } else {
      pluginSafeMode.textContent = 'Off';
      pluginSafeMode.classList.remove('safe');
    }
  }
  if (pluginWarnings && Array.isArray(payload.warnings) && payload.warnings.length > 0) {
    pluginWarnings.textContent = payload.warnings.join(' | ');
  }
  const plugins = Array.isArray(payload.plugins) ? payload.plugins : [];
  plugins.forEach((plugin) => {
    const card = document.createElement('div');
    card.className = 'plugin-card';

    const header = document.createElement('div');
    header.className = 'plugin-header';
    const titleWrap = document.createElement('div');
    const title = document.createElement('h3');
    title.className = 'plugin-title';
    title.textContent = plugin.name || plugin.plugin_id;
    const pluginId = document.createElement('div');
    pluginId.className = 'plugin-id';
    pluginId.textContent = plugin.plugin_id;
    titleWrap.appendChild(title);
    titleWrap.appendChild(pluginId);

    const status = document.createElement('div');
    const blocked = Boolean(plugin.blocked);
    const enabled = Boolean(plugin.enabled) && !blocked;
    let statusText = enabled ? 'Enabled' : 'Disabled';
    let statusClass = enabled ? 'enabled' : 'disabled';
    if (blocked) {
      statusText = 'Blocked';
      statusClass = 'blocked';
    }
    status.className = `plugin-status ${statusClass}`;
    status.textContent = statusText;
    header.appendChild(titleWrap);
    header.appendChild(status);
    card.appendChild(header);

    const meta = document.createElement('div');
    meta.className = 'plugin-meta';
    meta.textContent = `Version ${plugin.version || '--'} · Source ${plugin.source || '--'} · Lock ${plugin.lock_status || '--'}`;
    card.appendChild(meta);

    if (plugin.reason) {
      const reason = document.createElement('div');
      reason.className = 'plugin-meta';
      reason.textContent = `Reason: ${plugin.reason}`;
      card.appendChild(reason);
    }

    if (plugin.manifest_sha256) {
      const hashes = document.createElement('div');
      hashes.className = 'plugin-meta';
      hashes.textContent = `Manifest ${plugin.manifest_sha256} · Code ${plugin.code_sha256 || '--'}`;
      card.appendChild(hashes);
    }

    if (Array.isArray(plugin.warnings) && plugin.warnings.length > 0) {
      const warn = document.createElement('div');
      warn.className = 'plugin-meta';
      warn.textContent = `Warnings: ${plugin.warnings.join(' | ')}`;
      card.appendChild(warn);
    }

    const extensions = document.createElement('div');
    extensions.className = 'plugin-extensions';
    (plugin.extensions || []).forEach((ext) => {
      const chip = document.createElement('span');
      chip.className = 'plugin-extension';
      chip.textContent = `${ext.kind}:${ext.id}`;
      extensions.appendChild(chip);
    });
    card.appendChild(extensions);

    const actions = document.createElement('div');
    actions.className = 'plugin-actions';
    const needsApproval = plugin.reason === 'lock_missing' || plugin.reason === 'lock_mismatch';
    if (blocked && needsApproval) {
      const approveBtn = document.createElement('button');
      approveBtn.type = 'button';
      approveBtn.textContent = 'Re-approve';
      approveBtn.className = 'secondary';
      approveBtn.addEventListener('click', async () => {
        if (!confirm(`Re-approve plugin ${plugin.plugin_id}?`)) return;
        await apiFetch('/api/plugins/lock', {
          method: 'POST',
          body: JSON.stringify({ plugin_id: plugin.plugin_id })
        });
        await loadPlugins();
        await loadRoutingOptions();
      });
      actions.appendChild(approveBtn);
    } else if (!enabled) {
      const enableBtn = document.createElement('button');
      enableBtn.type = 'button';
      enableBtn.textContent = 'Enable';
      enableBtn.addEventListener('click', async () => {
        const ok = confirm(`Enable plugin ${plugin.plugin_id}?`);
        if (!ok) return;
        const response = await apiFetch('/api/plugins/enable', {
          method: 'POST',
          body: JSON.stringify({ plugin_id: plugin.plugin_id, accept_hashes: true })
        });
        if (!response.ok) {
          const detail = await response.json().catch(() => ({}));
          let message = detail.detail || 'Enable failed';
          if (message && typeof message === 'object') {
            const inner = message.detail || 'Enable failed';
            const manifest = message.manifest_sha256
              ? `\nManifest: ${message.manifest_sha256}`
              : '';
            const code = message.code_sha256 ? `\nCode: ${message.code_sha256}` : '';
            message = `${inner}${manifest}${code}`;
          }
          alert(message);
        }
        await loadPlugins();
        await loadRoutingOptions();
      });
      actions.appendChild(enableBtn);
    } else {
      const disableBtn = document.createElement('button');
      disableBtn.type = 'button';
      disableBtn.textContent = 'Disable';
      disableBtn.className = 'secondary';
      disableBtn.addEventListener('click', async () => {
        if (!confirm(`Disable plugin ${plugin.plugin_id}?`)) return;
        await apiFetch('/api/plugins/disable', {
          method: 'POST',
          body: JSON.stringify({ plugin_id: plugin.plugin_id })
        });
        await loadPlugins();
        await loadRoutingOptions();
      });
      actions.appendChild(disableBtn);
    }
    card.appendChild(actions);

    pluginList.appendChild(card);
  });
}

tabs.forEach((btn) => {
  btn.addEventListener('click', () => {
    tabs.forEach((b) => b.classList.remove('active'));
    sections.forEach((s) => s.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(btn.dataset.tab).classList.add('active');
  });
});

if (applyPerfProfile) {
  applyPerfProfile.addEventListener('click', async () => {
    const profile = perfProfileSelect ? perfProfileSelect.value : 'auto';
    try {
      const response = await apiFetch('/api/runtime/profile', {
        method: 'POST',
        body: JSON.stringify({ profile })
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        alert(detail.detail || 'Profile update failed');
      }
    } catch (err) {
      alert('Profile update failed');
    }
  });
}

if (refreshPerfLog) {
  refreshPerfLog.addEventListener('click', loadPerfLog);
}

if (perfLogComponent) {
  perfLogComponent.addEventListener('change', loadPerfLog);
}

async function init() {
  await loadRoutingOptions();
  await loadSettingsSchema();
  await loadSettingsEffective();
  await loadHighlightsList();
  await loadPlugins();
  await loadAudit();
  await loadPerfLog();
}

const stateStore = createStateStore((state) => {
  renderStatusBar(state);
  renderPerfPanel(state);
}, () => {});
stateStore.start();
init();

if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/static/sw.js');
}
