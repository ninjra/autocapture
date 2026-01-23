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
const refreshDashboard = document.getElementById('refreshDashboard');
const dashboardHealth = document.getElementById('dashboardHealth');
const dashboardStorage = document.getElementById('dashboardStorage');
const dashboardQueues = document.getElementById('dashboardQueues');
const dashboardComponents = document.getElementById('dashboardComponents');
const dashboardRuntime = document.getElementById('dashboardRuntime');
const dashboardApi = document.getElementById('dashboardApi');
const searchModeInputs = document.querySelectorAll('input[name="searchMode"]');
const explorerTimeRange = document.getElementById('explorerTimeRange');
const explorerSearch = document.getElementById('explorerSearch');
const explorerProcess = document.getElementById('explorerProcess');
const explorerWindowTitle = document.getElementById('explorerWindowTitle');
const explorerHasScreenshot = document.getElementById('explorerHasScreenshot');
const explorerHasFocus = document.getElementById('explorerHasFocus');
const facetApps = document.getElementById('facetApps');
const facetDomains = document.getElementById('facetDomains');
const applyExplorerFilters = document.getElementById('applyExplorerFilters');
const clearExplorerFilters = document.getElementById('clearExplorerFilters');
const explorerGridViewport = document.getElementById('explorerGridViewport');
const explorerGrid = document.getElementById('explorerGrid');
const loadMoreExplorer = document.getElementById('loadMoreExplorer');
const explorerCount = document.getElementById('explorerCount');
const explorerStatus = document.getElementById('explorerStatus');
const eventDrawer = document.getElementById('eventDrawer');
const closeDrawer = document.getElementById('closeDrawer');
const drawerImage = document.getElementById('drawerImage');
const drawerOverlay = document.getElementById('drawerOverlay');
const drawerTitle = document.getElementById('drawerTitle');
const drawerSubtitle = document.getElementById('drawerSubtitle');
const drawerMeta = document.getElementById('drawerMeta');
const drawerOcrText = document.getElementById('drawerOcrText');
const toggleOcrOverlay = document.getElementById('toggleOcrOverlay');
const openFullImage = document.getElementById('openFullImage');
const drawerPrev = document.getElementById('drawerPrev');
const drawerNext = document.getElementById('drawerNext');
const drawerDelete = document.getElementById('drawerDelete');
const pluginDetail = document.getElementById('pluginDetail');
let latestState = null;
let latestStorage = null;
let searchMode = 'text';
let explorerItems = [];
let explorerNextCursor = null;
let explorerLoading = false;
let explorerExhausted = false;
let explorerSelectedIndex = null;
let explorerSelectedApps = new Set();
let explorerSelectedDomains = new Set();
let explorerRenderScheduled = false;
let explorerLayout = {
  columns: 1,
  tileWidth: 220,
  tileHeight: 180,
  gap: 12,
  rowHeight: 192
};
let activePluginId = null;
let pluginPreviewState = null;
let currentDrawerDetail = null;

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

function formatTimestamp(value) {
  if (!value) return '--';
  try {
    return new Date(value).toLocaleString();
  } catch (err) {
    return String(value);
  }
}

function clearElement(el) {
  if (!el) return;
  while (el.firstChild) {
    el.removeChild(el.firstChild);
  }
}

function buildQuery(params) {
  const query = new URLSearchParams();
  Object.entries(params || {}).forEach(([key, value]) => {
    if (value === null || value === undefined || value === '') return;
    if (Array.isArray(value)) {
      value.forEach((item) => {
        if (item !== null && item !== undefined && item !== '') {
          query.append(key, item);
        }
      });
      return;
    }
    query.set(key, String(value));
  });
  return query.toString();
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
  renderStatusBar(latestState, latestStorage);
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

function createStorageStore(onUpdate, onError) {
  let delay = 10000;
  let timer = null;
  let stopped = false;

  async function poll() {
    if (stopped) return;
    if (document.hidden) {
      schedule(delay);
      return;
    }
    try {
      const response = await apiFetch('/api/storage/stats', { headers: {} });
      if (!response.ok) throw new Error('storage stats failed');
      const data = await response.json();
      delay = 10000;
      if (onUpdate) onUpdate(data);
    } catch (err) {
      delay = Math.min(delay * 2, 60000);
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
      delay = 10000;
      poll();
    }
  });

  return { start, stop };
}

function renderStatusBar(state, storageStats) {
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

  if (storageStats) {
    const storage = document.createElement('div');
    storage.className = 'status-chip';
    const total = formatBytes(storageStats.total_bytes);
    const free = formatBytes(storageStats.free_bytes);
    storage.textContent = `Storage: ${total} (free ${free})`;
    container.appendChild(storage);
  } else if (state.storage) {
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

function renderDashboardCard(target, title, rows) {
  if (!target) return;
  clearElement(target);
  const heading = document.createElement('h3');
  heading.textContent = title;
  target.appendChild(heading);
  if (!rows || rows.length === 0) {
    const empty = document.createElement('p');
    empty.textContent = 'No data available.';
    target.appendChild(empty);
    return;
  }
  rows.forEach(([label, value]) => {
    const line = document.createElement('p');
    line.textContent = `${label}: ${value}`;
    target.appendChild(line);
  });
}

function renderDashboard(state, storageStats) {
  if (!dashboardHealth) return;
  if (!state) {
    renderDashboardCard(dashboardHealth, 'Health', [['Status', 'Unavailable']]);
    renderDashboardCard(dashboardStorage, 'Storage', [['Status', 'Unavailable']]);
    renderDashboardCard(dashboardQueues, 'Queues', [['Status', 'Unavailable']]);
    renderDashboardCard(dashboardComponents, 'Components', [['Status', 'Unavailable']]);
    renderDashboardCard(dashboardRuntime, 'Runtime', [['Status', 'Unavailable']]);
    renderDashboardCard(dashboardApi, 'API', [['Status', 'Unavailable']]);
    return;
  }
  renderDashboardCard(dashboardHealth, 'Health', [
    ['Overall', state.health?.overall || 'unknown'],
    ['Issues', Array.isArray(state.health?.issues) ? state.health.issues.length : 0],
    ['Mode', state.app?.mode || '--'],
    ['Unlocked', state.lock?.unlocked ? 'yes' : 'no']
  ]);
  if (storageStats) {
    renderDashboardCard(dashboardStorage, 'Storage', [
      ['Media', formatBytes(storageStats.media_bytes)],
      ['Staging', formatBytes(storageStats.staging_bytes)],
      ['DB', formatBytes(storageStats.db_bytes)],
      ['Total', formatBytes(storageStats.total_bytes)],
      ['Free', formatBytes(storageStats.free_bytes)]
    ]);
  } else if (state.storage) {
    renderDashboardCard(dashboardStorage, 'Storage', [
      ['Media', formatBytes(state.storage.media_usage_bytes)],
      ['Free', formatBytes(state.storage.free_bytes)],
      ['TTL', `${state.storage.screenshot_ttl_days} days`]
    ]);
  } else {
    renderDashboardCard(dashboardStorage, 'Storage', [['Status', 'Unavailable']]);
  }
  renderDashboardCard(dashboardQueues, 'Queues', [
    ['OCR pending', state.queues?.ocr_pending ?? 0],
    ['OCR processing', state.queues?.ocr_processing ?? 0],
    ['Span embeddings', state.queues?.span_embed_pending ?? 0],
    ['Event embeddings', state.queues?.event_embed_pending ?? 0]
  ]);
  const stale = Array.isArray(state.components) ? state.components.filter((c) => c.stale).length : 0;
  renderDashboardCard(dashboardComponents, 'Components', [
    ['Total', Array.isArray(state.components) ? state.components.length : 0],
    ['Stale', stale],
    ['Plugins enabled', state.plugins?.enabled_count ?? 0],
    ['Plugins blocked', state.plugins?.blocked_count ?? 0]
  ]);
  const runtime = Array.isArray(state.components)
    ? state.components.find((c) => c.component === 'runtime')
    : null;
  const api = Array.isArray(state.components)
    ? state.components.find((c) => c.component === 'api')
    : null;
  renderDashboardCard(dashboardRuntime, 'Runtime', [
    ['Status', runtime?.status || 'unknown'],
    ['Stale', runtime?.stale ? 'yes' : 'no'],
    ['Interval', runtime?.interval_s ? `${runtime.interval_s}s` : '--']
  ]);
  renderDashboardCard(dashboardApi, 'API', [
    ['Status', api?.status || 'unknown'],
    ['Stale', api?.stale ? 'yes' : 'no'],
    ['Interval', api?.interval_s ? `${api.interval_s}s` : '--']
  ]);
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
      `/api/perf/log?component=${encodeURIComponent(component)}&tail=200`
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
  if (contextPack && Array.isArray(contextPack.evidence) && contextPack.evidence.length > 0) {
    const evidenceWrap = document.createElement('div');
    evidenceWrap.className = 'evidence-list';
    const title = document.createElement('small');
    title.className = 'citations';
    title.textContent = `Evidence items: ${contextPack.evidence.length}`;
    evidenceWrap.appendChild(title);
    contextPack.evidence.slice(0, 10).forEach((item) => {
      const row = document.createElement('div');
      row.className = 'evidence-item';
      const label = document.createElement('span');
      label.textContent = item.title || item.source || item.event_id || 'Evidence';
      row.appendChild(label);
      if (item.event_id) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.textContent = 'Open';
        btn.addEventListener('click', () => openEventDrawer(item.event_id, item));
        row.appendChild(btn);
      }
      evidenceWrap.appendChild(row);
    });
    if (contextPack.evidence.length > 10) {
      const more = document.createElement('small');
      more.className = 'hint';
      more.textContent = `Showing 10 of ${contextPack.evidence.length}`;
      evidenceWrap.appendChild(more);
    }
    div.appendChild(evidenceWrap);
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
  } else if (rangeKey === '90d') {
    start = new Date(now.getTime() - 90 * 24 * 60 * 60 * 1000);
  } else if (rangeKey === 'all') {
    return [null, null];
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

if (searchModeInputs) {
  searchModeInputs.forEach((input) => {
    if (input.checked) searchMode = input.value;
    input.addEventListener('change', () => {
      if (input.checked) searchMode = input.value;
    });
  });
}

function renderSearchMessage(message) {
  if (!searchResults) return;
  const card = document.createElement('div');
  card.className = 'result-card';
  card.textContent = message;
  searchResults.appendChild(card);
}

function renderSearchEvent(item) {
  const card = document.createElement('div');
  card.className = 'result-card';
  if (item.screenshot_url || item.focus_url) {
    const img = document.createElement('img');
    const thumbUrl = item.screenshot_url || item.focus_url;
    img.src = withUnlock(thumbUrl);
    img.alt = 'thumbnail';
    card.appendChild(img);
  }
  const strong = document.createElement('strong');
  strong.textContent = item.app_name || 'Unknown app';
  card.appendChild(strong);
  const title = document.createElement('p');
  title.textContent = item.window_title || '';
  card.appendChild(title);
  const snippet = document.createElement('p');
  snippet.textContent = item.ocr_snippet || '';
  card.appendChild(snippet);
  const small = document.createElement('small');
  small.textContent = item.event_id || '';
  card.appendChild(small);
  card.addEventListener('click', () => openEventDrawer(item.event_id, item));
  return card;
}

async function runTextSearch(query) {
  try {
    const response = await apiFetch(`/api/events?${buildQuery({ q: query, limit: 30 })}`);
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      renderSearchMessage(detail.detail || 'Search failed.');
      return;
    }
    const data = await response.json();
    const items = Array.isArray(data.items) ? data.items : [];
    if (!items.length) {
      renderSearchMessage('No matches found.');
      return;
    }
    items.forEach((item) => {
      searchResults.appendChild(renderSearchEvent(item));
    });
  } catch (err) {
    renderSearchMessage('failed to reach the local API server.');
  }
}

async function runSemanticSearch(query) {
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
    renderSearchMessage('failed to reach the local API server.');
    return;
  }
  if (!response.ok) {
    const detail = typeof data.detail === 'string' ? data.detail : 'request failed';
    renderSearchMessage(`Error: ${detail}.`);
    return;
  }
  if (data && data.no_evidence) {
    renderSearchMessage(data.message || 'No evidence found.');
    return;
  }
  const evidence = Array.isArray(data.evidence) ? data.evidence : [];
  if (!evidence.length) {
    renderSearchMessage('No evidence found.');
    return;
  }
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
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.textContent = 'Open event';
      btn.addEventListener('click', (ev) => {
        ev.stopPropagation();
        openEventDrawer(item.event_id, item);
      });
      card.appendChild(btn);
    }
    searchResults.appendChild(card);
  });
}

searchForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const query = searchInput.value.trim();
  if (!query) return;
  searchResults.textContent = '';
  if (searchMode === 'semantic') {
    await runSemanticSearch(query);
  } else {
    await runTextSearch(query);
  }
});

function buildExplorerFilters() {
  const [startUtc, endUtc] = rangeToTuple(explorerTimeRange ? explorerTimeRange.value : '30d');
  const q = explorerSearch ? explorerSearch.value.trim() : '';
  const process = explorerProcess ? explorerProcess.value.trim() : '';
  const windowTitle = explorerWindowTitle ? explorerWindowTitle.value.trim() : '';
  const hasScreenshot = explorerHasScreenshot && explorerHasScreenshot.checked ? true : null;
  const hasFocus = explorerHasFocus && explorerHasFocus.checked ? true : null;
  return {
    q: q || null,
    start_utc: startUtc,
    end_utc: endUtc,
    apps: Array.from(explorerSelectedApps),
    domains: Array.from(explorerSelectedDomains),
    process: process || null,
    window_title: windowTitle || null,
    has_screenshot: hasScreenshot,
    has_focus: hasFocus
  };
}

function resetExplorerState() {
  explorerItems = [];
  explorerNextCursor = null;
  explorerExhausted = false;
  explorerSelectedIndex = null;
  if (explorerGrid) {
    explorerGrid.style.height = '0px';
  }
  if (explorerGridViewport) {
    explorerGridViewport.scrollTop = 0;
  }
  updateExplorerCount();
  scheduleExplorerRender();
}

function updateExplorerStatus(message) {
  if (!explorerStatus) return;
  explorerStatus.textContent = message || '';
}

function updateExplorerCount() {
  if (!explorerCount) return;
  explorerCount.textContent = `${explorerItems.length} items`;
}

function renderFacetList(container, facets, selectedSet) {
  if (!container) return;
  clearElement(container);
  if (!Array.isArray(facets) || facets.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'hint';
    empty.textContent = 'No data';
    container.appendChild(empty);
    return;
  }
  facets.forEach((facet) => {
    const wrapper = document.createElement('label');
    wrapper.className = 'facet-item';
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.checked = selectedSet.has(facet.value);
    checkbox.addEventListener('change', () => {
      if (checkbox.checked) {
        selectedSet.add(facet.value);
      } else {
        selectedSet.delete(facet.value);
      }
    });
    const text = document.createElement('span');
    text.textContent = `${facet.value} (${facet.count})`;
    wrapper.appendChild(checkbox);
    wrapper.appendChild(text);
    container.appendChild(wrapper);
  });
}

async function loadExplorerFacets() {
  const filters = buildExplorerFilters();
  try {
    const response = await apiFetch(`/api/events/facets?${buildQuery(filters)}`);
    if (!response.ok) return;
    const data = await response.json();
    renderFacetList(facetApps, data.apps || [], explorerSelectedApps);
    renderFacetList(facetDomains, data.domains || [], explorerSelectedDomains);
  } catch (err) {
    return;
  }
}

async function loadExplorerPage(reset) {
  if (explorerLoading) return;
  if (!reset && explorerExhausted) return;
  if (reset) {
    resetExplorerState();
  }
  explorerLoading = true;
  updateExplorerStatus('Loading...');
  const filters = buildExplorerFilters();
  const params = {
    ...filters,
    limit: 100,
    cursor: reset ? null : explorerNextCursor
  };
  try {
    const response = await apiFetch(`/api/events?${buildQuery(params)}`);
    if (!response.ok) {
      updateExplorerStatus('Failed to load events.');
      return;
    }
    const data = await response.json();
    const items = Array.isArray(data.items) ? data.items : [];
    explorerItems = reset ? items : explorerItems.concat(items);
    explorerNextCursor = data.next_cursor || null;
    explorerExhausted = !explorerNextCursor;
    if (Number.isFinite(explorerSelectedIndex) && explorerSelectedIndex >= explorerItems.length) {
      explorerSelectedIndex = null;
      updateDrawerNav();
    }
    updateExplorerCount();
    updateExplorerLayout();
    if (reset && explorerItems.length === 0) {
      updateExplorerStatus('No results.');
    } else {
      updateExplorerStatus(explorerExhausted ? 'End of results.' : '');
    }
  } catch (err) {
    updateExplorerStatus('Failed to load events.');
  } finally {
    explorerLoading = false;
  }
}

function updateExplorerLayout() {
  if (!explorerGridViewport || !explorerGrid) return;
  const width = Math.max(0, explorerGridViewport.clientWidth - 24);
  const gap = explorerLayout.gap;
  const columns = Math.max(1, Math.floor(width / (explorerLayout.tileWidth + gap)));
  const totalGap = gap * (columns - 1);
  const tileWidth = Math.max(180, Math.floor((width - totalGap) / columns));
  explorerLayout = {
    ...explorerLayout,
    columns,
    tileWidth,
    rowHeight: explorerLayout.tileHeight + gap
  };
  const rows = Math.ceil(explorerItems.length / columns);
  explorerGrid.style.height = `${rows * explorerLayout.rowHeight}px`;
  scheduleExplorerRender();
}

function scheduleExplorerRender() {
  if (explorerRenderScheduled) return;
  explorerRenderScheduled = true;
  requestAnimationFrame(() => {
    explorerRenderScheduled = false;
    renderExplorerGrid();
  });
}

function renderExplorerGrid() {
  if (!explorerGridViewport || !explorerGrid) return;
  const { columns, tileWidth, tileHeight, gap, rowHeight } = explorerLayout;
  const scrollTop = explorerGridViewport.scrollTop;
  const viewportHeight = explorerGridViewport.clientHeight;
  const startRow = Math.max(0, Math.floor(scrollTop / rowHeight) - 1);
  const endRow = Math.min(
    Math.ceil((scrollTop + viewportHeight) / rowHeight) + 1,
    Math.ceil(explorerItems.length / columns)
  );
  const startIndex = startRow * columns;
  const endIndex = Math.min(endRow * columns, explorerItems.length);
  clearElement(explorerGrid);
  for (let i = startIndex; i < endIndex; i += 1) {
    const item = explorerItems[i];
    const row = Math.floor(i / columns);
    const col = i % columns;
    const tile = document.createElement('div');
    tile.className = 'explorer-tile';
    tile.style.width = `${tileWidth}px`;
    tile.style.height = `${tileHeight}px`;
    tile.style.transform = `translate(${col * (tileWidth + gap)}px, ${row * rowHeight}px)`;
    const thumb = document.createElement('img');
    thumb.className = 'explorer-thumb';
    const thumbUrl = item.screenshot_url || item.focus_url;
    if (thumbUrl) {
      thumb.src = withUnlock(thumbUrl);
    }
    thumb.alt = item.window_title || 'event';
    tile.appendChild(thumb);
    const title = document.createElement('div');
    title.className = 'explorer-title';
    title.textContent = item.app_name || 'Unknown app';
    tile.appendChild(title);
    const subtitle = document.createElement('div');
    subtitle.className = 'explorer-snippet';
    subtitle.textContent = item.window_title || '';
    tile.appendChild(subtitle);
    const snippet = document.createElement('div');
    snippet.className = 'explorer-snippet';
    snippet.textContent = item.ocr_snippet || '';
    tile.appendChild(snippet);
    tile.addEventListener('click', () => openEventDrawer(item.event_id, item, i));
    explorerGrid.appendChild(tile);
  }
}

function updateDrawerNav() {
  if (!drawerPrev || !drawerNext) return;
  const hasIndex = Number.isFinite(explorerSelectedIndex);
  drawerPrev.disabled = !hasIndex || explorerSelectedIndex <= 0;
  drawerNext.disabled = !hasIndex || explorerSelectedIndex >= explorerItems.length - 1;
}

function normalizeBBox(bbox) {
  if (!bbox) return null;
  let values = [];
  if (Array.isArray(bbox)) {
    values = bbox.filter((val) => typeof val === 'number');
  } else if (typeof bbox === 'object') {
    const x0 = bbox.x0 ?? bbox.x ?? 0;
    const y0 = bbox.y0 ?? bbox.y ?? 0;
    const x1 = bbox.x1 ?? (bbox.x ?? 0) + (bbox.w ?? 0);
    const y1 = bbox.y1 ?? (bbox.y ?? 0) + (bbox.h ?? 0);
    values = [x0, y0, x1, y1].map((val) => Number(val));
  }
  if (values.length < 4) return null;
  if (values.length >= 8) {
    const xs = values.filter((_, idx) => idx % 2 === 0);
    const ys = values.filter((_, idx) => idx % 2 === 1);
    return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
  }
  return values.slice(0, 4);
}

function renderDrawerOverlay(spans) {
  if (!drawerOverlay || !drawerImage) return;
  clearElement(drawerOverlay);
  if (!Array.isArray(spans) || spans.length === 0) return;
  const naturalWidth = drawerImage.naturalWidth || drawerImage.width;
  const naturalHeight = drawerImage.naturalHeight || drawerImage.height;
  const rect = drawerImage.getBoundingClientRect();
  const scaleX = rect.width / naturalWidth;
  const scaleY = rect.height / naturalHeight;
  spans.forEach((span) => {
    const bbox = normalizeBBox(span.bbox);
    if (!bbox) return;
    let [x0, y0, x1, y1] = bbox;
    const maxVal = Math.max(...bbox);
    const normalized = maxVal <= 1.0;
    if (normalized) {
      x0 *= naturalWidth;
      x1 *= naturalWidth;
      y0 *= naturalHeight;
      y1 *= naturalHeight;
    }
    const box = document.createElement('div');
    box.className = 'ocr-box';
    box.style.left = `${x0 * scaleX}px`;
    box.style.top = `${y0 * scaleY}px`;
    box.style.width = `${(x1 - x0) * scaleX}px`;
    box.style.height = `${(y1 - y0) * scaleY}px`;
    drawerOverlay.appendChild(box);
  });
}

function openEventDrawer(eventId, item, index) {
  if (!eventDrawer) return;
  explorerSelectedIndex = Number.isFinite(index) ? index : null;
  updateDrawerNav();
  eventDrawer.classList.remove('hidden');
  document.body.style.overflow = 'hidden';
  currentDrawerDetail = null;
  if (drawerTitle) {
    drawerTitle.textContent = item?.app_name || 'Event detail';
  }
  if (drawerSubtitle) {
    drawerSubtitle.textContent = item?.window_title || '';
  }
  if (drawerMeta) {
    drawerMeta.textContent = 'Loading...';
  }
  if (drawerOcrText) {
    drawerOcrText.textContent = '';
  }
  if (drawerOverlay) {
    drawerOverlay.textContent = '';
  }
  if (toggleOcrOverlay) {
    toggleOcrOverlay.checked = false;
  }
  if (drawerImage) {
    drawerImage.src = '';
  }
  apiFetch(`/api/events/${eventId}`)
    .then((response) => {
      if (!response.ok) throw new Error('event detail failed');
      return response.json();
    })
    .then((detail) => {
      currentDrawerDetail = detail;
      const screenshotUrl = detail.screenshot_path
        ? `/api/screenshot/${detail.event_id}?variant=full`
        : detail.focus_path
          ? `/api/focus/${detail.event_id}?variant=full`
          : null;
      if (drawerImage) {
        drawerImage.onload = () => {
          if (toggleOcrOverlay && toggleOcrOverlay.checked) {
            renderDrawerOverlay(detail.ocr_spans || []);
          }
        };
        if (screenshotUrl) {
          drawerImage.src = withUnlock(screenshotUrl);
        }
      }
      if (openFullImage && screenshotUrl) {
        openFullImage.href = withUnlock(screenshotUrl);
      }
      if (drawerMeta) {
        drawerMeta.textContent = `${formatTimestamp(detail.ts_start)} · ${detail.app_name || ''} · ${detail.window_title || ''}`;
      }
      if (drawerOcrText) {
        drawerOcrText.textContent = detail.ocr_text || '';
      }
      if (!screenshotUrl && drawerMeta) {
        drawerMeta.textContent += ' · No image available';
      }
    })
    .catch(() => {
      if (drawerMeta) {
        drawerMeta.textContent = 'Failed to load event detail.';
      }
    });
}

function closeEventDrawer() {
  if (!eventDrawer) return;
  eventDrawer.classList.add('hidden');
  document.body.style.overflow = '';
}

function activateTab(tabId) {
  if (!tabId) return;
  tabs.forEach((b) => b.classList.remove('active'));
  sections.forEach((s) => s.classList.remove('active'));
  const targetButton = Array.from(tabs).find((b) => b.dataset.tab === tabId);
  const targetSection = document.getElementById(tabId);
  if (targetButton) targetButton.classList.add('active');
  if (targetSection) targetSection.classList.add('active');
}

if (closeDrawer) {
  closeDrawer.addEventListener('click', closeEventDrawer);
}

if (eventDrawer) {
  eventDrawer.addEventListener('click', (event) => {
    if (event.target === eventDrawer) {
      closeEventDrawer();
    }
  });
}

if (toggleOcrOverlay) {
  toggleOcrOverlay.addEventListener('change', () => {
    if (!drawerOverlay) return;
    if (toggleOcrOverlay.checked && currentDrawerDetail) {
      renderDrawerOverlay(currentDrawerDetail.ocr_spans || []);
    } else {
      clearElement(drawerOverlay);
    }
  });
}

if (drawerPrev) {
  drawerPrev.addEventListener('click', () => {
    if (!Number.isFinite(explorerSelectedIndex)) return;
    const idx = explorerSelectedIndex - 1;
    if (idx < 0 || idx >= explorerItems.length) return;
    openEventDrawer(explorerItems[idx].event_id, explorerItems[idx], idx);
  });
}

if (drawerNext) {
  drawerNext.addEventListener('click', () => {
    if (!Number.isFinite(explorerSelectedIndex)) return;
    const idx = explorerSelectedIndex + 1;
    if (idx < 0 || idx >= explorerItems.length) return;
    openEventDrawer(explorerItems[idx].event_id, explorerItems[idx], idx);
  });
}

if (drawerDelete) {
  drawerDelete.addEventListener('click', () => {
    if (!currentDrawerDetail) return;
    const startField = document.getElementById('deleteStart');
    const endField = document.getElementById('deleteEnd');
    const processField = document.getElementById('deleteProcess');
    const windowField = document.getElementById('deleteWindowTitle');
    const kindField = document.getElementById('deleteKind');
    if (kindField) kindField.value = 'range';
    if (startField) startField.value = currentDrawerDetail.ts_start || '';
    if (endField) endField.value = currentDrawerDetail.ts_end || currentDrawerDetail.ts_start || '';
    if (processField) processField.value = currentDrawerDetail.app_name || '';
    if (windowField) windowField.value = currentDrawerDetail.window_title || '';
    activateTab('maintenance');
    closeEventDrawer();
  });
}

if (applyExplorerFilters) {
  applyExplorerFilters.addEventListener('click', async () => {
    await loadExplorerFacets();
    await loadExplorerPage(true);
  });
}

if (clearExplorerFilters) {
  clearExplorerFilters.addEventListener('click', async () => {
    explorerSelectedApps = new Set();
    explorerSelectedDomains = new Set();
    if (explorerSearch) explorerSearch.value = '';
    if (explorerProcess) explorerProcess.value = '';
    if (explorerWindowTitle) explorerWindowTitle.value = '';
    if (explorerHasScreenshot) explorerHasScreenshot.checked = true;
    if (explorerHasFocus) explorerHasFocus.checked = false;
    if (explorerTimeRange) explorerTimeRange.value = '30d';
    await loadExplorerFacets();
    await loadExplorerPage(true);
  });
}

if (loadMoreExplorer) {
  loadMoreExplorer.addEventListener('click', async () => {
    await loadExplorerPage(false);
  });
}

if (explorerGridViewport) {
  explorerGridViewport.addEventListener('scroll', () => {
    scheduleExplorerRender();
    if (!explorerGrid || explorerLoading || explorerExhausted) return;
    const threshold = explorerGrid.scrollHeight - explorerLayout.rowHeight * 2;
    if (explorerGridViewport.scrollTop + explorerGridViewport.clientHeight >= threshold) {
      loadExplorerPage(false);
    }
  });
}

window.addEventListener('resize', () => {
  updateExplorerLayout();
  if (toggleOcrOverlay && toggleOcrOverlay.checked && currentDrawerDetail) {
    renderDrawerOverlay(currentDrawerDetail.ocr_spans || []);
  }
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
      if (field.danger_level === 'danger') {
        wrapper.classList.add('danger');
      } else if (field.danger_level === 'warn') {
        wrapper.classList.add('warning');
      }
      const label = document.createElement('label');
      label.textContent = field.label;
      wrapper.appendChild(label);
      if (field.description) {
        const desc = document.createElement('span');
        desc.className = 'hint';
        desc.textContent = field.description;
        wrapper.appendChild(desc);
      }
      const badges = document.createElement('div');
      badges.className = 'field-badges';
      if (field.sensitive) {
        const badge = document.createElement('span');
        badge.className = 'badge danger';
        badge.textContent = 'Sensitive';
        badges.appendChild(badge);
      }
      if (field.requires_restart) {
        const badge = document.createElement('span');
        badge.className = 'badge warning';
        badge.textContent = 'Requires restart';
        badges.appendChild(badge);
      }
      if (field.danger_level === 'danger') {
        const badge = document.createElement('span');
        badge.className = 'badge danger';
        badge.textContent = 'Danger';
        badges.appendChild(badge);
      } else if (field.danger_level === 'warn') {
        const badge = document.createElement('span');
        badge.className = 'badge warning';
        badge.textContent = 'Warning';
        badges.appendChild(badge);
      }
      if (badges.childElementCount) {
        wrapper.appendChild(badges);
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
        clearElement(auditDetail);
        const header = document.createElement('h3');
        header.textContent = `Answer ${detail.answer.answer_id}`;
        auditDetail.appendChild(header);
        const meta = document.createElement('p');
        meta.textContent = `Mode ${detail.answer.mode || '--'} · Claims ${detail.answer.claims.length} · Citations ${detail.answer.citations_count}`;
        auditDetail.appendChild(meta);
        const exportBtn = document.createElement('button');
        exportBtn.type = 'button';
        exportBtn.textContent = 'Download JSON';
        exportBtn.addEventListener('click', () => {
          const blob = new Blob([JSON.stringify(detail, null, 2)], { type: 'application/json' });
          const url = URL.createObjectURL(blob);
          const link = document.createElement('a');
          link.href = url;
          link.download = `audit-${detail.answer.answer_id}.json`;
          link.click();
          URL.revokeObjectURL(url);
        });
        auditDetail.appendChild(exportBtn);
        const claimsTitle = document.createElement('h4');
        claimsTitle.textContent = 'Claims';
        auditDetail.appendChild(claimsTitle);
        detail.answer.claims.forEach((claim) => {
          const card = document.createElement('div');
          card.className = 'preview-panel';
          const text = document.createElement('p');
          text.textContent = claim.text || '';
          card.appendChild(text);
          const verdict = document.createElement('small');
          verdict.textContent = `Verdict: ${claim.entailment_verdict || '--'}`;
          card.appendChild(verdict);
          if (Array.isArray(claim.citations) && claim.citations.length) {
            const cites = document.createElement('div');
            cites.className = 'hint';
            cites.textContent = `Citations: ${claim.citations.map((c) => c.evidence_id || c.span_id).join(', ')}`;
            card.appendChild(cites);
          }
          auditDetail.appendChild(card);
        });
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
      approveBtn.addEventListener('click', async (event) => {
        event.stopPropagation();
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
      enableBtn.addEventListener('click', async (event) => {
        event.stopPropagation();
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
      disableBtn.addEventListener('click', async (event) => {
        event.stopPropagation();
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

    card.addEventListener('click', () => {
      loadPluginDetail(plugin.plugin_id);
    });

    pluginList.appendChild(card);
  });
  if (activePluginId) {
    loadPluginDetail(activePluginId);
  }
}

function renderPluginPreview(panel, preview) {
  if (!panel) return;
  clearElement(panel);
  if (!preview) return;
  const header = document.createElement('h4');
  header.textContent = `Preview ${preview.preview_id}`;
  panel.appendChild(header);
  const warnings = document.createElement('p');
  warnings.textContent = preview.warnings?.length ? `Warnings: ${preview.warnings.join(', ')}` : 'Warnings: none';
  panel.appendChild(warnings);
  const list = document.createElement('ul');
  (preview.diff || []).forEach((entry) => {
    const li = document.createElement('li');
    const before = typeof entry.before === 'object' ? JSON.stringify(entry.before) : entry.before;
    const after = typeof entry.after === 'object' ? JSON.stringify(entry.after) : entry.after;
    li.textContent = `${entry.path}: ${before} -> ${after}`;
    list.appendChild(li);
  });
  panel.appendChild(list);
}

async function loadPluginDetail(pluginId) {
  if (!pluginDetail) return;
  activePluginId = pluginId;
  pluginPreviewState = null;
  pluginDetail.textContent = 'Loading...';
  try {
    const [detailResp, schemaResp] = await Promise.all([
      apiFetch(`/api/plugins/${pluginId}`),
      apiFetch(`/api/plugins/${pluginId}/schema`)
    ]);
    if (!detailResp.ok) throw new Error('detail failed');
    const detail = await detailResp.json();
    const schema = schemaResp.ok ? await schemaResp.json() : null;
    clearElement(pluginDetail);

    const heading = document.createElement('h3');
    heading.textContent = detail.name || detail.plugin_id;
    pluginDetail.appendChild(heading);

    const meta = document.createElement('p');
    meta.textContent = `ID ${detail.plugin_id} · Version ${detail.version || '--'} · ${detail.enabled ? 'Enabled' : 'Disabled'}`;
    pluginDetail.appendChild(meta);

    if (detail.description) {
      const desc = document.createElement('p');
      desc.textContent = detail.description;
      pluginDetail.appendChild(desc);
    }

    if (schema && Array.isArray(schema.extensions) && schema.extensions.length) {
      const extTitle = document.createElement('h4');
      extTitle.textContent = 'Extensions';
      pluginDetail.appendChild(extTitle);
      schema.extensions.forEach((ext) => {
        const line = document.createElement('div');
        line.className = 'plugin-meta';
        line.textContent = `${ext.kind}:${ext.extension_id}`;
        pluginDetail.appendChild(line);
      });
    }

    const configWrap = document.createElement('div');
    configWrap.className = 'plugin-config';
    const configTitle = document.createElement('h4');
    configTitle.textContent = 'Config (JSON)';
    configWrap.appendChild(configTitle);

    const textarea = document.createElement('textarea');
    textarea.value = JSON.stringify(detail.config || {}, null, 2);
    configWrap.appendChild(textarea);

    const previewPanel = document.createElement('div');
    previewPanel.className = 'preview-panel';
    configWrap.appendChild(previewPanel);

    const actions = document.createElement('div');
    actions.className = 'plugin-config-actions';
    const previewBtn = document.createElement('button');
    previewBtn.type = 'button';
    previewBtn.textContent = 'Preview';
    previewBtn.addEventListener('click', async () => {
      let candidate = {};
      try {
        candidate = textarea.value ? JSON.parse(textarea.value) : {};
      } catch (err) {
        renderPluginPreview(previewPanel, null);
        previewPanel.textContent = 'Invalid JSON.';
        return;
      }
      try {
        const response = await apiFetch(`/api/plugins/${pluginId}/preview`, {
          method: 'POST',
          body: JSON.stringify({ candidate })
        });
        if (!response.ok) throw new Error('preview failed');
        pluginPreviewState = await response.json();
        renderPluginPreview(previewPanel, pluginPreviewState);
      } catch (err) {
        previewPanel.textContent = 'Preview failed.';
      }
    });
    const applyBtn = document.createElement('button');
    applyBtn.type = 'button';
    applyBtn.className = 'secondary';
    applyBtn.textContent = 'Apply';
    applyBtn.addEventListener('click', async () => {
      if (!pluginPreviewState) {
        previewPanel.textContent = 'Run preview first.';
        return;
      }
      let candidate = {};
      try {
        candidate = textarea.value ? JSON.parse(textarea.value) : {};
      } catch (err) {
        previewPanel.textContent = 'Invalid JSON.';
        return;
      }
      try {
        const response = await apiFetch(`/api/plugins/${pluginId}/apply`, {
          method: 'POST',
          body: JSON.stringify({
            candidate,
            preview_id: pluginPreviewState.preview_id
          })
        });
        if (!response.ok) throw new Error('apply failed');
        const data = await response.json();
        previewPanel.textContent = `Applied at ${data.applied_at_utc || '--'}`;
      } catch (err) {
        previewPanel.textContent = 'Apply failed.';
      }
    });
    actions.appendChild(previewBtn);
    actions.appendChild(applyBtn);
    configWrap.appendChild(actions);
    pluginDetail.appendChild(configWrap);

    if (schema && schema.extensions) {
      const schemaBlock = document.createElement('pre');
      schemaBlock.className = 'preview-panel';
      schemaBlock.textContent = `Schema\\n${JSON.stringify(schema.extensions, null, 2)}`;
      pluginDetail.appendChild(schemaBlock);
    }
  } catch (err) {
    pluginDetail.textContent = 'Failed to load plugin detail.';
  }
}

tabs.forEach((btn) => {
  btn.addEventListener('click', () => {
    activateTab(btn.dataset.tab);
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
  await loadExplorerFacets();
  await loadExplorerPage(true);
  await loadPlugins();
  await loadAudit();
  await loadPerfLog();
}

const stateStore = createStateStore((state) => {
  latestState = state;
  renderStatusBar(state, latestStorage);
  renderDashboard(state, latestStorage);
  renderPerfPanel(state);
}, () => {});
stateStore.start();

const storageStore = createStorageStore((stats) => {
  latestStorage = stats;
  renderStatusBar(latestState, stats);
  renderDashboard(latestState, stats);
}, () => {});
storageStore.start();

if (refreshDashboard) {
  refreshDashboard.addEventListener('click', async () => {
    const response = await apiFetch('/api/state', { headers: {} }).catch(() => null);
    if (response && response.ok) {
      latestState = await response.json();
    }
    const storageResp = await apiFetch('/api/storage/stats', { headers: {} }).catch(() => null);
    if (storageResp && storageResp.ok) {
      latestStorage = await storageResp.json();
    }
    renderStatusBar(latestState, latestStorage);
    renderDashboard(latestState, latestStorage);
  });
}

init();

if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/static/sw.js');
}
