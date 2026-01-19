const tabs = document.querySelectorAll('nav button');
const sections = document.querySelectorAll('.tab');
const urlParams = new URLSearchParams(window.location.search);
const unlockToken = urlParams.get('unlock');

function apiHeaders() {
  const headers = { 'Content-Type': 'application/json' };
  if (unlockToken) {
    headers['Authorization'] = `Bearer ${unlockToken}`;
  }
  return headers;
}

function withUnlock(url) {
  if (!unlockToken) return url;
  const separator = url.includes('?') ? '&' : '?';
  return `${url}${separator}unlock=${encodeURIComponent(unlockToken)}`;
}

async function apiFetch(path, options = {}) {
  const headers = { ...apiHeaders(), ...(options.headers || {}) };
  return fetch(path, { ...options, headers });
}

tabs.forEach((btn) => {
  btn.addEventListener('click', () => {
    tabs.forEach((b) => b.classList.remove('active'));
    sections.forEach((s) => s.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(btn.dataset.tab).classList.add('active');
  });
});

const thread = document.getElementById('thread');
const chatForm = document.getElementById('chatForm');
const chatInput = document.getElementById('chatInput');
const modelSelect = document.getElementById('modelSelect');
const sanitizeToggle = document.getElementById('sanitizeToggle');
const extractiveToggle = document.getElementById('extractiveToggle');
const timeRange = document.getElementById('timeRange');
const safeModeIndicator = document.getElementById('safeModeIndicator');

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
    routing: { llm: modelSelect.value === 'cloud' ? 'openai' : 'ollama' },
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

const saveSettings = document.getElementById('saveSettings');
const presetToggle = document.getElementById('privacyPresetToggle');
const storageTtl = document.getElementById('storageTtl');
const storageUsage = document.getElementById('storageUsage');
const storagePath = document.getElementById('storagePath');
const refreshStorage = document.getElementById('refreshStorage');
const promptRepeatSelect = document.getElementById('promptRepeatSelect');
const stepByStepToggle = document.getElementById('stepByStepToggle');
const stepByStepPhrase = document.getElementById('stepByStepPhrase');

saveSettings.addEventListener('click', async () => {
  const repeatStrategy = promptRepeatSelect ? promptRepeatSelect.value : 'off';
  const promptStrategyDefault = repeatStrategy === 'off' ? 'baseline' : repeatStrategy;
  const stepPhrase = stepByStepPhrase ? stepByStepPhrase.value.trim() : '';
  const payload = {
    active_preset: presetToggle.checked ? 'privacy_first' : 'high_fidelity',
    routing: {
      ocr: document.getElementById('routingOcr').value,
      embedding: document.getElementById('routingEmbedding').value,
      retrieval: document.getElementById('routingRetrieval').value,
      compressor: document.getElementById('routingCompressor').value,
      verifier: document.getElementById('routingVerifier').value
    },
    llm: {
      prompt_strategy_default: promptStrategyDefault,
      strategy_auto_mode: false,
      enable_step_by_step: stepByStepToggle ? stepByStepToggle.checked : false,
      step_by_step_phrase: stepPhrase || "Let's think step by step."
    }
  };
  await apiFetch('/api/settings', {
    method: 'POST',
    body: JSON.stringify({ settings: payload })
  });
  alert('Settings saved locally.');
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

loadHighlightsList();

async function loadSettings() {
  try {
    const response = await apiFetch('/api/settings');
    const data = await response.json();
    const routing = (data.settings && data.settings.routing) || {};
    if (routing.ocr) document.getElementById('routingOcr').value = routing.ocr;
    if (routing.embedding) document.getElementById('routingEmbedding').value = routing.embedding;
    if (routing.retrieval) document.getElementById('routingRetrieval').value = routing.retrieval;
    if (routing.compressor) document.getElementById('routingCompressor').value = routing.compressor;
    if (routing.verifier) document.getElementById('routingVerifier').value = routing.verifier;
    if (routing.llm) {
      modelSelect.value = routing.llm.startsWith('openai') ? 'cloud' : 'local';
    }
    if (data.settings && data.settings.active_preset) {
      presetToggle.checked = data.settings.active_preset === 'privacy_first';
    }
    const llm = (data.settings && data.settings.llm) || {};
    if (promptRepeatSelect) {
      const strategy = llm.prompt_strategy_default || 'repeat_2x';
      promptRepeatSelect.value = strategy === 'baseline' ? 'off' : strategy;
    }
    if (stepByStepToggle) {
      stepByStepToggle.checked = Boolean(llm.enable_step_by_step);
    }
    if (stepByStepPhrase) {
      stepByStepPhrase.value = llm.step_by_step_phrase || "Let's think step by step.";
    }
  } catch (err) {
    return;
  }
}

async function loadStatus() {
  try {
    const response = await apiFetch('/health', { headers: {} });
    const data = await response.json();
    const status = document.getElementById('remoteStatus');
    status.textContent = data.mode || 'local-only';
  } catch (err) {
    return;
  }
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

async function loadStorage() {
  if (!storageTtl || !storageUsage || !storagePath) return;
  try {
    const response = await apiFetch('/api/storage');
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'request failed');
    storageTtl.textContent = `${data.screenshot_ttl_days || '--'}`;
    storageUsage.textContent = formatBytes(data.media_usage_bytes);
    storagePath.textContent = data.media_path || '--';
  } catch (err) {
    storageTtl.textContent = '--';
    storageUsage.textContent = 'error';
    storagePath.textContent = '--';
  }
}

if (refreshStorage) {
  refreshStorage.addEventListener('click', async () => {
    await loadStorage();
  });
}

loadSettings();
loadStatus();
loadStorage();

if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/static/sw.js');
}
