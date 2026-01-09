const tabs = document.querySelectorAll('nav button');
const sections = document.querySelectorAll('.tab');

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
const sanitizeToggle = document.getElementById('sanitizeToggle');
const extractiveToggle = document.getElementById('extractiveToggle');
const cloudToggle = document.getElementById('cloudToggle');
const timeRange = document.getElementById('timeRange');

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

function appendAssistantMessage(answer, citations) {
  const div = document.createElement('div');
  div.className = 'msg';
  const strong = document.createElement('strong');
  strong.textContent = 'Assistant';
  const span = document.createElement('span');
  span.textContent = answer;
  div.appendChild(strong);
  div.appendChild(span);
  if (Array.isArray(citations) && citations.length > 0) {
    const citationText = document.createElement('small');
    citationText.className = 'citations';
    citationText.textContent = `Citations: ${citations.join(', ')}`;
    div.appendChild(citationText);
  }
  thread.appendChild(div);
  thread.scrollTop = thread.scrollHeight;
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
    routing: { llm: cloudToggle.checked ? 'openai' : 'ollama' },
    time_range: rangeToTuple(timeRange.value)
  };

  let response;
  let data = {};
  try {
    response = await fetch('/api/answer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
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
  appendAssistantMessage(data.answer || '', data.citations || []);
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
    response = await fetch('/api/retrieve', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, k: 10 })
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
    searchResults.appendChild(card);
  });
});

const saveSettings = document.getElementById('saveSettings');

saveSettings.addEventListener('click', async () => {
  const payload = {
    routing: {
      ocr: document.getElementById('routingOcr').value,
      embedding: document.getElementById('routingEmbedding').value,
      retrieval: document.getElementById('routingRetrieval').value,
      compressor: document.getElementById('routingCompressor').value,
      verifier: document.getElementById('routingVerifier').value
    }
  };
  await fetch('/api/settings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ settings: payload })
  });
  alert('Settings saved locally.');
});

if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/static/sw.js');
}
