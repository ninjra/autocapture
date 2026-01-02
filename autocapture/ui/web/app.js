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
  div.innerHTML = `<strong>${author}</strong><span>${text}</span>`;
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

  const response = await fetch('/api/answer', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  const data = await response.json();
  appendMessage('Assistant', `${data.answer}<br/><small>Citations: ${data.citations.join(', ')}</small>`);
});

const searchForm = document.getElementById('searchForm');
const searchInput = document.getElementById('searchInput');
const searchResults = document.getElementById('searchResults');

searchForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const query = searchInput.value.trim();
  if (!query) return;
  searchResults.innerHTML = '';
  const response = await fetch('/api/retrieve', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, k: 10 })
  });
  const data = await response.json();
  data.evidence.forEach((item) => {
    const card = document.createElement('div');
    card.className = 'result-card';
    card.innerHTML = `<strong>${item.app}</strong><p>${item.text}</p><small>${item.event_id}</small>`;
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
