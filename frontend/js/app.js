(function () {
  'use strict';

  const API_BASE = ''; // 同源，与 FastAPI 同端口

  // 后端 GET /api/v1/models 返回后覆盖；未加载前用默认
  let BUILTIN_LLM = [
    { id: 'llama3.2', name: 'Llama 3.2' },
    { id: 'qwen2-0.5b', name: 'Qwen2 0.5B' },
  ];

  const BUILTIN_EMBED = [
    { id: 'bge', name: 'BGE' },
    { id: 'gte', name: 'GTE' },
  ];

  let state = {
    page: 'launch',
    tab: 'llm',
    selectedModel: null,
    running: [],
    nextRunningId: 1,
    modelsLoaded: false,
    inferenceRecord: null,
    chatMessages: [],
  };

  var loadRunningAbortController = null;

  const $ = (sel, el = document) => el.querySelector(sel);
  const $$ = (sel, el = document) => el.querySelectorAll(sel);

  function renderModelCards() {
    const list = state.tab === 'llm' ? BUILTIN_LLM : BUILTIN_EMBED;
    const containerLlm = $('#model-cards-llm');
    const containerEmbed = $('#model-cards-embed');
    const container = state.tab === 'llm' ? containerLlm : containerEmbed;
    const other = state.tab === 'llm' ? containerEmbed : containerLlm;
    other.classList.add('hidden');
    container.classList.remove('hidden');
    container.innerHTML = list
      .map(
        (m) =>
          `<div class="model-card" data-id="${m.id}" data-name="${m.name}">${m.name}</div>`
      )
      .join('');

    container.querySelectorAll('.model-card').forEach((card) => {
      card.addEventListener('click', () => selectModel(card));
    });

    containerLlm.querySelectorAll('.model-card').forEach((card) => {
      if (state.selectedModel && state.selectedModel.id === card.dataset.id) {
        card.classList.add('selected');
      }
    });
    containerEmbed.querySelectorAll('.model-card').forEach((card) => {
      if (state.selectedModel && state.selectedModel.id === card.dataset.id) {
        card.classList.add('selected');
      }
    });
  }

  function selectModel(cardEl) {
    $$('.model-card').forEach((c) => c.classList.remove('selected'));
    cardEl.classList.add('selected');
    state.selectedModel = { id: cardEl.dataset.id, name: cardEl.dataset.name };
    showConfigForm(state.selectedModel);
  }

  function openConfigDrawer() {
    $('#config-drawer-backdrop').classList.add('open');
    $('#config-drawer').classList.add('open');
    $('#config-drawer-backdrop').setAttribute('aria-hidden', 'false');
    $('#config-drawer').setAttribute('aria-hidden', 'false');
  }

  function closeConfigDrawer() {
    $('#config-drawer-backdrop').classList.remove('open');
    $('#config-drawer').classList.remove('open');
    $('#config-drawer-backdrop').setAttribute('aria-hidden', 'true');
    $('#config-drawer').setAttribute('aria-hidden', 'true');
  }

  function closeConfigPanel() {
    closeConfigDrawer();
    state.selectedModel = null;
    $$('.model-card').forEach((c) => c.classList.remove('selected'));
  }

  function showConfigForm(model) {
    const form = $('#config-form');
    const nameEl = $('#config-model-name');
    if (!model) {
      closeConfigDrawer();
      return;
    }
    nameEl.textContent = model.name;
    form.dataset.modelId = model.id;
    form.dataset.modelName = model.name;
    resetFormToDefault();
    openConfigDrawer();
  }

  function resetFormToDefault() {
    const form = $('#config-form');
    if (!form) return;
    form.engine.value = 'ollama';
    form.format.value = 'pytorch';
    form.size.value = '3B';
    form.quantization.value = 'none';
    form.gpu_count.value = 'auto';
    form.replicas.value = '1';
    form.thought_mode.checked = true;
    form.parse_inference.checked = false;
    form.extra.value = '';
  }

  function getFormValues() {
    const form = $('#config-form');
    if (!form) return null;
    return {
      engine: form.engine.value,
      format: form.format.value,
      size: form.size.value,
      quantization: form.quantization.value,
      gpu_count: form.gpu_count.value.trim() || 'auto',
      replicas: parseInt(form.replicas.value, 10) || 1,
      thought_mode: form.thought_mode.checked,
      parse_inference: form.parse_inference.checked,
      extra: form.extra.value.trim(),
    };
  }

  function setLaunchProgressPercent(pct) {
    const fill = $('#launch-progress-fill');
    if (fill) fill.style.width = Math.min(100, Math.max(0, pct)) + '%';
  }

  function showLaunchProgress(show) {
    const wrap = $('#launch-progress-wrap');
    const fill = $('#launch-progress-fill');
    if (!wrap || !fill) return;
    if (show) {
      fill.style.width = '0%';
      wrap.classList.remove('hidden');
      wrap.setAttribute('aria-hidden', 'false');
    } else {
      wrap.classList.add('hidden');
      wrap.setAttribute('aria-hidden', 'true');
      fill.style.width = '0%';
    }
  }

  function finishLaunchProgress() {
    setLaunchProgressPercent(100);
    setTimeout(function () {
      showLaunchProgress(false);
    }, 350);
  }

  function onLaunch(e) {
    e.preventDefault();
    const modelId = $('#config-form').dataset.modelId;
    const modelName = $('#config-form').dataset.modelName;
    const cfg = getFormValues();
    if (!modelId || !modelName || !cfg) return;

    const btn = $('#btn-launch');
    if (btn.disabled) return;

    btn.disabled = true;
    showLaunchProgress(true);
    setLaunchProgressPercent(0);
    if (loadRunningAbortController) {
      loadRunningAbortController.abort();
      loadRunningAbortController = null;
    }

    const payload = {
      model_id: modelId,
      engine_type: cfg.engine,
      format: cfg.format,
      size: cfg.size,
      quantization: cfg.quantization,
      gpu_count: cfg.gpu_count,
      replicas: cfg.replicas,
      thought_mode: cfg.thought_mode,
      parse_inference: cfg.parse_inference,
    };

    var progressTick = setInterval(function () {
      var fill = $('#launch-progress-fill');
      if (!fill) return;
      var w = parseFloat(fill.style.width) || 0;
      if (w < 90) setLaunchProgressPercent(w + 8);
    }, 800);

    fetch(API_BASE + '/api/v1/models/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
      .then(function (res) {
        return res.json().then(function (data) {
          if (!res.ok) {
            throw new Error(data.msg || data.detail || '启动失败');
          }
          if (data.code !== 200 || !data.data) {
            throw new Error(data.msg || '启动失败');
          }
          var runId = data.data.run_id || data.data.uid;
          var address = data.data.address || 'local:' + runId;
          var record = {
            id: runId,
            run_id: runId,
            name: modelName,
            modelId: modelId,
            address: address,
            gpuIndex: cfg.gpu_count === 'auto' ? 'auto' : cfg.gpu_count,
            quantization: cfg.quantization,
            size: cfg.size,
            replicas: cfg.replicas,
            engine: cfg.engine,
            addedAt: Date.now(),
          };
          state.running.push(record);
          closeConfigPanel();
          setTab('llm');
          setPage('running');
          renderRunningTable();
          return data;
        });
      })
      .catch(function (err) {
        alert('启动失败: ' + (err.message || String(err)));
      })
      .finally(function () {
        clearInterval(progressTick);
        finishLaunchProgress();
        btn.disabled = false;
        btn.innerHTML = '<span class="icon">🚀</span> 启动';
      });
  }

  function stopRunning(id) {
    const record = state.running.find(function (r) {
      return r.id === id;
    });
    if (record && record.run_id) {
      fetch(API_BASE + '/api/v1/models/running/' + encodeURIComponent(record.run_id) + '/stop', {
        method: 'POST',
      })
        .then(function (res) {
          return res.json();
        })
        .then(function () {
          state.running = state.running.filter(function (r) {
            return r.id !== id;
          });
          renderRunningTable();
        })
        .catch(function () {
          state.running = state.running.filter(function (r) {
            return r.id !== id;
          });
          renderRunningTable();
        });
    } else {
      state.running = state.running.filter(function (r) {
        return r.id !== id;
      });
      renderRunningTable();
    }
  }

  function renderRunningTable() {
    const tbody = $('#running-tbody');
    const empty = $('#running-empty');
    const wrap = $('.table-wrap');
    // 运行模型页：语言模型 tab 显示 ollama/vllm/sglang，嵌入 tab 显示嵌入模型
    const list = state.tab === 'llm'
      ? state.running.filter((r) => r.engine && ['ollama', 'vllm', 'sglang'].includes(String(r.engine).toLowerCase()))
      : state.running.filter((r) => BUILTIN_EMBED.some((m) => m.id === r.modelId));
    if (list.length === 0) {
      tbody.innerHTML = '';
      if (empty) empty.classList.remove('hidden');
      if (wrap) wrap.classList.add('table-wrap-empty');
      return;
    }
    if (empty) empty.classList.add('hidden');
    if (wrap) wrap.classList.remove('table-wrap-empty');
    tbody.innerHTML = list
      .map(
        (r) =>
          `<tr>
            <td class="run-id-cell">${r.id.length > 12 ? r.id.slice(0, 8) + '…' : r.id}</td>
            <td>${r.name}</td>
            <td>${r.address}</td>
            <td>${r.engine}</td>
            <td>${r.gpuIndex != null ? r.gpuIndex : 'auto'}</td>
            <td>${r.quantization != null ? r.quantization : '-'}</td>
            <td>${r.size != null ? r.size : '-'}</td>
            <td>${r.replicas != null ? r.replicas : 1}</td>
            <td class="actions-cell">
              <button type="button" class="btn btn-sm btn-primary btn-infer" data-run-index="${list.indexOf(r)}">推理</button>
              <button type="button" class="btn btn-sm btn-danger" data-stop-id="${r.id}">停止</button>
            </td>
          </tr>`
      )
      .join('');

    tbody.querySelectorAll('[data-stop-id]').forEach((btn) => {
      btn.addEventListener('click', () => stopRunning(btn.dataset.stopId));
    });
    tbody.querySelectorAll('.btn-infer').forEach((btn) => {
      const idx = parseInt(btn.dataset.runIndex, 10);
      const record = list[idx];
      if (record) btn.addEventListener('click', () => openInferenceDrawer(record));
    });
  }

  function setPage(page) {
    state.page = page;
    $$('.nav-item').forEach((a) => a.classList.toggle('active', a.dataset.page === page));
    $$('.page').forEach((p) => p.classList.toggle('active', p.id === 'page-' + page));
    if (page === 'running') {
      state.tab = 'llm';
      $$('.tabs .tab').forEach((t) => t.classList.toggle('active', t.dataset.tab === 'llm'));
      renderRunningTable();
    }
  }

  function setTab(tab) {
    state.tab = tab;
    $$('.tabs .tab').forEach((t) => t.classList.toggle('active', t.dataset.tab === tab));
    renderModelCards();
    if (state.page === 'running') renderRunningTable();
  }

  function openInferenceDrawer(record) {
    state.inferenceRecord = record;
    state.chatMessages = [];
    $('#inference-model-name').textContent = record.name + ' - 推理';
    $('#inference-prompt').value = '';
    $('#inference-result-generate').textContent = '';
    $('#inference-result-chat').textContent = '';
    $('#chat-messages').innerHTML = '';
    $('#chat-input').value = '';
    $('#inference-panel-generate').classList.remove('hidden');
    $('#inference-panel-chat').classList.add('hidden');
    $$('.inference-tab').forEach((t) => t.classList.toggle('active', t.dataset.inferenceTab === 'generate'));
    $('#inference-backdrop').classList.add('open');
    $('#inference-drawer').classList.add('open');
  }

  function closeInferenceDrawer() {
    $('#inference-backdrop').classList.remove('open');
    $('#inference-drawer').classList.remove('open');
    state.inferenceRecord = null;
    state.chatMessages = [];
  }

  function buildInferenceBody(record, promptOrMessages, temperature, maxTokens) {
    const temperatureVal = parseFloat(temperature, 10) || 0.7;
    const maxTokensVal = parseInt(maxTokens, 10) || 1024;
    const base = { temperature: temperatureVal, max_tokens: maxTokensVal };
    if (record.run_id) {
      base.run_id = record.run_id;
    } else {
      base.engine_type = record.engine;
      base.model_name = record.modelId;
    }
    return base;
  }

  function callGenerate() {
    const record = state.inferenceRecord;
    if (!record) return;
    const prompt = $('#inference-prompt').value.trim();
    if (!prompt) {
      alert('请输入提示词');
      return;
    }
    const temperature = $('#inference-temperature').value;
    const maxTokens = $('#inference-max-tokens').value;
    const body = buildInferenceBody(record, prompt, temperature, maxTokens);
    body.prompt = prompt;

    const resultEl = $('#inference-result-generate');
    resultEl.textContent = '生成中...';
    const btn = $('#btn-generate');
    btn.disabled = true;

    fetch(API_BASE + '/api/v1/llm/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
      .then(function (res) {
        return res.json();
      })
      .then(function (data) {
        if (data.code !== 200 || !data.data) {
          resultEl.textContent = '错误: ' + (data.msg || '生成失败');
          return;
        }
        const text = data.data.response;
        resultEl.textContent = typeof text === 'string' ? text : JSON.stringify(text);
      })
      .catch(function (err) {
        resultEl.textContent = '请求失败: ' + (err.message || String(err));
      })
      .finally(function () {
        btn.disabled = false;
      });
  }

  function callChatSend() {
    const record = state.inferenceRecord;
    if (!record) return;
    const input = $('#chat-input').value.trim();
    if (!input) {
      alert('请输入消息');
      return;
    }
    state.chatMessages.push({ role: 'user', content: input });
    $('#chat-input').value = '';
    renderChatMessages();

    const temperature = ($('#chat-temperature') && $('#chat-temperature').value) || '0.7';
    const maxTokens = ($('#chat-max-tokens') && $('#chat-max-tokens').value) || '1024';
    const body = buildInferenceBody(record, state.chatMessages, temperature, maxTokens);
    body.messages = state.chatMessages.map(function (m) {
      return { role: m.role, content: m.content };
    });

    const resultEl = $('#inference-result-chat');
    resultEl.textContent = '回复中...';
    const btn = $('#btn-chat-send');
    btn.disabled = true;

    fetch(API_BASE + '/api/v1/llm/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
      .then(function (res) {
        return res.json();
      })
      .then(function (data) {
        if (data.code !== 200 || !data.data) {
          resultEl.textContent = '错误: ' + (data.msg || '对话失败');
          return;
        }
        const text = data.data.response;
        const reply = typeof text === 'string' ? text : (text && text.content) || JSON.stringify(text);
        state.chatMessages.push({ role: 'assistant', content: reply });
        renderChatMessages();
        resultEl.textContent = '';
      })
      .catch(function (err) {
        resultEl.textContent = '请求失败: ' + (err.message || String(err));
      })
      .finally(function () {
        btn.disabled = false;
      });
  }

  function renderChatMessages() {
    const container = $('#chat-messages');
    container.innerHTML = state.chatMessages
      .map(function (m) {
        const cls = m.role === 'user' ? 'chat-msg-user' : 'chat-msg-assistant';
        return '<div class="' + cls + '"><strong>' + m.role + '</strong>: ' + escapeHtml(m.content) + '</div>';
      })
      .join('');
  }

  function escapeHtml(s) {
    const div = document.createElement('div');
    div.textContent = s;
    return div.innerHTML;
  }

  function loadModelsFromBackend() {
    fetch(API_BASE + '/api/v1/models')
      .then(function (res) {
        return res.json();
      })
      .then(function (data) {
        if (data.code === 200 && data.data && Array.isArray(data.data.models)) {
          BUILTIN_LLM = data.data.models.map(function (m) {
            return { id: m.id, name: m.name || m.id };
          });
          state.modelsLoaded = true;
        }
      })
      .catch(function () {})
      .finally(function () {
        renderModelCards();
      });
  }

  function mapBackendToRecord(r) {
    return {
      id: r.run_id,
      run_id: r.run_id,
      name: r.model_name || r.model_id,
      modelId: r.model_id,
      address: r.address,
      gpuIndex: r.gpu_count != null ? String(r.gpu_count) : 'auto',
      quantization: r.quantization != null ? String(r.quantization) : 'none',
      size: r.size != null ? String(r.size) : '-',
      replicas: r.replicas != null ? Number(r.replicas) : 1,
      engine: r.engine_type,
    };
  }

  function loadRunningFromBackend() {
    if (loadRunningAbortController) loadRunningAbortController.abort();
    loadRunningAbortController = new AbortController();
    var signal = loadRunningAbortController.signal;

    fetch(API_BASE + '/api/v1/models/running', { signal: signal })
      .then(function (res) {
        return res.json();
      })
      .then(function (data) {
        if (data.code !== 200 || !data.data || !Array.isArray(data.data.running)) return;
        state.running = data.data.running.map(mapBackendToRecord);
        renderRunningTable();
      })
      .catch(function (err) {
        if (err && err.name === 'AbortError') return;
      })
      .finally(function () {
        if (loadRunningAbortController && loadRunningAbortController.signal.aborted) return;
        loadRunningAbortController = null;
      });
  }

  function init() {
    renderModelCards();
    renderRunningTable();
    loadModelsFromBackend();
    loadRunningFromBackend();

    $$('.nav-item').forEach((a) => {
      a.addEventListener('click', (e) => {
        e.preventDefault();
        setPage(a.dataset.page);
      });
    });

    $$('.tabs .tab').forEach((t) => {
      t.addEventListener('click', () => setTab(t.dataset.tab));
    });

    $('#btn-refresh-running')?.addEventListener('click', function () {
      loadRunningFromBackend();
    });
    $('#config-form')?.addEventListener('submit', onLaunch);
    $('#btn-cancel')?.addEventListener('click', closeConfigPanel);
    $('#config-drawer-backdrop')?.addEventListener('click', closeConfigPanel);

    $('#btn-close-inference')?.addEventListener('click', closeInferenceDrawer);
    $('#inference-backdrop')?.addEventListener('click', closeInferenceDrawer);
    $('#btn-generate')?.addEventListener('click', callGenerate);
    $('#btn-chat-send')?.addEventListener('click', callChatSend);
    $$('.inference-tab').forEach(function (t) {
      t.addEventListener('click', function () {
        const tab = t.dataset.inferenceTab;
        $$('.inference-tab').forEach(function (x) {
          x.classList.toggle('active', x.dataset.inferenceTab === tab);
        });
        $('#inference-panel-generate').classList.toggle('hidden', tab !== 'generate');
        $('#inference-panel-chat').classList.toggle('hidden', tab !== 'chat');
      });
    });
  }

  init();
})();
