(function () {
  'use strict';

  // 嵌入模式：URL 参数 embed=1 或在 iframe 中时隐藏侧栏；api_base 可指定后端地址
  const params = new URLSearchParams(window.location.search);
  const isEmbed = params.get('embed') === '1' || (window.self !== window.top);
  if (isEmbed) {
    document.body.classList.add('embed-mode');
  }
  const API_BASE = params.get('api_base') || ''; // 同源或由父页面传入

  // 后端 GET /api/v1/models 返回后覆盖；每项可为 { id, name, description?, sizes, quantizations, engines, formats }
  let BUILTIN_LLM = [
    { id: 'llama3.2', name: 'Llama 3.2', sizes: [{ size: '1B' }], quantizations: ['none'], engines: ['ollama', 'vllm', 'sglang'], formats: ['pytorch', 'safetensors'] },
    { id: 'qwen2', name: 'Qwen2', sizes: [{ size: '0.5B' }], quantizations: ['none'], engines: ['ollama', 'vllm', 'sglang'], formats: ['pytorch', 'safetensors'] },
  ];

  const BUILTIN_EMBED = [
    { id: 'bge', name: 'BGE' },
    { id: 'gte', name: 'GTE' },
  ];

  let state = {
    tab: 'llm',
    selectedModel: null,
    running: [],
    nextRunningId: 1,
    modelsLoaded: false,
    inferenceRecord: null,
    chatMessages: [],
    logAutoRefreshTimer: null,
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
    const isLlm = state.tab === 'llm';
    container.innerHTML = list
      .map(function (m) {
        const desc = (isLlm && m.description) ? m.description : '模型简介，支持生成与对话。';
        const sizesLabel = (isLlm && m.sizes && m.sizes.length) ? m.sizes.map(function (s) { return s.size || s; }).join(' / ') : '';
        const tags = sizesLabel ? sizesLabel + ' · generate model' : '4K · generate model';
        return '<div class="model-card" data-id="' + escapeHtml(m.id) + '" data-name="' + escapeHtml(m.name) + '">' +
          '<div class="name">' + escapeHtml(m.name) + '</div>' +
          '<div class="desc">' + escapeHtml(desc) + '</div>' +
          '<div class="tags">' + escapeHtml(tags) + '</div>' +
          '</div>';
      })
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
    const id = cardEl.dataset.id;
    const name = cardEl.dataset.name;
    const full = (state.tab === 'llm' && BUILTIN_LLM) ? BUILTIN_LLM.find(function (m) { return m.id === id; }) : null;
    state.selectedModel = full || { id: id, name: name };
    showConfigForm(state.selectedModel);
  }

  function closeConfigPanel() {
    state.selectedModel = null;
    var nameEl = document.getElementById('config-model-name');
    if (nameEl) nameEl.textContent = '请选择模型';
    $$('.model-card').forEach((c) => c.classList.remove('selected'));
  }

  function showConfigForm(model) {
    const form = $('#config-form');
    const nameEl = $('#config-model-name');
    if (!model) return;
    nameEl.textContent = model.name || model.id;
    form.dataset.modelId = model.id;
    form.dataset.modelName = model.name || model.id;
    fillConfigOptionsFromModel(model);
  }

  function fillConfigOptionsFromModel(model) {
    const form = $('#config-form');
    if (!form) return;
    var engineSel = form.querySelector('#config-engine') || form.querySelector('[name="engine"]');
    var formatSel = form.querySelector('#config-format') || form.querySelector('[name="format"]');
    var sizeSel = form.querySelector('#config-size') || form.querySelector('[name="size"]');
    var quantSel = form.querySelector('#config-quantization') || form.querySelector('[name="quantization"]');
    var engines = (model.engines && model.engines.length) ? model.engines : ['ollama', 'vllm', 'sglang'];
    var formats = (model.formats && model.formats.length) ? model.formats : ['pytorch', 'safetensors'];
    var sizes = (model.sizes && model.sizes.length) ? model.sizes : [{ size: '1B' }];
    var quants = (model.quantizations && model.quantizations.length) ? model.quantizations : ['none'];
    var engineLabels = { ollama: 'Ollama', vllm: 'vLLM', sglang: 'SGLang' };
    var formatLabels = { pytorch: 'PyTorch', safetensors: 'SafeTensors' };
    var quantLabels = { none: '无', int4: 'INT4', int8: 'INT8' };
    if (engineSel) {
      engineSel.innerHTML = engines.map(function (v) { return '<option value="' + v + '">' + (engineLabels[v] || v) + '</option>'; }).join('');
      engineSel.value = engines[0];
    }
    if (formatSel) {
      formatSel.innerHTML = formats.map(function (v) { return '<option value="' + v + '">' + (formatLabels[v] || v) + '</option>'; }).join('');
      formatSel.value = formats[0];
    }
    if (sizeSel) {
      sizeSel.innerHTML = sizes.map(function (s) {
        var sizeVal = typeof s === 'string' ? s : (s.size || s.hf_repo || '');
        sizeVal = String(sizeVal || '');
        return '<option value="' + escapeHtml(sizeVal) + '">' + escapeHtml(sizeVal) + '</option>';
      }).join('');
      var firstSize = sizes[0];
      sizeSel.value = firstSize ? (typeof firstSize === 'string' ? firstSize : (firstSize.size || '1B')) : '1B';
    }
    if (quantSel) {
      quantSel.innerHTML = quants.map(function (v) { return '<option value="' + v + '">' + (quantLabels[v] || v) + '</option>'; }).join('');
      quantSel.value = quants[0];
    }
    form.gpu_count.value = 'auto';
    form.replicas.value = '1';
    form.thought_mode.checked = true;
    form.parse_inference.checked = false;
    if (form.extra) form.extra.value = '';
  }

  function resetFormToDefault() {
    const form = $('#config-form');
    if (!form) return;
    form.engine.value = 'ollama';
    form.format.value = 'pytorch';
    form.size.value = '1B';
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
    const form = $('#config-form');
    const modelId = form && form.dataset.modelId;
    const modelName = form && form.dataset.modelName;
    const cfg = getFormValues();
    if (!modelId || !modelName || !cfg) {
      alert('请先选择模型');
      return;
    }

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
          renderRunningTable();
          loadLogsFromBackend();
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
          loadLogsFromBackend();
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
      tbody.innerHTML = '<tr><td colspan="9" class="table-empty-cell">暂无运行中的模型</td></tr>';
      if (empty) empty.classList.add('hidden');
      if (wrap) wrap.classList.remove('table-wrap-empty');
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

  function setTab(tab) {
    state.tab = tab;
    $$('.tabs .tab').forEach((t) => t.classList.toggle('active', t.dataset.tab === tab));
    renderModelCards();
    renderRunningTable();
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
    if (s == null) return '';
    const div = document.createElement('div');
    div.textContent = s;
    return div.innerHTML;
  }

  function formatLogTs(ts) {
    if (ts == null) return '';
    var d = new Date(ts * 1000);
    var pad = function (n) { return n < 10 ? '0' + n : n; };
    return d.getFullYear() + '-' + pad(d.getMonth() + 1) + '-' + pad(d.getDate()) + ' ' +
      pad(d.getHours()) + ':' + pad(d.getMinutes()) + ':' + pad(d.getSeconds());
  }

  function renderLogs(logs) {
    const el = $('#log-content');
    if (!el) return;
    if (!logs || !logs.length) {
      el.textContent = '暂无运行日志';
      return;
    }
    var levelClass = function (level) {
      if (level === 'ERROR') return 'log-line--error';
      if (level === 'WARNING') return 'log-line--warning';
      return 'log-line--info';
    };
    el.innerHTML = logs
      .map(function (e) {
        var ts = formatLogTs(e.ts);
        var level = (e.level || 'INFO').toUpperCase();
        var msg = e.message || '';
        var extra = [];
        if (e.engine) extra.push('engine=' + e.engine);
        if (e.run_id) extra.push('run_id=' + (e.run_id.length > 8 ? e.run_id.slice(0, 8) + '…' : e.run_id));
        if (e.model_id) extra.push('model=' + e.model_id);
        var line = '[' + ts + '] ' + level + '  ' + msg;
        if (extra.length) line += '  (' + extra.join(', ') + ')';
        return '<span class="' + levelClass(level) + '">' + escapeHtml(line) + '</span>';
      })
      .join('\n');
    el.scrollTop = el.scrollHeight;
  }

  function loadLogsFromBackend() {
    fetch(API_BASE + '/api/v1/logs?limit=200')
      .then(function (res) { return res.json(); })
      .then(function (data) {
        if (data.code === 200 && data.data && Array.isArray(data.data.logs)) {
          renderLogs(data.data.logs);
        } else {
          $('#log-content').textContent = '加载失败';
        }
      })
      .catch(function () {
        $('#log-content').textContent = '加载失败';
      });
  }

  function loadModelsFromBackend() {
    fetch(API_BASE + '/api/v1/models')
      .then(function (res) {
        return res.json();
      })
      .then(function (data) {
        if (data.code === 200 && data.data && Array.isArray(data.data.models)) {
          BUILTIN_LLM = data.data.models.map(function (m) {
            return {
              id: m.id,
              name: m.name || m.id,
              description: m.description || '',
              sizes: m.sizes || [],
              quantizations: m.quantizations || ['none'],
              engines: m.engines || ['ollama', 'vllm', 'sglang'],
              formats: m.formats || ['pytorch', 'safetensors'],
            };
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
    loadLogsFromBackend();

    $$('.tabs .tab').forEach((t) => {
      t.addEventListener('click', () => setTab(t.dataset.tab));
    });

    $('#btn-refresh-running')?.addEventListener('click', function () {
      loadRunningFromBackend();
    });

    $('#btn-refresh-logs')?.addEventListener('click', function () {
      loadLogsFromBackend();
    });

    $('#log-auto-refresh')?.addEventListener('change', function () {
      if (state.logAutoRefreshTimer) {
        clearInterval(state.logAutoRefreshTimer);
        state.logAutoRefreshTimer = null;
      }
      if (this.checked) {
        state.logAutoRefreshTimer = setInterval(loadLogsFromBackend, 3000);
      }
    });
    $('#config-form')?.addEventListener('submit', onLaunch);
    $('#btn-cancel')?.addEventListener('click', closeConfigPanel);

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
