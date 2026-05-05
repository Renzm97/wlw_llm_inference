(function () {
  'use strict';

  // Promise.prototype.finally polyfill for older browsers (Chrome < 63, Safari < 11.1)
  if (!Promise.prototype.finally) {
    Promise.prototype.finally = function (callback) {
      var P = this.constructor;
      return this.then(
        function (value) { return P.resolve(callback()).then(function () { return value; }); },
        function (reason) { return P.resolve(callback()).then(function () { throw reason; }); }
      );
    };
  }

  // 嵌入模式：URL 参数 embed=1 或在 iframe 中时隐藏侧栏；api_base 可指定后端地址
  var params;
  if (typeof URLSearchParams !== 'undefined') {
    params = new URLSearchParams(window.location.search);
  } else {
    params = {
      get: function (key) {
        var m = window.location.search.match(new RegExp('[?&]' + key + '=([^&]+)'));
        return m ? decodeURIComponent(m[1]) : null;
      }
    };
  }
  const isEmbed = params.get('embed') === '1' || (window.self !== window.top);
  if (isEmbed) {
    document.body.classList.add('embed-mode');
  }
  const API_BASE = params.get('api_base') || ''; // 同源或由父页面传入

  // 后端 GET /api/v1/models 返回后覆盖；每项可为 { id, name, description?, sizes, quantizations, engines, formats }
  let BUILTIN_LLM = [
    { id: 'llama3.2', name: 'Llama 3.2', sizes: [{ size: '1B' }], quantizations: ['none'], engines: ['vllm', 'ollama', 'sglang'], formats: ['pytorch', 'safetensors'] },
    { id: 'qwen2', name: 'Qwen2', sizes: [{ size: '0.5B' }], quantizations: ['none'], engines: ['vllm', 'ollama', 'sglang'], formats: ['pytorch', 'safetensors'] },
  ];

  // 嵌入模型后端尚未实现，暂不展示
  const BUILTIN_EMBED = [];

  let state = {
    tab: 'llm',
    selectedModel: null,
    running: [],
    logAutoRefreshTimer: null,
  };

  var loadRunningAbortController = null;

  // AbortController polyfill for older browsers (Safari < 15.4, etc.)
  if (typeof AbortController === 'undefined') {
    window.AbortController = function () {
      this.signal = { aborted: false };
      this.abort = function () { this.signal.aborted = true; };
    };
  }

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
        const tags = sizesLabel ? sizesLabel + ' · generate model' : 'generate model';
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
    const form = $('#config-form');
    if (form) {
      delete form.dataset.modelId;
      delete form.dataset.modelName;
    }
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

  function _toggleParamVisibility(form, paramName, visible) {
    var group = form.querySelector('[data-param="' + paramName + '"]');
    if (!group) return;
    if (visible) {
      group.classList.remove('hidden');
    } else {
      group.classList.add('hidden');
    }
  }

  function fillConfigOptionsFromModel(model) {
    const form = $('#config-form');
    if (!form) return;
    var engineSel = form.querySelector('#config-engine') || form.querySelector('[name="engine"]');
    var formatSel = form.querySelector('#config-format') || form.querySelector('[name="format"]');
    var sizeSel = form.querySelector('#config-size') || form.querySelector('[name="size"]');
    var quantSel = form.querySelector('#config-quantization') || form.querySelector('[name="quantization"]');
    var engines = (model.engines && model.engines.length) ? model.engines : ['vllm', 'ollama', 'sglang'];
    var formats = (model.formats && model.formats.length) ? model.formats : ['pytorch', 'safetensors'];
    var sizes = (model.sizes && model.sizes.length) ? model.sizes : [{ size: '1B' }];
    var quants = (model.quantizations && model.quantizations.length) ? model.quantizations : ['none'];
    var qRepos = model.quantization_repos || {};
    var engineLabels = { ollama: 'Ollama', vllm: 'vLLM', sglang: 'SGLang' };
    var formatLabels = { pytorch: 'PyTorch', safetensors: 'SafeTensors' };
    var quantLabels = { none: '无', int4: 'INT4', int8: 'INT8' };

    if (engineSel) {
      engineSel.innerHTML = engines.map(function (v) { return '<option value="' + v + '">' + (engineLabels[v] || v) + '</option>'; }).join('');
      engineSel.value = engines[0];
    }
    // 模型引擎始终显示
    _toggleParamVisibility(form, 'engine', true);

    if (formatSel) {
      formatSel.innerHTML = formats.map(function (v) { return '<option value="' + v + '">' + (formatLabels[v] || v) + '</option>'; }).join('');
      formatSel.value = formats[0];
    }
    // 模型格式始终显示
    _toggleParamVisibility(form, 'format', true);

    if (sizeSel) {
      sizeSel.innerHTML = sizes.map(function (s) {
        var sizeVal = typeof s === 'string' ? s : (s.size || s.hf_repo || '');
        sizeVal = String(sizeVal || '');
        return '<option value="' + escapeHtml(sizeVal) + '">' + escapeHtml(sizeVal) + '</option>';
      }).join('');
      var firstSize = sizes[0];
      sizeSel.value = firstSize ? (typeof firstSize === 'string' ? firstSize : (firstSize.size || '1B')) : '1B';
    }
    // 模型大小始终显示（即使只有一个选项，也保留该参数供后续扩展）
    _toggleParamVisibility(form, 'size', true);

    if (quantSel) {
      quantSel.innerHTML = quants.map(function (v) {
        var label = quantLabels[v] || v;
        // 若声明了量化但 quantization_repos 中未配置对应仓库，标记为禁用
        if (v !== 'none' && !qRepos[v]) {
          label += '（未配置）';
          return '<option value="' + v + '" disabled title="该模型的 ' + v + ' 量化版本尚未在 models.json 中配置 hf_repo">' + label + '</option>';
        }
        return '<option value="' + v + '">' + label + '</option>';
      }).join('');
      quantSel.value = quants[0];
    }
    // 量化始终显示（即使只有 none，也保留该参数供后续扩展）
    _toggleParamVisibility(form, 'quantization', true);

    form.gpu_count.value = 'auto';
    form.replicas.value = '1';
    form.thought_mode.checked = true;
    form.parse_inference.checked = false;
    if (form.extra) form.extra.value = '';
  }

  function resetFormToDefault() {
    const form = $('#config-form');
    if (!form) return;
    form.engine.value = 'vllm';
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
          if (address && typeof address === 'string' && address.startsWith('http')) {
            var confirmed = confirm('模型已启动，是否在新标签页打开服务地址？\n' + address);
            if (confirmed) {
              window.open(address, '_blank');
            }
          }
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
        .then(function (data) {
          if (data && data.code === 200) {
            state.running = state.running.filter(function (r) {
              return r.id !== id;
            });
            renderRunningTable();
            loadLogsFromBackend();
          } else {
            alert('停止失败: ' + (data.msg || '未知错误'));
          }
        })
        .catch(function (err) {
          alert('停止请求失败: ' + (err.message || String(err)));
        });
    } else {
      state.running = state.running.filter(function (r) {
        return r.id !== id;
      });
      renderRunningTable();
    }
  }

  function copyToClipboard(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).catch(function () {});
    } else {
      var ta = document.createElement('textarea');
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
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
        (r) => {
          var addrDisplay = r.address && r.address.startsWith('http')
            ? '<a href="' + escapeHtml(r.address) + '" target="_blank" rel="noopener noreferrer">' + escapeHtml(r.address) + '</a>'
            : escapeHtml(r.address);
          var modelForTest = r.modelId || '';
          if (r.engine === 'ollama' && r.name && r.name.includes('qwen')) {
            modelForTest = 'qwen2:0.5b';
          } else if (r.engine === 'ollama') {
            modelForTest = r.name || r.modelId;
          }
          var testHref = '/test?address=' + encodeURIComponent(r.address) + '&engine=' + encodeURIComponent(r.engine) + '&model=' + encodeURIComponent(modelForTest);
          return `<tr>
            <td class="run-id-cell">${r.id.length > 12 ? r.id.slice(0, 8) + '…' : r.id}</td>
            <td>${escapeHtml(r.name)}</td>
            <td>
              ${addrDisplay}
              <button type="button" class="btn btn-sm btn-copy" data-copy-address="${escapeHtml(r.address)}" title="复制地址">复制</button>
              <a href="${testHref}" target="_blank" class="btn btn-sm btn-test">测试</a>
            </td>
            <td>${escapeHtml(r.engine)}</td>
            <td>${escapeHtml(r.gpuIndex != null ? r.gpuIndex : 'auto')}</td>
            <td>${escapeHtml(r.quantization != null ? r.quantization : '-')}</td>
            <td>${escapeHtml(r.size != null ? r.size : '-')}</td>
            <td>${escapeHtml(String(r.replicas != null ? r.replicas : 1))}</td>
            <td class="actions-cell">
              <button type="button" class="btn btn-sm btn-danger" data-stop-id="${r.id}">停止</button>
            </td>
          </tr>`;
        }
      )
      .join('');

    tbody.querySelectorAll('[data-stop-id]').forEach((btn) => {
      btn.addEventListener('click', () => stopRunning(btn.dataset.stopId));
    });
    tbody.querySelectorAll('.btn-copy').forEach((btn) => {
      btn.addEventListener('click', function () {
        copyToClipboard(btn.dataset.copyAddress);
        btn.textContent = '已复制';
        setTimeout(function () { btn.textContent = '复制'; }, 1500);
      });
    });
  }

  function setTab(tab) {
    state.tab = tab;
    $$('.tabs .tab').forEach((t) => t.classList.toggle('active', t.dataset.tab === tab));
    renderModelCards();
    renderRunningTable();
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
              quantization_repos: m.quantization_repos || {},
              engines: m.engines || ['vllm', 'ollama', 'sglang'],
              formats: m.formats || ['pytorch', 'safetensors'],
            };
          });
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
        var backendIds = new Set();
        var merged = data.data.running.map(mapBackendToRecord);
        merged.forEach(function (r) { backendIds.add(r.id); });
        // 保留前端本地已添加但后端尚未返回的实例（避免启动过程中闪烁消失）
        // 仅保留最近 12 秒内前端本地添加的实例，防止已停止实例永久残留
        var now = Date.now();
        state.running.forEach(function (r) {
          if (!backendIds.has(r.id)) {
            if (r.addedAt && (now - r.addedAt < 12000)) {
              merged.push(r);
            }
          }
        });
        state.running = merged;
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
    // 若嵌入模型未配置，隐藏嵌入相关 Tab
    if (!BUILTIN_EMBED || BUILTIN_EMBED.length === 0) {
      $$('.tabs .tab[data-tab="embed"]').forEach(function (el) { el.style.display = 'none'; });
    }
    renderModelCards();
    renderRunningTable();
    loadModelsFromBackend();
    loadRunningFromBackend();
    loadLogsFromBackend();

    $$('.tabs .tab').forEach((t) => {
      t.addEventListener('click', () => setTab(t.dataset.tab));
    });

    var btnRefreshRunning = $('#btn-refresh-running');
    if (btnRefreshRunning) btnRefreshRunning.addEventListener('click', function () {
      loadRunningFromBackend();
    });

    var btnRefreshLogs = $('#btn-refresh-logs');
    if (btnRefreshLogs) btnRefreshLogs.addEventListener('click', function () {
      loadLogsFromBackend();
    });

    var logAutoRefresh = $('#log-auto-refresh');
    if (logAutoRefresh) logAutoRefresh.addEventListener('change', function () {
      if (state.logAutoRefreshTimer) {
        clearInterval(state.logAutoRefreshTimer);
        state.logAutoRefreshTimer = null;
      }
      if (this.checked) {
        state.logAutoRefreshTimer = setInterval(loadLogsFromBackend, 3000);
      }
    });
    var configForm = $('#config-form');
    if (configForm) configForm.addEventListener('submit', onLaunch);
    var btnCancel = $('#btn-cancel');
    if (btnCancel) btnCancel.addEventListener('click', closeConfigPanel);


  }

  init();
})();
