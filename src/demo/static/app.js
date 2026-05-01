
        let currentJobId = null;
        let eventSource = null;
        
        // Preset queries (loaded from backend); index in select value
        let presetQueriesList = [];
        
        // Feature flags (declared early so all handlers see them; not related to fetch/network).
        const SHOW_CARD2TAB2CARD_MODEL_TABLES = false;
        const ENABLE_POST_INTEGRATION_ANALYSIS = false;
        const DEFAULT_QUERY_FALLBACK = 'Are there table foundation models that can handle small tables (≤100 rows/columns) with many missing values and produce column embeddings?';
        const INT_TITLE_C2C_HTML = '<span class="number-badge">1</span> Query2Card Results';
        const INT_TITLE_C2T2C_HTML = '<span class="number-badge">2</span> Query2Tab2Card Results';
        const INT_MODEL_IDS_C2C = 'Model IDs (1 Query2Card Results)';
        const INT_MODEL_IDS_C2T2C = 'Model IDs (2 Query2Tab2Card Results)';
        const INTEGRATION_TABLE_VIEWPORT_STYLE = 'height: 480px; width: 100%; max-width: 100%; overflow-x: auto; overflow-y: auto; border: 1px solid #dee2e6; border-radius: 6px; background: #fff;';
        const DISPLAY_MAX_ROWS = 50;
        const DISPLAY_MAX_COLS = 20;
        const INTEGRATION_MAX_CELL_CHARS = 120;
        window.__integrationTables = window.__integrationTables || {};
        window.__modelSearchRuns = window.__modelSearchRuns || [];
        window.__tableSearchRuns = window.__tableSearchRuns || [];
        
        function formatFetchError(err) {
            const m = err && err.message ? String(err.message) : 'Unknown error';
            if (m === 'Failed to fetch' || m === 'Load failed' || m.includes('NetworkError')) {
                return 'Cannot reach the API. Start the backend on port 5002 (e.g. python -m src.demo.backend in another terminal). The UI on port 5001 only serves the page; API requests go to port 5002.';
            }
            return m;
        }
        
        function escapeHtmlIntegration(s) {
            return String(s == null ? '' : s)
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;');
        }
        
        function integrationTablePageHref(fullPath) {
            const p = String(fullPath || '').trim();
            if (!p) return '#';
            return '{{BACKEND_URL}}/api/table-page?path=' + encodeURIComponent(p);
        }

        function integrationReviewPageHref(jobId) {
            const j = String(jobId || currentJobId || '').trim();
            if (!j) return '#';
            return '{{BACKEND_URL}}/api/integration-review-page/' + encodeURIComponent(j);
        }

        function evaluationPageHref(jobId) {
            const j = String(jobId || currentJobId || '').trim();
            if (!j) return '#';
            return '{{BACKEND_URL}}/api/evaluation-page/' + encodeURIComponent(j);
        }

        function normalizeNuggetMethod(method) {
            return String(method || '').trim().toLowerCase();
        }

        function buildNuggetMethodMap(summary) {
            const out = {};
            const methods = Array.isArray(summary && summary.methods) ? summary.methods : [];
            methods.forEach(row => {
                const key = normalizeNuggetMethod(row && row.method);
                if (key) out[key] = row;
            });
            return out;
        }

        function nuggetScoreInlineInner(row) {
            if (!row) {
                return 'Score*: <span style="color:#999;">-</span>';
            }
            return `Score*: <strong style="color:#b54708;">${Number(row.filter_dedup || 0)}</strong>`;
        }

        function applyNuggetScoresToRetrievalCards(summary) {
            const scoreByMethod = buildNuggetMethodMap(summary);
            document.querySelectorAll('[data-nugget-score-inline-method]').forEach(el => {
                const row = scoreByMethod[normalizeNuggetMethod(el.getAttribute('data-nugget-score-inline-method'))];
                el.innerHTML = nuggetScoreInlineInner(row || null);
            });
        }

        function describeTableSearchCount(run, stats) {
            const queryTableCount = Array.isArray(run && run.query_tables) ? run.query_tables.length : 0;
            const baseTablesCount = (stats && stats.total_unique_tables != null)
                ? stats.total_unique_tables
                : ((run && Array.isArray(run.table_paths)) ? run.table_paths.length : 0);
            const integrationInputCount = Array.isArray(run && run.integration_input_table_paths) && run.integration_input_table_paths.length
                ? run.integration_input_table_paths.length
                : (baseTablesCount + queryTableCount);
            const tablesSource = String((run && run.tables_source) || (stats && stats.tables_source) || 'intermediate');
            const baseLabel = tablesSource === 'all_from_modelcards' ? 'related tables' : 'retrieved tables';
            if (queryTableCount > 0) {
                return `${integrationInputCount} input tables (${queryTableCount} query table + ${baseTablesCount} ${baseLabel})`;
            }
            return `${baseTablesCount} ${baseLabel}`;
        }

        function getGroupedIntegratedTables(run) {
            if (!run || !Array.isArray(run.grouped_integrated_tables)) return [];
            return run.grouped_integrated_tables.filter(group => group && group.integrated_table && Array.isArray(group.integrated_table.columns));
        }
        
        /** Open full HTML table view; label defaults to file basename. */
        function integrationTablePathLink(fullPath, displayLabel) {
            const p = String(fullPath || '').trim();
            if (!p) return '';
            const label = displayLabel != null ? String(displayLabel) : (p.split('/').pop() || p);
            const href = integrationTablePageHref(p);
            return `<a href="${href}" target="_blank" rel="noopener noreferrer" style="color:#0056b3;text-decoration:none;">${escapeHtmlIntegration(label)}</a>`;
        }
        
        function integrationTablePathLinksRow(paths, joinHtml) {
            const list = (paths || []).map(p => String(p).trim()).filter(Boolean);
            if (!list.length) return '';
            const j = joinHtml != null ? joinHtml : ', ';
            return list.map(p => integrationTablePathLink(p)).join(j);
        }
        
        function integrationSavedPathLink(fullPath) {
            const p = String(fullPath || '').trim();
            if (!p) return '';
            const href = integrationTablePageHref(p);
            return `<a href="${href}" target="_blank" rel="noopener noreferrer" style="color:#0056b3;text-decoration:none;"><code style="background: #f1f3f5; padding: 2px 6px; border-radius: 4px; font-size: 11px;">${escapeHtmlIntegration(p)}</code></a>`;
        }

        function renderGroupedIntegrationTables(groups, options) {
            const opts = options || {};
            const title = opts.title || 'Integration';
            const successColor = opts.successColor || '#28a745';
            const extraHtml = opts.extraHtml || '';
            const downloadPrefix = opts.downloadPrefix || 'grouped-table';
            const validGroups = (groups || []).filter(group => group && group.integrated_table);
            if (!validGroups.length) {
                return `<div style="padding: 15px; background: #fff; border-radius: 4px; border: 1px solid #dee2e6;">${extraHtml}<div style="color:#6c757d;">No grouped integration tables.</div></div>`;
            }
            const sections = validGroups.map((group, idx) => {
                const queryPath = String(group.query_table_path || '').trim();
                const retrievedPaths = Array.isArray(group.retrieved_table_paths) ? group.retrieved_table_paths : [];
                const savedPath = String(group.saved_path || '').trim();
                const groupLabel = queryPath
                    ? `Group ${idx + 1}: ${integrationBasename(queryPath)}`
                    : `Group ${idx + 1}`;
                const headerLines = [
                    `<div style="font-weight:600;color:${successColor};margin-bottom:4px;">${escapeHtmlIntegration(groupLabel)}</div>`,
                ];
                if (queryPath) {
                    headerLines.push(`<div style="font-size:12px;color:#495057;margin-bottom:4px;"><strong>Query table:</strong> ${integrationTablePathLink(queryPath, integrationBasename(queryPath))}</div>`);
                }
                headerLines.push(`<div style="font-size:12px;color:#495057;margin-bottom:8px;"><strong>Retrieved tables:</strong> ${retrievedPaths.length ? integrationTablePathLinksRow(retrievedPaths, ', ') : '<span style="color:#999;">—</span>'}</div>`);
                return `<div style="margin-top:${idx === 0 ? '0' : '14px'};">${renderIntegrationTable(group.integrated_table, group.integrated_table.stats || {}, {
                    title,
                    successColor,
                    extraHtml: `<div style="margin-bottom:10px;padding:8px;background:#f8f9fa;border-radius:4px;">${headerLines.join('')}</div>`,
                    savedPath,
                    downloadId: `${downloadPrefix}-${idx + 1}`,
                    omitSectionTitle: idx > 0,
                })}</div>`;
            }).join('');
            return `${extraHtml}${sections}`;
        }
        
        function integrationBasename(p) {
            const s = String(p == null ? '' : p).trim();
            if (!s) return '';
            const norm = s.replace(/\\/g, '/');
            const i = norm.lastIndexOf('/');
            return i >= 0 ? norm.slice(i + 1) : norm;
        }
        
        function buildRetrievedTableModelRowsFallback(modelToTables, tablePathsList) {
            const rev = {};
            Object.entries(modelToTables || {}).forEach(([mid, paths]) => {
                (paths || []).forEach(p => {
                    const b = integrationBasename(p);
                    if (!b) return;
                    if (!rev[b]) rev[b] = [];
                    const sm = String(mid);
                    if (!rev[b].includes(sm)) rev[b].push(sm);
                });
            });
            return (tablePathsList || []).map(tp => {
                const tpS = String(tp);
                return {
                    table: integrationBasename(tpS),
                    table_path: tpS,
                    models: rev[integrationBasename(tpS)] || []
                };
            });
        }
        
        function formatQuery2CardModelTableLinesHtml(modelIds, modelToTables) {
            const ids = Array.isArray(modelIds) ? modelIds : [];
            if (!ids.length) return '';
            return ids.map(m => {
                const mid = String(m).trim();
                if (!mid) return '';
                const paths = (modelToTables && modelToTables[mid]) ? modelToTables[mid] : [];
                const link = `<a href="https://huggingface.co/${mid}" target="_blank" rel="noopener noreferrer" style="color:#0056b3;text-decoration:none;">${escapeHtmlIntegration(mid)}</a>`;
                const tbl = paths.length
                    ? integrationTablePathLinksRow(paths, ', ')
                    : '<span style="color:#999;">—</span>';
                return `<div style="margin-top:4px;line-height:1.4;">Model id: ${link} → Related table: ${tbl}</div>`;
            }).join('');
        }
        
        function formatQuery2Tab2CardTraceHtml(queryTables, retrievedRows, options) {
            const opts = options || {};
            const tablesSource = String(opts.tablesSource || 'intermediate');
            const modelToTablePaths = opts.modelToTablePaths || {};
            const rankedModelIds = Array.isArray(opts.rankedModelIds) ? opts.rankedModelIds.map(String) : [];
            const pipelineTrace = opts.pipelineTrace && typeof opts.pipelineTrace === 'object' ? opts.pipelineTrace : null;
            const tab2tabTraceRows = Array.isArray(opts.tab2tabTraceRows) ? opts.tab2tabTraceRows : [];
            const afterModelCapTraceRows = Array.isArray(opts.afterModelCapTraceRows) ? opts.afterModelCapTraceRows : [];

            const qts = Array.isArray(queryTables) ? queryTables : [];
            const qLine = qts.length
                ? `<div style="margin-bottom:6px;line-height:1.4;"><strong>Query table(s):</strong> ${qts.map(q => {
                    const full = String(q).trim();
                    const bn = integrationBasename(full);
                    return integrationTablePathLink(full || bn, bn || full);
                }).filter(Boolean).join(', ')}</div>`
                : '';

            function modelIdLinks(ids) {
                const arr = Array.isArray(ids) ? ids : [];
                if (!arr.length) return '<span style="color:#999;">—</span>';
                return arr.map(mid => {
                    const s = String(mid).trim();
                    if (!s) return '';
                    return `<a href="https://huggingface.co/${s}" target="_blank" rel="noopener noreferrer" style="color:#0056b3;text-decoration:none;">${escapeHtmlIntegration(s)}</a>`;
                }).filter(Boolean).join(', ');
            }

            function renderTableToModelRows(rows) {
                const arr = Array.isArray(rows) ? rows : [];
                return arr.map((row, i) => {
                    const tPath = row.table_path || row.table || '';
                    const tLabel = row.table || integrationBasename(tPath);
                    const tableFrag = tPath
                        ? integrationTablePathLink(tPath, tLabel)
                        : (tLabel ? `<code>${escapeHtmlIntegration(tLabel)}</code>` : '<span style="color:#999;">—</span>');
                    const models = Array.isArray(row.models) ? row.models : [];
                    const label = arr.length > 1 ? `Retrieved table ${i + 1}:` : 'Retrieved table:';
                    const modelLinks = models.length ? modelIdLinks(models) : '<span style="color:#999;">—</span>';
                    return `<div style="margin-top:4px;line-height:1.4;"><strong>${label}</strong> ${tableFrag} <span style="color:#666;">→ related models:</span> ${modelLinks}</div>`;
                }).join('');
            }

            const useAllFromCardsPipeline = tablesSource === 'all_from_modelcards' && rankedModelIds.length > 0;

            if (useAllFromCardsPipeline) {
                let html = qLine;
                const searchedRows = tab2tabTraceRows.length
                    ? tab2tabTraceRows
                    : (afterModelCapTraceRows.length ? afterModelCapTraceRows : (Array.isArray(retrievedRows) ? retrievedRows : []));
                if (searchedRows.length) {
                    html += renderTableToModelRows(searchedRows);
                }
                const post = pipelineTrace && Array.isArray(pipelineTrace.model_ids_after_dense_rerank)
                    ? pipelineTrace.model_ids_after_dense_rerank
                    : rankedModelIds;
                if (post.length) {
                    html += `<div style="margin-top:10px;line-height:1.35;"><strong>Reranked models -> related tables</strong></div>`;
                }
                html += post.map((mid, idx) => {
                    const s = String(mid).trim();
                    if (!s) return '';
                    const paths = modelToTablePaths[s] || [];
                    const tbls = paths.length ? integrationTablePathLinksRow(paths, ', ') : '<span style="color:#999;">—</span>';
                    const link = `<a href="https://huggingface.co/${s}" target="_blank" rel="noopener noreferrer" style="color:#0056b3;text-decoration:none;">${escapeHtmlIntegration(s)}</a>`;
                    return `<div style="margin:6px 0 2px 4px;padding-left:8px;border-left:3px solid #2e7d32;line-height:1.35;"><strong>${idx + 1}.</strong> ${link} <span style="color:#666;">→ related tables:</span> ${tbls}</div>`;
                }).join('');
                return html;
            }

            const bridgeIntermediate = afterModelCapTraceRows.length ? afterModelCapTraceRows : (Array.isArray(retrievedRows) ? retrievedRows : []);
            return qLine + renderTableToModelRows(bridgeIntermediate);
        }
        
        async function loadPresetQueries() {
            const sel = document.getElementById('preset_query_select');
            const queryInput = document.getElementById('query');
            if (!sel) return;
            try {
                const response = await fetch('{{BACKEND_URL}}/api/preset-queries');
                const data = await response.json();
                if (data.status === 'success') {
                    presetQueriesList = data.queries;
                    sel.innerHTML = '<option value="">— select preset —</option>';
                    (data.queries || []).forEach(function(q, i) {
                        const opt = document.createElement('option');
                        opt.value = String(i);
                        opt.textContent = q.title || q.id || ('Query ' + (i + 1));
                        sel.appendChild(opt);
                    });
                    if ((data.queries || []).length > 0) {
                        sel.value = '0';
                        if (queryInput) queryInput.value = (data.queries[0].query || '').trim();
                    } else if (queryInput) {
                        if (!String(queryInput.value || '').trim()) queryInput.value = DEFAULT_QUERY_FALLBACK;
                        sel.innerHTML = '<option value="">— no preset found —</option>';
                    }
                }
            } catch (e) {
                console.warn('Preset queries load failed:', e);
                if (queryInput && !String(queryInput.value || '').trim()) queryInput.value = DEFAULT_QUERY_FALLBACK;
            }
        }
        
        function onPresetQueryChange() {
            const sel = document.getElementById('preset_query_select');
            const queryInput = document.getElementById('query');
            if (!sel || !queryInput || sel.value === '') return;
            const idx = parseInt(sel.value, 10);
            if (idx >= 0 && idx < presetQueriesList.length) {
                queryInput.value = presetQueriesList[idx].query || '';
            }
        }
        
        // Initialize on page load
        window.addEventListener('DOMContentLoaded', function() {
            const loadPreviousCheckbox = document.getElementById('load_previous_search');
            if (loadPreviousCheckbox) {
                loadPreviousCheckbox.checked = false;
                loadPreviousCheckbox.addEventListener('change', toggleLoadPrevious);
            }
            toggleLoadPrevious();
            
            const diagramImg = document.getElementById('search-diagram');
            if (diagramImg) diagramImg.src = '/static/docs/modelsearch_wquery.png';

            const tableTopK = document.getElementById('table_search_k');
            if (tableTopK) {
                tableTopK.addEventListener('change', () => clampTopKInput('table_search_k'));
                tableTopK.addEventListener('blur', () => clampTopKInput('table_search_k'));
            }
            const modelTopK = document.getElementById('model_top_k');
            if (modelTopK) {
                modelTopK.addEventListener('change', () => clampTopKInput('model_top_k'));
                modelTopK.addEventListener('blur', () => clampTopKInput('model_top_k'));
            }
            
            loadPresetQueries();
        });

        async function fetchEvaluationSummary(jobId) {
            const j = String(jobId || currentJobId || '').trim();
            if (!j) return null;
            try {
                const resp = await fetch('{{BACKEND_URL}}/api/evaluation-summary/' + encodeURIComponent(j));
                const data = await resp.json();
                return data;
            } catch (e) {
                return { status: 'error', message: e && e.message ? e.message : String(e) };
            }
        }

        async function refreshEvaluationSummary(jobId, mountId) {
            const mount = document.getElementById(mountId || 'evaluationSummaryMount');
            if (!mount) return;
            const j = String(jobId || currentJobId || '').trim();
            if (!j) {
                mount.innerHTML = '<span style="font-size:12px;color:#888;">Evaluation summary will appear here when available.</span>';
                return;
            }
            mount.innerHTML = '<div class="evaluation-summary-panel"><span style="font-size:12px;color:#888;">Loading evaluation summary…</span></div>';
            const data = await fetchEvaluationSummary(j);
            if (!data || data.status !== 'success' || !data.available) {
                mount.innerHTML = '<div class="evaluation-summary-panel"><span style="font-size:12px;color:#888;">No evaluation summary found for this job yet.</span></div>';
                return;
            }
            applyNuggetScoresToRetrievalCards(data);
            const reportLink = data.markdown_path
                ? `<a href="${evaluationPageHref(j)}" target="_blank" rel="noopener noreferrer" style="font-size:12px;color:#0056b3;text-decoration:none;">review full nugget extraction report</a>`
                : '<span style="font-size:12px;color:#888;">Report not available yet.</span>';
            mount.innerHTML = `
                <div class="evaluation-summary-panel">
                    <div style="font-size:12px;color:#57606a;line-height:1.45;margin-bottom:8px;">
                        Score: evaluation is nugget-based.
                        <span style="margin-left:8px;">${reportLink}</span>
                    </div>
                    <div class="pdf-section pipeline-diagram-frame">
                        <img class="pipeline-diagram-img" src="/static/docs/evaluation.png" alt="Nugget-based evaluation" />
                    </div>
                </div>
            `;
        }

        async function fetchEvaluationRunStatus(jobId) {
            const j = String(jobId || currentJobId || '').trim();
            if (!j) return null;
            try {
                const resp = await fetch('{{BACKEND_URL}}/api/evaluation-run/' + encodeURIComponent(j));
                return await resp.json();
            } catch (e) {
                return { status: 'error', message: e && e.message ? e.message : String(e) };
            }
        }

        async function pollEvaluationRunUntilDone(jobId, mountId) {
            const mount = document.getElementById(mountId || 'retrievalEvaluationSummaryMount');
            const j = String(jobId || currentJobId || '').trim();
            if (!j) return;
            for (let i = 0; i < 120; i++) {
                const data = await fetchEvaluationRunStatus(j);
                if (!data || data.status !== 'success') {
                    if (mount) mount.innerHTML = '<div class="evaluation-summary-panel"><span style="font-size:12px;color:#888;">Evaluation status check failed.</span></div>';
                    return;
                }
                const runStatus = String(data.run_status || '');
                if (runStatus === 'running') {
                    if (mount) mount.innerHTML = '<div class="evaluation-summary-panel"><span style="font-size:12px;color:#888;">Running nugget-based evaluation automatically...</span></div>';
                    await new Promise(r => setTimeout(r, 2000));
                    continue;
                }
                if (runStatus === 'completed') {
                    await refreshEvaluationSummary(j, mountId || 'retrievalEvaluationSummaryMount');
                    return;
                }
                if (runStatus === 'failed') {
                    if (mount) mount.innerHTML = `<div class="evaluation-summary-panel"><span style="font-size:12px;color:#888;">${escapeHtmlIntegration(data.message || 'Evaluation failed.')}</span></div>`;
                    return;
                }
                if (mount) mount.innerHTML = '<div class="evaluation-summary-panel"><span style="font-size:12px;color:#888;">Evaluation not started.</span></div>';
                return;
            }
            if (mount) mount.innerHTML = '<div class="evaluation-summary-panel"><span style="font-size:12px;color:#888;">Evaluation still running. Please check again shortly.</span></div>';
        }

        async function ensureEvaluationForResults(jobId, mountId) {
            const mount = document.getElementById(mountId || 'retrievalEvaluationSummaryMount');
            const j = String(jobId || currentJobId || '').trim();
            if (!mount || !j) return;

            const summary = await fetchEvaluationSummary(j);
            if (summary && summary.status === 'success' && summary.available) {
                await refreshEvaluationSummary(j, mountId || 'retrievalEvaluationSummaryMount');
                return;
            }

            mount.innerHTML = '<div class="evaluation-summary-panel"><span style="font-size:12px;color:#888;">Running nugget-based evaluation automatically...</span></div>';
            const runState = await fetchEvaluationRunStatus(j);
            const runStatus = String((runState && runState.run_status) || '');
            if (runStatus === 'running') {
                await pollEvaluationRunUntilDone(j, mountId || 'retrievalEvaluationSummaryMount');
                return;
            }
            if (runStatus === 'completed') {
                await refreshEvaluationSummary(j, mountId || 'retrievalEvaluationSummaryMount');
                return;
            }

            try {
                const resp = await fetch('{{BACKEND_URL}}/api/evaluation-run', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ job_id: j, llm_mode: 'iter' }),
                });
                const data = await resp.json();
                if (!data || data.status !== 'success') {
                    mount.innerHTML = '<div class="evaluation-summary-panel"><span style="font-size:12px;color:#888;">Failed to start automatic evaluation.</span></div>';
                    return;
                }
                await pollEvaluationRunUntilDone(j, mountId || 'retrievalEvaluationSummaryMount');
            } catch (e) {
                mount.innerHTML = `<div class="evaluation-summary-panel"><span style="font-size:12px;color:#888;">Automatic evaluation failed: ${escapeHtmlIntegration(formatFetchError(e))}</span></div>`;
            }
        }
        
        function updateTopKValue(value) {
            const num = document.getElementById('top_k');
            if (num) num.value = value;
            updateTableSearchKDefault();
        }
        function updateTopKSlider(value) {
            const slider = document.getElementById('top_k_slider');
            const num = document.getElementById('top_k');
            const v = parseInt(value, 10);
            if (slider && num && v >= 1 && v <= 100) {
                slider.value = v;
                num.value = v;
            }
            updateTableSearchKDefault();
        }
        function updateTableSearchKValue(value) {
            const num = document.getElementById('table_search_k');
            const slider = document.getElementById('table_search_k_slider');
            if (num) num.value = value;
            if (slider) slider.value = value;
        }
        function updateModelTopKValue(value) {
            const num = document.getElementById('model_top_k');
            const slider = document.getElementById('model_top_k_slider');
            if (num) num.value = value;
            if (slider) slider.value = value;
        }
        function updateTableSearchKSlider(value) {
            const slider = document.getElementById('table_search_k_slider');
            const num = document.getElementById('table_search_k');
            const v = parseInt(value, 10);
            if (slider && num && v >= parseInt(slider.min) && v <= parseInt(slider.max)) {
                slider.value = v;
                num.value = v;
            }
        }
        function updateModelTopKSlider(value) {
            const slider = document.getElementById('model_top_k_slider');
            const num = document.getElementById('model_top_k');
            const v = parseInt(value, 10);
            if (slider && num && v >= parseInt(slider.min) && v <= parseInt(slider.max)) {
                slider.value = v;
                num.value = v;
            }
        }
        function clampTopKInput(id) {
            const el = document.getElementById(id);
            if (!el) return;
            const v = parseInt(el.value, 10);
            const next = Number.isFinite(v) ? Math.max(1, Math.min(5, v)) : 3;
            el.value = String(next);
        }
        function stepTopK(id, delta) {
            const el = document.getElementById(id);
            if (!el) return;
            const base = parseInt(el.value, 10);
            const cur = Number.isFinite(base) ? base : 3;
            const next = Math.max(1, Math.min(5, cur + (delta || 0)));
            el.value = String(next);
        }
        function updateTableSearchKDefault() {
            // Per-table k is independent; no sync with top_k
        }
        function updateIntegrationKValue(value) {
            const num = document.getElementById('integration_k');
            const slider = document.getElementById('integration_k_slider');
            if (num) num.value = value;
            if (slider) slider.value = value;
        }
        function updateIntegrationKSlider(value) {
            const slider = document.getElementById('integration_k_slider');
            const num = document.getElementById('integration_k');
            const v = parseInt(value, 10);
            if (slider && num && v >= 1 && v <= 50) { slider.value = v; num.value = v; }
        }
        function updateIntegrationMaxModelsValue(value) {
            const num = document.getElementById('integration_max_models');
            const slider = document.getElementById('integration_max_models_slider');
            if (num) num.value = value;
            if (slider) slider.value = value;
        }
        function updateIntegrationMaxModelsSlider(value) {
            const slider = document.getElementById('integration_max_models_slider');
            const num = document.getElementById('integration_max_models');
            const v = parseInt(value, 10);
            if (slider && num && v >= 1 && v <= 50) { slider.value = v; num.value = v; }
        }
        
        function toggleLoadPrevious() {
            const loadPrevious = document.getElementById('load_previous_search').checked;
            const previousSection = document.getElementById('previous-search-section');
            const newSearchInputs = document.getElementById('new-search-inputs');
            
            if (loadPrevious) {
                previousSection.style.display = 'block';
                newSearchInputs.style.display = 'none';
                loadSavedSearches();
            } else {
                previousSection.style.display = 'none';
                newSearchInputs.style.display = 'block';
            }
        }
        
        async function loadSavedSearches() {
            try {
                const response = await fetch('{{BACKEND_URL}}/api/saved-searches');
                const data = await response.json();
                
                if (data.status === 'success') {
                    const searches = data.searches || [];
                    const selectEl = document.getElementById('saved_search_select');
                    if (selectEl) {
                        selectEl.innerHTML = '<option value="">— select job —</option>';
                        searches.forEach(search => {
                            const opt = document.createElement('option');
                            opt.value = search.folder_name || search.id || '';
                            const qShort = search.query
                                ? (search.query.length > 96 ? search.query.substring(0, 96) + '...' : search.query)
                                : '';
                            const tail = search.query ? qShort : (search.model_id || search.folder_name || opt.value || '');
                            const label = [search.timestamp_str || '', tail].filter(Boolean).join(' ').trim() || opt.value;
                            opt.textContent = label;
                            selectEl.appendChild(opt);
                        });
                    }
                }
            } catch (error) {}
        }
        
        function clearResultsMetaStrip() {
            const el = document.getElementById('resultsMetaStrip');
            if (!el) return;
            el.style.display = 'none';
            el.innerHTML = '';
        }
        
        function clearIntegrationPanel() {
            const m = document.getElementById('integrationPanelMount');
            if (!m) return;
            m.innerHTML = '<span style="font-size:12px;color:#888;">Table Integration will appear here after a search completes.</span>';
        }

        async function fetchJson(url, options) {
            const response = await fetch(url, options);
            return await response.json();
        }

        function rowsToTableToModels(rows) {
            const tableToModels = {};
            (Array.isArray(rows) ? rows : []).forEach(row => {
                const tablePath = String((row && (row.table_path || row.table)) || '').trim();
                if (!tablePath) return;
                tableToModels[tablePath] = (Array.isArray(row.models) ? row.models : []).map(String);
            });
            return tableToModels;
        }

        async function loadSearchDisplayData(jobId) {
            const baseData = await fetchJson(`{{BACKEND_URL}}/api/results/${jobId}`);
            if (baseData.status !== 'success') return baseData;
            const meta = baseData.results || {};
            const retrievalModes = ['dense', 'sparse', 'hybrid'];
            const tableSearchTypes = ['unionable', 'single_column', 'keyword'];
            const modePreviews = await Promise.all(retrievalModes.map(async mode => {
                const preview = await fetchJson('{{BACKEND_URL}}/api/model-search-preview', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ job_id: jobId, query2modelcard_retrieval_mode: mode, max_models: meta.model_top_k || 5 })
                });
                return [mode, preview];
            }));
            const tablePreviews = await Promise.all(tableSearchTypes.map(async searchType => {
                const preview = await fetchJson('{{BACKEND_URL}}/api/table-search-preview', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ job_id: jobId, search_type: searchType })
                });
                return [searchType, preview];
            }));
            const query2modelcardAllModes = Object.fromEntries(
                modePreviews.map(([mode, preview]) => [mode, preview.status === 'success' ? (preview.model_ids || []) : []])
            );
            const card2tab2cardResults = Object.fromEntries(
                tablePreviews
                    .filter(([, preview]) => preview.status === 'success')
                    .map(([searchType, preview]) => [
                        searchType,
                        {
                            model_ids: preview.model_ids || preview.models_with_tables || [],
                            models_with_tables: preview.models_with_tables || [],
                            model_to_table_paths: preview.model_to_table_paths || {},
                            query_tables: preview.query_tables || [],
                            searched_tables: (preview.retrieved_table_model_rows || []).map(row => row.table_path || row.table).filter(Boolean),
                            intermediate: {
                                table_to_models: rowsToTableToModels(preview.retrieved_table_model_rows || []),
                                retrieved_table_filenames: (preview.retrieved_table_model_rows || []).map(row => row.table_path || row.table).filter(Boolean),
                            },
                            pipeline_trace: preview.pipeline_trace || {},
                            tab2tab_trace_rows: preview.tab2tab_trace_rows || [],
                            after_model_cap_trace_rows: preview.after_model_cap_trace_rows || [],
                            retrieved_table_model_rows: preview.retrieved_table_model_rows || [],
                            preview_meta: preview.preview_meta || {},
                            job_context: preview.job_context || {},
                            stats: preview.stats || {},
                        }
                    ])
            );
            const densePreview = Object.fromEntries(modePreviews).dense || {};
            const firstTablePreview = Object.values(card2tab2cardResults)[0] || {};
            return {
                ...meta,
                job_id: jobId,
                model_id: densePreview.job_context ? densePreview.job_context.table_search_seed_model_id : null,
                table_search_seed_model_id: firstTablePreview.query_seed_model_id || (firstTablePreview.job_context ? firstTablePreview.job_context.table_search_seed_model_id : null),
                query2modelcard_results: query2modelcardAllModes.dense || [],
                query2modelcard_all_modes: query2modelcardAllModes,
                card2tab2card_results: card2tab2cardResults,
            };
        }
        
        function setIntegrationDropdownsFromSaved(modelRes, tableRes) {
            const setSelect = (id, value) => {
                const el = document.getElementById(id);
                if (el && value != null && value !== '') el.value = value;
            };
            const setInput = (id, value) => {
                const el = document.getElementById(id);
                if (el && value != null && value !== '') el.value = value;
            };
            if (modelRes) {
                setSelect('integration_type', modelRes.integration_type);
                setSelect('integration_q2m_mode', modelRes.query2modelcard_retrieval_mode);
                setInput('integration_k', modelRes.k);
                setInput('integration_max_models', modelRes.max_models);
            }
            if (tableRes) {
                setSelect('integration_type', tableRes.integration_type);
                setSelect('integration_search_type', tableRes.search_type);
                setSelect('integration_tables_source', tableRes.tables_source);
                setInput('integration_k', tableRes.k);
                setInput('integration_max_models', tableRes.max_models);
            }
        }
        
        async function restoreIntegrationEvaluationQA(data) {
            const container = document.getElementById('integrationResultsContainer');
            const leftDiv = document.getElementById('integrationModelSearchResults');
            const rightDiv = document.getElementById('integrationResults');
            if (!container || !leftDiv || !rightDiv) return;
            const jobId = data.job_id || currentJobId;
            let modelRuns = data.model_search_runs || [];
            let tableRuns = data.table_search_runs || [];
            if (modelRuns.length === 0 && tableRuns.length === 0 && jobId) {
                try {
                    const r = await fetch('{{BACKEND_URL}}/api/integration-runs/' + jobId);
                    const apiData = await r.json();
                    if (apiData.status === 'success') {
                        modelRuns = apiData.model_search_runs || [];
                        tableRuns = apiData.table_search_runs || [];
                    }
                } catch (e) { /* ignore */ }
            }
            window.__modelSearchRuns = modelRuns;
            window.__tableSearchRuns = tableRuns;
            if (modelRuns.length > 0 || tableRuns.length > 0) {
                container.style.display = 'block';
                if (modelRuns.length > 0) setIntegrationDropdownsFromSaved(modelRuns[0], null);
                if (tableRuns.length > 0) setIntegrationDropdownsFromSaved(null, tableRuns[0]);
                if (tableRuns.length > 0) {
                    const t = tableRuns[0];
                    const key = getTableSearchKey(t.integration_type, t.search_type, t.tables_source);
                    const activeKey = getTableSearchKey(
                        (document.getElementById('integration_type') || {}).value || 'alite',
                        (document.getElementById('integration_search_type') || {}).value || 'single_column',
                        (document.getElementById('integration_tables_source') || {}).value || 'intermediate',
                    );
                    if (key !== activeKey) {
                        setIntegrationDropdownsFromSaved(null, t);
                    }
                }
                syncBothIntegrationDisplays();
            } else {
            const hasModel = data.integration_model_search && data.integration_model_search.status === 'success' && data.integration_model_search.integrated_table;
            const hasTable = data.integration_table_search && data.integration_table_search.status === 'success' && getGroupedIntegratedTables(data.integration_table_search).length > 0;
            if (hasModel || hasTable) {
                container.style.display = 'block';
                if (hasModel) {
                    const m = data.integration_model_search;
                    const key = getModelSearchKey(m.integration_type, m.query2modelcard_retrieval_mode);
                    window.__modelSearchRuns = [{ key, ...m }];
                }
                if (hasTable) {
                    const t = data.integration_table_search;
                    const key = getTableSearchKey(t.integration_type, t.search_type, t.tables_source);
                    window.__tableSearchRuns = [{ key, ...t }];
                }
                setIntegrationDropdownsFromSaved(data.integration_model_search, data.integration_table_search);
                syncBothIntegrationDisplays();
            }
            }
            if (data.evaluation_results && data.evaluation_results.evaluation) {
                const resultsDiv = document.getElementById('evaluationResults');
                if (resultsDiv) {
                    resultsDiv.style.display = 'block';
                    displayEvaluationResults(data.evaluation_results.evaluation, resultsDiv, data.evaluation_results.table1 || null, data.evaluation_results.table2 || null);
                }
            }
            if (data.qa_results) {
                const afterClick = document.getElementById('qa_after_click');
                const divTable = document.getElementById('qaResultsTableSearch');
                const divModel = document.getElementById('qaResultsModelSearch');
                if (afterClick) afterClick.style.display = 'block';
                if (data.qa_results.table_search && divTable) displayQAResults(data.qa_results.table_search.qa, data.qa_results.table_search.query || '', divTable);
                if (data.qa_results.model_search && divModel) displayQAResults(data.qa_results.model_search.qa, data.qa_results.model_search.query || '', divModel);
            }
        }
        
        async function loadMimicSearch() {
            const select = document.getElementById('saved_search_select');
            const folderName = select.value;
            
            if (!folderName) {
                showError('Please select a saved search folder');
                return;
            }
            
            const loadBtn = document.getElementById('loadMimicBtn');
            loadBtn.disabled = true;
            loadBtn.textContent = '⏳ Loading...';
            
            // Reset UI
            document.getElementById('progressSection').classList.add('active');
            document.getElementById('resultsSection').classList.remove('active');
            clearResultsMetaStrip();
            clearIntegrationPanel();
            document.getElementById('errorMsg').style.display = 'none';
            document.getElementById('logContainer').innerHTML = '';
            
            try {
                currentJobId = folderName;
                const results = await loadSearchDisplayData(folderName);
                if (results.status === 'success' || !results.status) {
                    displayResults(results);
                    await restoreIntegrationEvaluationQA({ job_id: folderName });
                    document.getElementById('progressSection').classList.remove('active');
                } else {
                    showError(results.message || 'Failed to load saved job');
                }
            } catch (error) {
                showError('Error: ' + error.message);
            } finally {
                loadBtn.disabled = false;
                loadBtn.textContent = '🔄 Load Saved Results';
            }
        }
        
        async function startSearch() {
            // Check if user wants to load previous search
            const loadPrevious = document.getElementById('load_previous_search').checked;
            
            if (loadPrevious) {
                showError('Please select a saved search.');
                return;
            }
            
            // Continue with new search logic
            
            const query = document.getElementById('query').value.trim();
            const topK = parseInt((document.getElementById('top_k') || {}).value, 10) || 100;  // Left aligns to right; high default
            const tableSearchKRaw = parseInt(document.getElementById('table_search_k').value, 10) || 1;
            const modelTopKRaw = parseInt((document.getElementById('model_top_k') || {}).value, 10) || 3;
            const tableSearchK = Math.min(5, Math.max(1, tableSearchKRaw));
            const modelTopK = Math.min(5, Math.max(1, modelTopKRaw));
            // Table retrieval: always run search (no load-from-JSON option)
            const tab2tabMode = 'search';
            // Validate input for current search flow
            if (!query) {
                showError('Please enter a query');
                return;
            }
            // Reset UI
            document.getElementById('searchBtn').disabled = true;
            document.getElementById('progressSection').classList.add('active');
            document.getElementById('resultsSection').classList.remove('active');
            clearResultsMetaStrip();
            clearIntegrationPanel();
            document.getElementById('errorMsg').style.display = 'none';
            document.getElementById('logContainer').innerHTML = '';
            
            try {
                // Start search
                const requestBody = {
                    top_k: topK,
                    table_search_k: tableSearchK,
                    model_top_k: modelTopK,
                    query: query,
                };
                
                const response = await fetch('{{BACKEND_URL}}/api/search', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(requestBody)
                });
                
                const data = await response.json();
                
                if (data.status === 'started') {
                    currentJobId = data.job_id;
                    startLogStreaming(currentJobId);
                    pollResults(currentJobId);
                } else {
                    showError(data.message || 'Failed to start search');
                    document.getElementById('searchBtn').disabled = false;
                }
            } catch (error) {
                showError('Error: ' + error.message);
                document.getElementById('searchBtn').disabled = false;
            }
        }
        window.startSearch = startSearch;
        
        function startLogStreaming(jobId) {
            if (eventSource) {
                eventSource.close();
            }
            
            eventSource = new EventSource(`{{BACKEND_URL}}/api/logs/${jobId}`);
            
            eventSource.onmessage = function(event) {
                const data = JSON.parse(event.data);
                
                if (data.status === 'completed') {
                    eventSource.close();
                    return;
                }
                
                addLog(data);
            };
            
            eventSource.onerror = function(error) {
                console.error('SSE error:', error);
                eventSource.close();
            };
        }
        
        function addLog(logData) {
            const logContainer = document.getElementById('logContainer');
            const logEntry = document.createElement('div');
            logEntry.className = 'log-entry';
            
            // Handle both string and object formats
            let message, timestamp;
            if (typeof logData === 'string') {
                message = logData;
                timestamp = new Date().toISOString();
            } else {
                message = logData.message || '';
                timestamp = logData.timestamp || new Date().toISOString();
            }
            
            // Format timestamp for display
            const date = new Date(timestamp);
            const timestampStr = date.toLocaleTimeString('en-US', { 
                hour12: false, 
                hour: '2-digit', 
                minute: '2-digit', 
                second: '2-digit',
                fractionalSecondDigits: 3
            });
            
            logEntry.innerHTML = `<span class="log-timestamp">[${timestampStr}]</span><span class="log-message">${message}</span>`;
            logContainer.appendChild(logEntry);
            logContainer.scrollTop = logContainer.scrollHeight;
        }
        
        async function pollResults(jobId) {
            const interval = setInterval(async () => {
                try {
                    const response = await fetch(`{{BACKEND_URL}}/api/results/${jobId}`);
                    const data = await response.json();
                    
                    if (data.status === 'success') {
                        clearInterval(interval);
                        const results = await loadSearchDisplayData(jobId);
                        displayResults(results);
                        await restoreIntegrationEvaluationQA(data);
                        document.getElementById('searchBtn').disabled = false;
                    } else if (data.status === 'error') {
                        clearInterval(interval);
                        showError('Job failed');
                        document.getElementById('searchBtn').disabled = false;
                    }
                } catch (error) {
                    console.error('Poll error:', error);
                }
            }, 2000);
        }
        
        function displayResults(results) {
            window.__latestSearchResults = results || {};
            const container = document.getElementById('resultsContent');
            // Pipeline error (e.g. query2modelcard failed or no model from query) - shown at top of results
            const errorBlock = results.error
                ? `<div style="padding: 12px; margin-bottom: 15px; color: #721c24; background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 6px;"><strong>❌ Pipeline error:</strong> ${results.error}</div>`
                : '';
            const escapeHtml = (s) => String(s == null ? '' : s)
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;');
            // Seed model + Query tables: same row, two columns (aligned with the two result cards below)
            const tableSearchSeedId = results.table_search_seed_model_id || null;
            const seedModelCell = results.error ? '' : (tableSearchSeedId
                ? `<div class="retrieval-seed-col"><div class="retrieval-seed-line"><strong>Table-search anchor seed:</strong> <a class="retrieval-seed-link" href="https://huggingface.co/${tableSearchSeedId}" target="_blank" rel="noopener noreferrer">${tableSearchSeedId}</a></div></div>`
                : `<span style="font-size: 12px; color: #856404;">⚠️ Table-search seed missing</span>`);
            // Query table path(s) from seed model card — used to run table search; result items below are model cards hit by that search
            let queryTables = [];
            let searchedTables = [];
            const c2t2c = results.card2tab2card_results || {};
            const searchTypeOrder = ['single_column', 'keyword'];
            for (const st of searchTypeOrder) {
                const data = c2t2c[st];
                if (!data) continue;
                if (Array.isArray(data.query_tables) && data.query_tables.length > 0) {
                    data.query_tables.forEach(p => { if (p && !queryTables.includes(p)) queryTables.push(p); });
                }
                if (Array.isArray(data.searched_tables) && data.searched_tables.length > 0) {
                    searchedTables = data.searched_tables;
                    break;
                }
            }
            // Filter by source: exclude s2orc/llm by ModelTables naming rules (e.g. 215768677_table2.csv)
            const classifyTableSource = (path) => {
                const b = String(path).split('/').pop().replace(/_s\.csv$/, '.csv').replace(/_t\.csv$/, '.csv');
                if (/^[0-9a-f]{32}_table_\d+\.csv$/.test(b)) return 'github';
                if (/^\d+\.\d+(?:v\d+)?_table\d+\.csv$/.test(b)) return 'html';
                if (/^[0-9a-f]{10}_table\d+\.csv$/.test(b)) return 'huggingface';
                if (/^\d+_table\d+\.csv$/.test(b)) return 'llm';
                return 'unknown';
            };
            queryTables = queryTables.filter(p => classifyTableSource(p) !== 'llm');
            const basename = (p) => String(p).split('/').pop();
            const tablesNoteCell = results.error
                ? '<span style="font-size: 12px; color: #666;">—</span>'
                : `<div class="retrieval-seed-col"><div class="retrieval-seed-line wrap"><strong>Query table(s):</strong><div class="retrieval-table-links">${
                    queryTables.length
                        ? queryTables.map((p) => {
                            const bn = basename(p);
                            const href = '{{BACKEND_URL}}/api/table-page?path=' + encodeURIComponent(p);
                            return `<a href="${href}" target="_blank" rel="noopener noreferrer">${escapeHtml(bn)}</a>`;
                          }).join(' ')
                        : '—'
                  }</div></div></div>`;
            const headerRowHtml = `<div class="results-grid retrieval-header-strip retrieval-header-strip--scaled" style="margin-top: 2px; margin-bottom: 4px;">
                <div>${seedModelCell}</div>
                <div>${tablesNoteCell}</div>
            </div>`;

            let jobIdStrip = (results.job_id != null && String(results.job_id).trim()) ? String(results.job_id).trim() : '';
            if (!jobIdStrip && typeof currentJobId !== 'undefined' && currentJobId) {
                jobIdStrip = String(currentJobId).trim();
            }
            if (!jobIdStrip && results.folder_path) {
                const segs = String(results.folder_path).replace(/\\/g, '/').split('/').filter(Boolean);
                if (segs.length) jobIdStrip = segs[segs.length - 1];
            }
            const metaStrip = document.getElementById('resultsMetaStrip');
            if (metaStrip) {
                const metaParts = [];
                if (results.query) {
                    metaParts.push(`<div style="margin:0;"><strong>Query:</strong> <span style="font-family: ui-monospace, monospace; font-size: 12px;">${escapeHtml(results.query)}</span></div>`);
                }
                if (jobIdStrip) {
                    metaParts.push(`<div style="margin-top:3px;font-size:11px;color:#555;"><strong>JOB_ID:</strong> <code style="font-size:11px;background:#f1f3f5;padding:1px 6px;border-radius:3px;">${escapeHtml(jobIdStrip)}</code></div>`);
                }
                const metaInline = [];
                if (results.model_top_k != null && results.model_top_k !== '') {
                    metaInline.push(`<strong>model_top_k:</strong> <code style="font-size:11px;background:#f1f3f5;padding:1px 6px;border-radius:3px;">${escapeHtml(results.model_top_k)}</code>`);
                }
                if (results.table_search_k != null && results.table_search_k !== '') {
                    metaInline.push(`<strong>table_search_k:</strong> <code style="font-size:11px;background:#f1f3f5;padding:1px 6px;border-radius:3px;">${escapeHtml(results.table_search_k)}</code>`);
                }
                if (results.timestamp) {
                    metaInline.push(`<strong>Timestamp:</strong> <code style="font-size:11px;background:#f1f3f5;padding:1px 6px;border-radius:3px;">${escapeHtml(results.timestamp)}</code>`);
                }
                if (metaInline.length) {
                    metaParts.push(`<div style="margin-top:4px;font-size:11px;color:#555;display:flex;gap:10px;flex-wrap:wrap;">${metaInline.join(' <span style="color:#c2c8cf;">|</span> ')}</div>`);
                }
                metaStrip.innerHTML = metaParts.join('');
                metaStrip.style.display = metaParts.length ? 'block' : 'none';
            }
            
            // Helper function to format model display (handle both string and object formats)
            function formatModel(model) {
                if (typeof model === 'string') {
                    return `<a href="https://huggingface.co/${model}" target="_blank">${model}</a>`;
                } else if (model && model.model_id) {
                    return `<a href="${model.url || `https://huggingface.co/${model.model_id}`}" target="_blank">${model.model_id}</a>`;
                }
                return model;
            }
            
            // Generate unique IDs for expandable sections
            const query2modelcardMoreId = 'q2m-more-' + Date.now();
            const card2tab2cardIds = {};
            const card2tab2cardResults = results.card2tab2card_results || {};
            Object.keys(card2tab2cardResults).forEach((type, idx) => {
                card2tab2cardIds[type] = 'card2tab2card-' + type + '-' + Date.now() + '-' + idx;
            });
            // All table search types that actually have card2tab2card results for this job
            const availableTableSearchTypes = Object.keys(card2tab2cardResults).filter(t => !!card2tab2cardResults[t]);
            
            // Get Query2Card (model search) results for all modes
            const query2modelcardAllModes = results.query2modelcard_all_modes || {};
            const retrievalModes = [
                { key: 'dense', label: 'Dense', desc: 'Semantic similarity using embeddings' },
                { key: 'sparse', label: 'Sparse', desc: 'Sparse retrieval via Pyserini Lucene BM25' },
                { key: 'hybrid', label: 'Hybrid', desc: 'Pyserini sparse + FAISS dense, then combine' }
            ];
            
            let retrievalHtml = `
                ${errorBlock}
                ${headerRowHtml}
                <div class="results-grid">
                    <div class="result-card" style="min-width: 0;">
                        <h3 style="margin-top: 0; margin-bottom: 8px; font-size: 14px; color: #495057;">
                            <span class="number-badge">1</span> Query2Card Results
                        </h3>
                        ${retrievalModes.map((modeInfo, idx) => {
                            const modeKey = modeInfo.key;
                            const modeResults = query2modelcardAllModes[modeKey] || [];
                            const isError = modeResults.error !== undefined;
                            const resultList = isError ? [] : (Array.isArray(modeResults) ? modeResults : []);
                            const sectionId = `q2m-${modeKey}-${Date.now()}-${idx}`;
                            
                            return `
                                <div class="search-type-section retrieval-method-tight" style="margin-bottom: 10px;">
                                    <div class="search-type-header expanded" onclick="toggleSearchType('${sectionId}', this)">
                                        <h4 style="margin: 0; display: flex; align-items: center; gap: 8px;">
                                            ${modeInfo.label}
                                            <span style="font-size: 12px; color: #666; font-weight: normal;">${isError ? 'Error' : resultList.length + ' models'}</span>
                                        </h4>
                                        <span data-nugget-score-inline-method="${modeKey}" class="retrieval-method-score" style="font-size:11px;color:#8a6d3b;flex-shrink:0;white-space:nowrap;margin-left:8px;">${nuggetScoreInlineInner(null)}</span>
                                    </div>
                                    <div class="collapsible-content expanded" id="${sectionId}" style="display:block;">
                                        ${isError ? `
                                            <div style="padding: 10px; color: #dc3545; background: #f8d7da; border-radius: 4px; margin: 10px 0;">
                                                ❌ Error: ${modeResults.error || 'Unknown error'}
                                            </div>
                                        ` : resultList.length > 0 ? `
                                            <ul class="result-list" style="list-style: none; padding: 0;">
                                                ${resultList.slice(0, 10).map(m => `<li class="result-item" style="font-size:12px;line-height:1.35;">${formatModel(m)}</li>`).join('')}
                                                ${resultList.length > 10 ? `
                                                    <li class="collapsible-content expanded" id="${sectionId}-more" style="display:block;">
                                                        ${resultList.slice(10).map(m => `<div class="result-item" style="font-size:12px;line-height:1.35;">${formatModel(m)}</div>`).join('')}
                                                    </li>
                                                    <li>
                                                        <span class="expand-toggle" onclick="toggleExpand('${sectionId}-more', this)">
                                                            Show ${resultList.length - 10} more
                                                        </span>
                                                    </li>
                                                ` : ''}
                                            </ul>
                                        ` : `
                                            <div style="padding: 10px; color: #666; background: #f8f9fa; border-radius: 4px; margin: 10px 0;">
                                                No results available
                                            </div>
                                        `}
                                    </div>
                                </div>
                            `;
                        }).join('')}
                    </div>
                    <div class="result-card" style="min-width: 0; box-shadow: 0 2px 6px rgba(0,0,0,0.06); border-radius: 6px;">
                        <h3 style="margin-top: 0; margin-bottom: 8px; font-size: 14px; color: #495057;"><span class="number-badge">2</span> Query2Tab2Card Results</h3>
                        ${(() => {
                            // Filter: show keyword, unionable, and joinable types
                            const allowedTypes = ['keyword', 'unionable', 'single_column', 'multi_column'];
                            // Map joinable types to display names
                            const typeDisplayNames = {
                                'single_column': 'Joinable (single_column)',
                                'multi_column': 'Joinable (multi_column)',
                                'keyword': 'Keyword',
                                'unionable': 'Unionable'
                            };
                            const entries = Object.entries(card2tab2cardResults)
                                .filter(([type, data]) => allowedTypes.includes(type))
                                .map(([type, data]) => {
                                    const models = Array.isArray(data) ? data : (data.model_ids || []);
                                    const displayName = typeDisplayNames[type] || type;
                                    return { type, displayName, data, models, count: models.length };
                                })
                                .sort((a, b) => b.count - a.count); // Sort by count descending
                            
                            return entries.map(({type, displayName, data, models}) => {
                            const sectionId = card2tab2cardIds[type];
                            // Handle both old format (array) and new format (object with model_ids and intermediate)
                            const intermediate = data.intermediate || {};
                            const tableToModels = intermediate.table_to_models || {};
                            
                            // Real table count for this search type
                            const retrievedTableFilenames = intermediate.retrieved_table_filenames || [];
                            const tableCountFromFilenames = Array.isArray(retrievedTableFilenames) ? retrievedTableFilenames.length : 0;
                            const tableCountFromMapping = Object.keys(tableToModels).length;
                            const realTableCount = tableCountFromFilenames || tableCountFromMapping || 0;
                            
                            // Build reverse mapping: model_id -> list of retrieved tables (searched tables only, no query tables)
                            const modelToTables = {};
                            Object.entries(tableToModels).forEach(([table, modelList]) => {
                                const normalizedModelList = Array.isArray(modelList) ? modelList : [];
                                normalizedModelList.forEach(modelIdOrObj => {
                                    let modelId = typeof modelIdOrObj === 'string'
                                        ? modelIdOrObj
                                        : (modelIdOrObj?.model_id || modelIdOrObj);
                                    if (modelId) {
                                        modelId = String(modelId).trim();
                                        if (modelId) {
                                            if (!modelToTables[modelId]) modelToTables[modelId] = [];
                                            modelToTables[modelId].push(table);
                                        }
                                    }
                                });
                            });
                            // Sort models by number of retrieved tables descending: Model2 (by t2,t3,t4) before Model1 (by t1,t3)
                            const sortedModels = [...models].sort((a, b) => {
                                const idA = (typeof a === 'string' ? a : (a?.model_id || a))?.toString().trim() || '';
                                const idB = (typeof b === 'string' ? b : (b?.model_id || b))?.toString().trim() || '';
                                return (modelToTables[idB]?.length || 0) - (modelToTables[idA]?.length || 0);
                            });
                            
                            return `
                                <div class="search-type-section retrieval-method-tight" style="margin-bottom: 10px;">
                                    <div class="search-type-header expanded" onclick="toggleSearchType('${sectionId}', this)">
                                        <h4 style="margin: 0; display: flex; align-items: center; gap: 8px;">
                                            ${displayName}
                                            <span style="font-size: 12px; color: #666; font-weight: normal;">
                                                ${models.length} models${(SHOW_CARD2TAB2CARD_MODEL_TABLES && realTableCount) ? ` from ${realTableCount} tables` : ''}
                                            </span>
                                        </h4>
                                        <span data-nugget-score-inline-method="${type}" class="retrieval-method-score" style="font-size:11px;color:#8a6d3b;flex-shrink:0;white-space:nowrap;margin-left:8px;">${nuggetScoreInlineInner(null)}</span>
                                    </div>
                                    <div class="collapsible-content expanded" id="${sectionId}" style="display:block;">
                                        <ul class="result-list" style="list-style: none; padding: 0;">
                                            ${sortedModels.length > 0 ? sortedModels.map((m, idx) => {
                                                if (!SHOW_CARD2TAB2CARD_MODEL_TABLES) {
                                                    return `<li class="result-item" style="font-size:12px;line-height:1.35;">${formatModel(m)}</li>`;
                                                }
                                                let modelId = typeof m === 'string' ? m : (m.model_id || m);
                                                modelId = String(modelId).trim();
                                                const modelUrl = typeof m === 'string' ? `https://huggingface.co/${modelId}` : (m.url || `https://huggingface.co/${modelId}`);
                                                const modelTables = modelToTables[modelId] || [];
                                                const hasTables = modelTables.length > 0;
                                                const tableLine = hasTables ? modelTables.map(t => String(t).split('/').pop()).join(' ') : '';
                                                return `
                                                    <li class="result-item">
                                                        <div style="display: flex; align-items: baseline; gap: 4px; flex-wrap: wrap; line-height: 1.45;">
                                                            <a href="${modelUrl}" target="_blank" style="color: #0056b3; text-decoration: none; font-size: 12px;">${modelId}</a>
                                                            ${hasTables ? ` <span style="font-size: 10px; color: #888;">(${modelTables.length} tables)</span>` : ''}
                                                            ${hasTables ? `<span style="font-size: 10px; color: #999; font-family: monospace;">${tableLine}</span>` : ''}
                                                        </div>
                                                    </li>
                                                `;
                                            }).join('') : '<li>No results</li>'}
                                        </ul>
                                    </div>
                                </div>
                            `;
                            }).join('');
                        })()}
                    </div>
                </div>
                <div class="result-card" style="margin-top: 12px; padding: 10px 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.06); border-radius: 6px;">
                    <div id="retrievalEvaluationSummaryMount" style="font-size: 12px; color: #888;">
                        <div class="evaluation-summary-panel">Running nugget-based evaluation automatically...</div>
                    </div>
                </div>
            `;
            
            // Table Integration: 1 Query2Card vs 2 Query2Tab2Card — separate params and dropdown switching
            const integrationCardStyle = 'padding: 12px; background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%); border-radius: 8px; border: 1px solid #dee2e6; font-size: 13px; color: #212529; min-width: 0;';
            const integrationTitleStyle = 'margin-top: 0; margin-bottom: 6px; font-size: 15px; font-weight: 600; color: #1a1d21;';
            const topKLabelStyle = 'display: block; margin-bottom: 2px; font-size: 11px; font-weight: 500; color: #212529;';
            const intH4Flex = 'margin: 0 0 6px 0; font-size: 13px; display: flex; align-items: center; gap: 6px; flex-wrap: wrap;';
            const defaultIntegrationK = results.table_search_k || 10;
            const defaultIntegrationMaxModels = results.model_top_k || 3;
            let integrationPanelHtml = `
                <div class="integration-section" style="${integrationCardStyle}; margin-top: 0;">
                    <h3 style="${integrationTitleStyle}">Table Integration</h3>
                    <p style="font-size: 12px; color: #5a6268; margin-bottom: 10px;">Integrate tables from both searches.</p>
                    <div class="pdf-section pipeline-diagram-frame integration-diagram-frame">
                        <img class="pipeline-diagram-img integration-diagram-img" src="/static/docs/tableintegration.png" alt="Table integration overview" />
                    </div>
                    <div style="display: flex; gap: 10px; align-items: flex-end; flex-wrap: wrap; margin-bottom: 6px;">
                        <div style="flex: 0 0 auto;"><label style="${topKLabelStyle}">integration method:</label><select id="integration_type" class="form-control" onchange="syncBothIntegrationDisplays();" style="width: 100px; box-sizing: border-box; padding: 4px 6px; font-size: 12px;">
                            <option value="alite">ALITE</option>
                        </select></div>
                        <!-- top k tables/models: commented out - use defaults; integrate prints #tables/#models -->
                        <div style="display: none;"><input type="number" id="integration_k" value="${defaultIntegrationK}" min="1" max="50"><input type="number" id="integration_max_models" value="${defaultIntegrationMaxModels}" min="1" max="50"></div>
                        <button id="integrationRunBothBtn" onclick="runBothIntegrations('${results.job_id || currentJobId}')" style="padding: 6px 14px; font-size: 13px; font-weight: 600;">Integrated</button>
                        <a href="${integrationReviewPageHref(results.job_id || currentJobId)}" target="_blank" rel="noopener noreferrer" style="font-size: 12px; color: #0056b3; text-decoration: none; padding: 6px 0;">
                            Review integration details here
                        </a>
                    </div>
                    <div id="integrationResultsContainer" style="margin-top: 12px;">
                        <div style="margin-bottom: 16px; padding: 10px; background: #e7f3ff; border-radius: 6px; border-left: 4px solid #007bff;">
                            <h4 style="${intH4Flex} color: #004085;">${INT_TITLE_C2C_HTML}</h4>
                            <div style="display: flex; gap: 8px; align-items: flex-end; flex-wrap: wrap; margin-bottom: 8px;">
                                <div style="flex: 0 0 auto;">
                                    <label style="${topKLabelStyle}">Query2Card mode:</label>
                                    <select id="integration_q2m_mode" class="form-control" onchange="syncModelSearchDisplay()" style="width: 100px; box-sizing: border-box; padding: 4px 6px; font-size: 12px;">
                                        <option value="dense">Dense</option>
                                        <option value="sparse">Sparse</option>
                                        <option value="hybrid">Hybrid</option>
                                    </select>
                                </div>
                            </div>
                            <div id="integrationModelSearchResults"></div>
                        </div>
                        <div style="padding: 10px; background: #d4edda; border-radius: 6px; border-left: 4px solid #28a745;">
                            <h4 style="${intH4Flex} color: #155724;">${INT_TITLE_C2T2C_HTML}</h4>
                            <div style="display: flex; gap: 8px; align-items: flex-end; flex-wrap: wrap; margin-bottom: 8px;">
                                <div style="flex: 0 0 auto;"><label style="${topKLabelStyle}">tables source:</label><select id="integration_tables_source" class="form-control" onchange="syncTableSearchDisplay()" style="width: 175px; box-sizing: border-box; padding: 4px 6px; font-size: 12px;" title="Searched tables: from this job's table search. All from Modelcards: parquet (DuckDB).">
                                    <option value="intermediate" selected>Searched tables</option>
                                    <option value="all_from_modelcards">All tables from Modelcards</option>
                                </select></div>
                                <div style="flex: 0 0 auto;"><label style="${topKLabelStyle}">search type:</label><select id="integration_search_type" class="form-control" onchange="syncTableSearchDisplay()" style="width: 110px; box-sizing: border-box; padding: 4px 6px; font-size: 12px;">
                                    <!-- Currently only these three have non-empty results in pipeline -->
                                    <option value="unionable">Unionable</option>
                                    <option value="single_column">Single Column</option>
                                    <option value="keyword">Keyword</option>
                                    <!--
                                    <option value="multi_column">Multi Column</option>
                                    <option value="complex">Complex</option>
                                    <option value="correlation">Correlation</option>
                                    <option value="imputation">Imputation</option>
                                    <option value="augmentation">Augmentation</option>
                                    <option value="dependent_data">Dependent Data</option>
                                    <option value="feature_for_ml">Feature for ML</option>
                                    <option value="multi_column_collinearity">Multi-Column Collinearity</option>
                                    <option value="negative_example">Negative Example</option>
                                    -->
                                </select></div>
                            </div>
                            <div id="integrationResults"></div>
                        </div>
                    </div>
                </div>
            `;
            if (ENABLE_POST_INTEGRATION_ANALYSIS) {
            integrationPanelHtml += `
                <div id="integrationShortAnalysis" class="integration-summary-section" style="margin-top: 16px; padding: 14px; background: #e2e3e5; border-radius: 6px; border: 2px solid #6c757d; display: none;">
                    <h4 style="margin: 0 0 6px 0; font-size: 14px; color: #383d41;">Summary (between Table Integration and Evaluation)</h4>
                    <p style="font-size: 11px; color: #6c757d; margin: 0 0 10px 0;">Deterministic comparison: column overlap, Jaccard, containment, coverage. No LLM.</p>
                    <div id="integrationShortAnalysisContent"></div>
                </div>
                <div class="evaluation-section" style="margin-top: 16px; padding: 12px; background: #fff3cd; border-radius: 6px; border: 2px solid #ffc107;">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                        <div>
                            <h3 style="margin: 0 0 4px 0; color: #856404; font-size: 15px;">📊 Evaluation on Integrated Tables</h3>
                            <p style="font-size: 12px; color: #666; margin: 0;">Evaluate diversity between 2 Query2Tab2Card Results and 1 Query2Card Results integrations using LLM.</p>
                        </div>
                        <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                            <button id="evalQaBothBtn" onclick="runEvaluationAndQABoth('${results.job_id || currentJobId}')" 
                                    style="padding: 6px 14px; font-size: 13px; background: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: 500;">
                                📊💬 Both (parallel)
                            </button>
                            <button id="evaluationBtn" onclick="runEvaluation('${results.job_id || currentJobId}')" 
                                    style="padding: 6px 14px; font-size: 13px; background: #ffc107; color: #000; border: none; border-radius: 4px; cursor: pointer; font-weight: 500;">
                                📊 Evaluation
                            </button>
                        </div>
                    </div>
                    <div id="evaluationResults" style="margin-top: 12px; display: none;"></div>
                    </div>
                    
                    <div class="qa-section" style="margin-top: 16px; padding: 12px; background: #d1ecf1; border-radius: 6px; border: 2px solid #17a2b8;">
                        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                            <div>
                                <h3 style="margin: 0 0 4px 0; color: #0c5460; font-size: 15px;">📊 Table Ranking</h3>
                                <p style="font-size: 12px; color: #666; margin: 0;">Rank / compare models based on answers over both integrated tables.</p>
                            </div>
                            <button id="qaBtn" onclick="runQABoth('${results.job_id || currentJobId}')" 
                                    style="padding: 6px 14px; font-size: 13px; background: #17a2b8; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: 500;">
                                📊 Rank Tables
                            </button>
                        </div>
                        <div id="qa_after_click" style="display: none; margin-top: 20px;">
                            <h4 style="margin: 0 0 10px 0; font-size: 14px; color: #0c5460;">Integrated tables</h4>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 6px;">
                                <div><strong style="font-size: 13px; color: #0c5460;">2 Query2Tab2Card Results</strong></div>
                                <div><strong style="font-size: 13px; color: #0c5460;">1 Query2Card Results</strong></div>
                            </div>
                            <div id="qaIntegratedPaths" style="font-size: 11px; color: #555; margin-bottom: 12px;"></div>
                            <h4 style="margin: 0 0 10px 0; font-size: 14px; color: #0c5460;">Answers compare</h4>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                                <div>
                                    <strong style="font-size: 13px; color: #0c5460;">2 Query2Tab2Card Results</strong>
                                    <div id="qaResultsTableSearch" style="margin-top: 8px;"></div>
                                </div>
                                <div>
                                    <strong style="font-size: 13px; color: #0c5460;">1 Query2Card Results</strong>
                                    <div id="qaResultsModelSearch" style="margin-top: 8px;"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            }
            
            container.innerHTML = retrievalHtml;
            const intMount = document.getElementById('integrationPanelMount');
            if (intMount) {
                intMount.innerHTML = integrationPanelHtml;
            } else {
                container.innerHTML += integrationPanelHtml;
            }
            window.__modelSearchRuns = [];
            window.__tableSearchRuns = [];
            document.getElementById('resultsSection').classList.add('active');
            syncBothIntegrationDisplays();
            ensureEvaluationForResults(results.job_id || currentJobId, 'retrievalEvaluationSummaryMount');
        }

        function normalizeModelSearchRunKey(key) {
            return String(key || '').toLowerCase();
        }
        function getModelSearchKey(integrationType, retrievalMode) {
            const it = String(integrationType || 'alite').toLowerCase().replace(/[^a-z0-9_]/g, '_');
            const rm = String(retrievalMode || 'dense').toLowerCase().replace(/[^a-z0-9_]/g, '_');
            return `${it}_${rm}`;
        }
        function modelRunMatchesKey(run, wantKey) {
            const raw = run.key != null && String(run.key) !== ''
                ? String(run.key)
                : getModelSearchKey(run.integration_type, run.query2modelcard_retrieval_mode);
            return normalizeModelSearchRunKey(raw) === normalizeModelSearchRunKey(wantKey);
        }
        function getTableSearchKey(integrationType, searchType, tablesSource) {
            const src = (tablesSource || 'intermediate').toLowerCase().replace(/-/g, '_').replace(/[^a-z0-9_]/g, '_');
            return [(integrationType || 'alite'), (searchType || 'single_column'), src].map(s => String(s).toLowerCase().replace(/[^a-z0-9_]/g, '_')).join('_');
        }

        function refreshQAIntegratedPaths() {
            const target = document.getElementById('qaIntegratedPaths');
            if (!target) return;
            // Use run data so paths show even when DOM structure differs; same key logic as sync*Display
            const integrationType = (document.getElementById('integration_type') || {}).value || 'alite';
            const searchType = (document.getElementById('integration_search_type') || {}).value || 'single_column';
            const tablesSource = (document.getElementById('integration_tables_source') || {}).value || 'intermediate';
            const tableKey = getTableSearchKey(integrationType, searchType, tablesSource);
            const retrievalMode = (document.getElementById('integration_q2m_mode') || {}).value || 'dense';
            const modelKey = getModelSearchKey(integrationType, retrievalMode);
            const tableRun = (window.__tableSearchRuns || []).find(r => (r.key || getTableSearchKey(r.integration_type, r.search_type, r.tables_source)) === tableKey);
            const modelRun = (window.__modelSearchRuns || []).find(r => modelRunMatchesKey(r, modelKey));
            const grouped = getGroupedIntegratedTables(tableRun);
            const tablePath = grouped.length === 1 ? (grouped[0].saved_path || '') : '';
            const modelPath = (modelRun && modelRun.saved_path) ? modelRun.saved_path : '';
            const tableHtml = grouped.length > 1
                ? grouped.map((group, idx) => `${idx + 1}. ${integrationSavedPathLink(group.saved_path || '')}`).join('<br>')
                : (tablePath ? integrationSavedPathLink(tablePath) : '<span style="color:#999;">N/A</span>');
            const modelHtml = modelPath ? integrationSavedPathLink(modelPath) : '<span style="color:#999;">N/A</span>';
            target.innerHTML = `
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                    <div>2 Query2Tab2Card Results CSV: ${tableHtml}</div>
                    <div>1 Query2Card Results CSV: ${modelHtml}</div>
                </div>
            `;
        }
        
        function syncBothIntegrationDisplays() {
            try { syncModelSearchDisplay(); } catch (e) { console.error('syncModelSearchDisplay error:', e); }
            try { syncTableSearchDisplay(); } catch (e) { console.error('syncTableSearchDisplay error:', e); }
        }
        
        async function syncModelSearchDisplay() {
            const leftDiv = document.getElementById('integrationModelSearchResults');
            const container = document.getElementById('integrationResultsContainer');
            if (!leftDiv || !container) return;
            container.style.display = 'block';
            const integrationType = (document.getElementById('integration_type') || {}).value || 'alite';
            const retrievalMode = (document.getElementById('integration_q2m_mode') || {}).value || 'dense';
            const key = getModelSearchKey(integrationType, retrievalMode);
            const runs = window.__modelSearchRuns || [];
            const run = runs.find(r => modelRunMatchesKey(r, key));
            const placeholder = '<div style="padding: 12px; background: #f8f9fa; border: 1px dashed #dee2e6; border-radius: 6px; color: #6c757d; font-size: 13px;">No result for this combination. Click <strong>Integrated</strong> to run.</div>';
            const noResultMsg = placeholder;
            if (run && run.status === 'success' && run.integrated_table) {
                const stats = run.stats || {};
                let extra = '';
                const modelIds = run.models_with_tables || [];
                const tablePathsList = run.table_paths || [];
                if (modelIds.length > 0) {
                    const modelToTables = run.model_to_table_paths || {};
                    const linesHtml = formatQuery2CardModelTableLinesHtml(modelIds, modelToTables);
                    const modelsCount = stats.models_with_tables != null ? stats.models_with_tables : modelIds.length;
                    const tablesCount = stats.total_unique_tables != null ? stats.total_unique_tables : tablePathsList.length;
                    extra = `<div style="margin-bottom: 10px; padding: 8px; background: #e7f3ff; border-radius: 4px; font-size: 12px;">
                        <details style="margin: 0;">
                            <summary style="cursor: pointer; list-style: none; outline: none;">
                                <span style="font-weight: 600;">${INT_MODEL_IDS_C2C}</span>
                                <span style="color:#004085;">(${modelsCount} models, ${tablesCount} tables)</span>
                            </summary>
                            <div style="margin-top: 6px; font-size: 11px;">
                                ${linesHtml}
                            </div>
                        </details>
                    </div>`;
                } else {
                    extra = '<div style="margin-bottom: 10px; padding: 8px; background: #f8f9fa; border-radius: 4px; font-size: 12px; color: #6c757d;">Model IDs (1 Query2Card Results): — (none or not available)</div>';
                }
                leftDiv.innerHTML = renderIntegrationTable(run.integrated_table, stats, { title: INT_TITLE_C2C_HTML, successColor: '#007bff', extraHtml: extra, savedPath: run.saved_path || '', downloadId: 'model-search-' + key });
            } else if (run && run.single_table_preview && run.single_table_preview.columns) {
                const single = run.single_table_preview;
                const extra = `<div style="margin-bottom: 10px; padding: 8px; background: #e7f3ff; border-radius: 4px; font-size: 12px;">Single related table found, so no integration was needed. Showing that table directly.</div>`;
                leftDiv.innerHTML = renderIntegrationTable(single, single.stats || {}, { title: INT_TITLE_C2C_HTML, successColor: '#007bff', extraHtml: extra, savedPath: single.path || '', downloadId: 'model-search-single-preview-' + key });
            } else {
                leftDiv.innerHTML = run && run.status !== 'success' ? `<div style="padding: 10px; border-radius: 4px; border: 1px solid #dc3545; color: #dc3545; font-size: 12px;">❌ ${run.message || run.error || 'Integration failed'}</div>` : noResultMsg;
            }
            initTablePanZoom(leftDiv);
            refreshIntegrationShortAnalysis();
        }
        
        function refreshIntegrationShortAnalysis() {
            const container = document.getElementById('integrationShortAnalysis');
            const content = document.getElementById('integrationShortAnalysisContent');
            if (!container || !content) return;
            const integrationType = (document.getElementById('integration_type') || {}).value || 'alite';
            const searchType = (document.getElementById('integration_search_type') || {}).value || 'single_column';
            const tablesSource = (document.getElementById('integration_tables_source') || {}).value || 'intermediate';
            const tableKey = getTableSearchKey(integrationType, searchType, tablesSource);
            const retrievalMode = (document.getElementById('integration_q2m_mode') || {}).value || 'dense';
            const modelKey = getModelSearchKey(integrationType, retrievalMode);
            const tableRun = (window.__tableSearchRuns || []).find(r => (r.key || getTableSearchKey(r.integration_type, r.search_type, r.tables_source)) === tableKey);
            const modelRun = (window.__modelSearchRuns || []).find(r => modelRunMatchesKey(r, modelKey));
            const groupedTableRuns = getGroupedIntegratedTables(tableRun);
            const tableT = groupedTableRuns.length === 1 ? groupedTableRuns[0].integrated_table : null;
            const modelT = modelRun && modelRun.integrated_table;
            if (!tableT || !modelT || !tableT.columns || !modelT.columns) {
                container.style.display = 'none';
                return;
            }
            const colsA = new Set(tableT.columns);
            const colsB = new Set(modelT.columns);
            const overlap = [...colsA].filter(c => colsB.has(c));
            const onlyTable = [...colsA].filter(c => !colsB.has(c));
            const onlyModel = [...colsB].filter(c => !colsA.has(c));
            const totalRowsA = (tableT.data || []).length;
            const totalRowsB = (modelT.data || []).length;
            const nonNullPct = (t, colIdx) => {
                const data = t.data || [];
                if (!data.length) return 0;
                let n = 0;
                for (let r = 0; r < data.length; r++) {
                    const v = data[r][colIdx];
                    if (v != null && v !== '') n++;
                }
                return (n / data.length * 100).toFixed(1);
            };
            const colsOnlyInTable = onlyTable.length;
            const colsOnlyInModel = onlyModel.length;
            const colsCommon = overlap.length;
            const unionSize = colsA.size + colsB.size - colsCommon;
            const jaccardCols = unionSize > 0 ? (colsCommon / unionSize).toFixed(3) : '0';
            const containmentTableInModel = colsA.size > 0 ? (colsCommon / colsA.size).toFixed(3) : '-';
            const containmentModelInTable = colsB.size > 0 ? (colsCommon / colsB.size).toFixed(3) : '-';
            let missingHtml = '';
            if (colsCommon > 0 && tableT.data && modelT.data && tableT.data.length > 0 && modelT.data.length > 0) {
                const sampleCommon = overlap.slice(0, 5);
                missingHtml = '<p style="margin: 4px 0; font-size: 12px;"><strong>Coverage (non-null %) on overlap columns:</strong></p><ul style="margin: 0 0 8px 0; padding-left: 18px; font-size: 11px;">';
                sampleCommon.forEach(col => {
                    const iA = tableT.columns.indexOf(col);
                    const iB = modelT.columns.indexOf(col);
                    const pctA = iA >= 0 ? nonNullPct(tableT, iA) : '-';
                    const pctB = iB >= 0 ? nonNullPct(modelT, iB) : '-';
                    missingHtml += `<li><code>${col}</code>: 2 Query2Tab2Card ${pctA}%, 1 Query2Card ${pctB}%</li>`;
                });
                if (overlap.length > 5) missingHtml += `<li>… and ${overlap.length - 5} more</li>`;
                missingHtml += '</ul>';
            }
            content.innerHTML = `
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; font-size: 12px;">
                    <div style="padding: 8px; background: #fff; border-radius: 4px; border: 1px solid #dee2e6;">
                        <strong>Schema-level (column overlap)</strong>
                        <ul style="margin: 6px 0 0 0; padding-left: 18px;">
                            <li>In both: <strong>${colsCommon}</strong></li>
                            <li>Only in 2 Query2Tab2Card Results: <strong>${colsOnlyInTable}</strong>${onlyTable.length ? ' (' + onlyTable.slice(0, 3).join(', ') + (onlyTable.length > 3 ? '…' : '') + ')' : ''}</li>
                            <li>Only in 1 Query2Card Results: <strong>${colsOnlyInModel}</strong>${onlyModel.length ? ' (' + onlyModel.slice(0, 3).join(', ') + (onlyModel.length > 3 ? '…' : '') + ')' : ''}</li>
                        </ul>
                    </div>
                    <div style="padding: 8px; background: #fff; border-radius: 4px; border: 1px solid #dee2e6;">
                        <strong>Deterministic metrics (no LLM)</strong>
                        <ul style="margin: 6px 0 0 0; padding-left: 18px;">
                            <li>Column Jaccard: <strong>${jaccardCols}</strong> (|A∩B|/|A∪B|)</li>
                            <li>Containment (2 Query2Tab2Card→1 Query2Card): <strong>${containmentTableInModel}</strong> (|A∩B|/|A|)</li>
                            <li>Containment (1 Query2Card→2 Query2Tab2Card): <strong>${containmentModelInTable}</strong> (|A∩B|/|B|)</li>
                        </ul>
                    </div>
                </div>
                <div style="margin-top: 8px; padding: 8px; background: #fff; border-radius: 4px; border: 1px solid #dee2e6; font-size: 12px;">
                    <strong>Row counts</strong>: 2 Query2Tab2Card Results <strong>${totalRowsA}</strong> rows, ${tableT.columns.length} cols; 1 Query2Card Results <strong>${totalRowsB}</strong> rows, ${modelT.columns.length} cols.
                </div>
                ${missingHtml}
                <p style="margin: 8px 0 0 0; font-size: 10px; color: #6c757d;">Schema + data-consistency metrics only (overlap, Jaccard, containment, coverage). Deterministic; no LLM.</p>
            `;
            container.style.display = 'block';
        }
        
        async function syncTableSearchDisplay() {
            const rightDiv = document.getElementById('integrationResults');
            const container = document.getElementById('integrationResultsContainer');
            if (!rightDiv || !container) return;
            container.style.display = 'block';
            const integrationType = (document.getElementById('integration_type') || {}).value || 'alite';
            const searchType = (document.getElementById('integration_search_type') || {}).value || 'single_column';
            const tablesSource = (document.getElementById('integration_tables_source') || {}).value || 'intermediate';
            const key = getTableSearchKey(integrationType, searchType, tablesSource);
            const runs = window.__tableSearchRuns || [];
            const run = runs.find(r => {
                const rKey = r.key || getTableSearchKey(r.integration_type, r.search_type, r.tables_source);
                return rKey === key;
            });
            const placeholder = '<div style="padding: 12px; background: #f8f9fa; border: 1px dashed #dee2e6; border-radius: 6px; color: #6c757d; font-size: 13px;">No result for this combination. Click <strong>Integrated</strong> to run.</div>';
            const noResultMsg = placeholder;
            const groupedRuns = getGroupedIntegratedTables(run);
            if (run && run.status === 'success' && groupedRuns.length > 0) {
                let extra = '';
                const modelIds = run.models_with_tables || [];
                const tablePathsList = run.table_paths || [];
                if (modelIds.length > 0) {
                    const modelToTables = run.model_to_table_paths || {};
                    let rtRows = run.retrieved_table_model_rows;
                    if (!Array.isArray(rtRows) || !rtRows.length) {
                        rtRows = buildRetrievedTableModelRowsFallback(modelToTables, tablePathsList);
                    }
                    const tsRun = run.tables_source || (run.stats && run.stats.tables_source) || 'intermediate';
                    const traceHtml = formatQuery2Tab2CardTraceHtml(run.query_tables || [], rtRows, {
                        tablesSource: tsRun,
                        modelToTablePaths: run.model_to_table_paths || {},
                        rankedModelIds: modelIds,
                        pipelineTrace: run.pipeline_trace,
                        tab2tabTraceRows: run.tab2tab_trace_rows,
                        afterModelCapTraceRows: run.after_model_cap_trace_rows,
                    });
                    const stats = run.stats || {};
                    const modelsCount = stats.models_with_tables != null ? stats.models_with_tables : modelIds.length;
                    extra = `<div style="margin-bottom: 10px; padding: 8px; background: #d4edda; border-radius: 4px; font-size: 12px;">
                        <details style="margin: 0;">
                            <summary style="cursor: pointer; list-style: none; outline: none;">
                                <span style="font-weight: 600;">${INT_MODEL_IDS_C2T2C}</span>
                                <span style="color:#155724;">(${modelsCount} models, ${describeTableSearchCount(run, stats)})</span>
                            </summary>
                            <div style="margin-top: 6px; font-size: 11px;">
                                ${traceHtml}
                            </div>
                        </details>
                    </div>`;
                } else {
                    extra = '<div style="margin-bottom: 10px; padding: 8px; background: #f8f9fa; border-radius: 4px; font-size: 12px; color: #6c757d;">Model IDs (2 Query2Tab2Card Results): — (none or not available)</div>';
                }
                rightDiv.innerHTML = renderGroupedIntegrationTables(groupedRuns, { title: INT_TITLE_C2T2C_HTML, successColor: '#28a745', extraHtml: extra, downloadPrefix: 'table-search-' + key });
            } else if (run && run.single_table_preview && run.single_table_preview.columns) {
                const single = run.single_table_preview;
                const extra = `<div style="margin-bottom: 10px; padding: 8px; background: #d4edda; border-radius: 4px; font-size: 12px;">Only one table is available for this setting, so no integration was needed. Showing that table directly.</div>`;
                rightDiv.innerHTML = renderIntegrationTable(single, single.stats || {}, { title: INT_TITLE_C2T2C_HTML, successColor: '#28a745', extraHtml: extra, savedPath: single.path || '', downloadId: 'table-search-single-preview-' + key });
            } else if (run && groupedRuns.length === 0 && (run.pipeline_trace || run.tab2tab_trace_rows || run.query_tables || run.model_to_table_paths || run.models_with_tables)) {
                // Pre-integration preview (already have retrieval relationship info, but integration not run yet).
                const modelIds = run.models_with_tables || [];
                const tablePathsList = run.table_paths || [];
                const modelToTables = run.model_to_table_paths || {};
                let rtRows = run.retrieved_table_model_rows;
                if (!Array.isArray(rtRows) || !rtRows.length) {
                    rtRows = buildRetrievedTableModelRowsFallback(modelToTables, tablePathsList);
                }
                const tsRun = run.tables_source || (run.stats && run.stats.tables_source) || 'intermediate';
                const traceHtml = formatQuery2Tab2CardTraceHtml(run.query_tables || [], rtRows, {
                    tablesSource: tsRun,
                    modelToTablePaths: modelToTables,
                    rankedModelIds: modelIds,
                    pipelineTrace: run.pipeline_trace,
                    tab2tabTraceRows: run.tab2tab_trace_rows,
                    afterModelCapTraceRows: run.after_model_cap_trace_rows,
                });
                rightDiv.innerHTML = `<div style="padding: 12px; background: #f8f9fa; border: 1px dashed #dee2e6; border-radius: 6px;">
                    <div style="font-size: 13px; color: #6c757d; margin-bottom: 8px;">Integration not run yet. Click <strong>Integrated</strong> to build merged table.</div>
                    <details open style="margin: 0;">
                        <summary style="cursor: pointer; list-style: none; outline: none;">
                            <span style="font-weight: 600;">${INT_MODEL_IDS_C2T2C}</span>
                            <span style="color:#155724;">(pre-integration preview)</span>
                        </summary>
                        <div style="margin-top: 8px; font-size: 11px;">
                            ${traceHtml}
                        </div>
                    </details>
                </div>`;
            } else {
                rightDiv.innerHTML = run && run.status !== 'success' ? `<div style="padding: 10px; border-radius: 4px; border: 1px solid #dc3545; color: #dc3545; font-size: 12px;">❌ ${run.message || run.error || 'Integration failed'}</div>` : noResultMsg;
            }
            initTablePanZoom(rightDiv);
            refreshIntegrationShortAnalysis();
        }
        
        function initTablePanZoom(root) {}
        
        function showTableAsImage(tableId, downloadId) {
            const t = window.__integrationTables[downloadId];
            if (!t || !t.columns || !t.data) return;
            const modalId = `table-image-modal-${tableId}`;
            const existing = document.getElementById(modalId);
            if (existing) { existing.remove(); }
            const modal = document.createElement('div');
            modal.id = modalId;
            modal.style.cssText = 'position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.85); z-index: 10000; display: flex; align-items: center; justify-content: center; cursor: move;';
            const container = document.createElement('div');
            container.style.cssText = 'position: relative; max-width: 95%; max-height: 95%; overflow: auto; background: white; border-radius: 8px; padding: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.3);';
            container.style.transform = 'scale(1)';
            container.style.transformOrigin = 'center center';
            let scale = 1, isDragging = false, startX = 0, startY = 0, offsetX = 0, offsetY = 0;
            const buildTableHtml = () => {
                return `<table style="width: max-content; border-collapse: collapse; font-size: 12px; margin: 0 auto;">
                    <thead><tr style="background: #f8f9fa;">
                        ${t.columns.map(col => `<th style="border: 1px solid #dee2e6; padding: 8px; text-align: left; background: #f8f9fa; white-space: nowrap;">${normalizeIntegrationText(col)}</th>`).join('')}
                    </tr></thead>
                    <tbody>
                        ${t.data.map(row => `<tr>${row.map(cell => `<td style="border: 1px solid #dee2e6; padding: 6px; white-space: nowrap;">${formatIntegrationCell(cell)}</td>`).join('')}</tr>`).join('')}
                    </tbody>
                </table>`;
            };
            container.innerHTML = `<div style="position: absolute; top: 10px; right: 10px; display: flex; gap: 8px; z-index: 10;">
                <button onclick="this.closest('[id^=\\'table-image-modal-\\']').querySelector('[data-scale-up]').click()" style="padding: 6px 12px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer;">🔍+</button>
                <button onclick="this.closest('[id^=\\'table-image-modal-\\']').querySelector('[data-scale-down]').click()" style="padding: 6px 12px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer;">🔍-</button>
                <button onclick="this.closest('[id^=\\'table-image-modal-\\']').querySelector('[data-reset]').click()" style="padding: 6px 12px; background: #6c757d; color: white; border: none; border-radius: 4px; cursor: pointer;">↺ Reset</button>
                <button onclick="this.closest('[id^=\\'table-image-modal-\\']').remove()" style="padding: 6px 12px; background: #dc3545; color: white; border: none; border-radius: 4px; cursor: pointer;">✕ Close</button>
            </div>
            <div data-table-content>${buildTableHtml()}</div>`;
            const scaleUp = document.createElement('button');
            scaleUp.setAttribute('data-scale-up', '');
            scaleUp.style.display = 'none';
            scaleUp.onclick = () => { scale = Math.min(scale * 1.2, 5); container.style.transform = `scale(${scale})`; };
            const scaleDown = document.createElement('button');
            scaleDown.setAttribute('data-scale-down', '');
            scaleDown.style.display = 'none';
            scaleDown.onclick = () => { scale = Math.max(scale / 1.2, 0.2); container.style.transform = `scale(${scale})`; };
            const reset = document.createElement('button');
            reset.setAttribute('data-reset', '');
            reset.style.display = 'none';
            reset.onclick = () => { scale = 1; offsetX = 0; offsetY = 0; container.style.transform = 'scale(1)'; container.style.left = ''; container.style.top = ''; };
            container.appendChild(scaleUp);
            container.appendChild(scaleDown);
            container.appendChild(reset);
            modal.appendChild(container);
            modal.onmousedown = (e) => { if (e.target === modal) { isDragging = true; startX = e.clientX - offsetX; startY = e.clientY - offsetY; } };
            modal.onmousemove = (e) => { if (isDragging && e.target === modal) { offsetX = e.clientX - startX; offsetY = e.clientY - startY; container.style.left = offsetX + 'px'; container.style.top = offsetY + 'px'; } };
            modal.onmouseup = () => { isDragging = false; };
            modal.onwheel = (e) => { e.preventDefault(); scale = Math.max(0.2, Math.min(5, scale - e.deltaY * 0.001)); container.style.transform = `scale(${scale})`; };
            document.body.appendChild(modal);
        }
        
        function downloadIntegrationTableAsCsv(downloadId) {
            const t = window.__integrationTables[downloadId];
            if (!t || !t.columns || !t.data) return;
            const escape = (v) => {
                const s = (v == null || v === '') ? '' : String(v);
                return /[",\\n\\r]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
            };
            const header = t.columns.map(escape).join(',');
            const rows = t.data.map(row => row.map(escape).join(','));
            const csv = [header].concat(rows).join('\\r\\n');
            const blob = new Blob(['\\ufeff' + csv], { type: 'text/csv;charset=utf-8' });
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = (downloadId || 'integrated') + '.csv';
            a.click();
            URL.revokeObjectURL(a.href);
        }
        
        function buildTableInfo(table, maxColumns) {
            const totalRows = (table.data || []).length;
            const allCols = table.columns || [];
            const colLimit = maxColumns == null ? allCols.length : Math.min(maxColumns, allCols.length);
            const cols = allCols.slice(0, colLimit);
            return cols.map((col, i) => {
                let nonNull = 0;
                let hasNum = true;
                for (let r = 0; r < table.data.length; r++) {
                    const row = table.data[r] || [];
                    const v = row[i];
                    const vs = normalizeIntegrationText(v);
                    const isNaNString = vs.trim().toLowerCase() === 'nan';
                    if (vs !== '' && !isNaNString) {
                        nonNull++;
                        if (hasNum && isNaN(Number(vs))) hasNum = false;
                    }
                }
                const pct = totalRows ? (nonNull / totalRows * 100).toFixed(1) : '0';
                const nonNullPct = pct + '%';
                const dtype = nonNull === 0 ? 'object' : (hasNum ? 'number' : 'object');
                return { col, nonNullPct, dtype };
            });
        }
        function maybeFixMojibakeUtf8(s) {
            if (!s || typeof s !== 'string') return s;
            const likelyMojibake = /[ÃÂâãäå]/.test(s);
            if (!likelyMojibake) return s;
            for (let i = 0; i < s.length; i++) {
                if (s.charCodeAt(i) > 255) return s;
            }
            try {
                const bytes = new Uint8Array(s.length);
                for (let i = 0; i < s.length; i++) bytes[i] = s.charCodeAt(i) & 0xff;
                const fixed = new TextDecoder('utf-8').decode(bytes);
                const srcBad = (s.match(/[ÃÂâãäå]/g) || []).length;
                const fixedBad = (fixed.match(/[ÃÂâãäå]/g) || []).length;
                return fixedBad < srcBad ? fixed : s;
            } catch (_) {
                return s;
            }
        }
        function normalizeIntegrationText(v) {
            if (v == null || v === '') return '';
            const asStr = String(v);
            if (asStr.trim().toLowerCase() === 'nan') return '';
            return maybeFixMojibakeUtf8(asStr);
        }
        function formatIntegrationCell(cell) {
            const asStr = normalizeIntegrationText(cell);
            if (asStr === '') return '';
            if (asStr.length > INTEGRATION_MAX_CELL_CHARS) {
                return asStr.slice(0, INTEGRATION_MAX_CELL_CHARS - 1) + '…';
            }
            return asStr;
        }

        function renderIntegrationTable(table, stats, options) {
            const { title = 'Integration', successColor = '#28a745', extraHtml = '', savedPath = '', downloadId = '', omitSectionTitle = true } = options || {};
            const titleHtml = omitSectionTitle ? '' : `<h4 style="margin-top: 0; color: ${successColor};">✅ ${title}</h4>`;
            if (!table || (stats && stats.output_rows === 0)) {
                return `<div style="padding: 15px; background: #fff; border-radius: 4px; border: 1px solid #dee2e6;">
                    ${titleHtml}
                    <div style="margin-bottom: 15px;">${(stats && stats.output_columns === 0) ? '⚠️ No common columns. Intersection result is empty.' : '⚠️ No common rows. Intersection result is empty.'}</div></div>`;
            }
            if (downloadId) window.__integrationTables[downloadId] = table;
            const totalRows = (table.data || []).length;
            const totalCols = (table.columns || []).length;
            const displayRowCount = Math.min(totalRows, DISPLAY_MAX_ROWS);
            const displayColCount = Math.min(totalCols, DISPLAY_MAX_COLS);
            const displayCols = (table.columns || []).slice(0, displayColCount);
            const displayRows = (table.data || []).slice(0, displayRowCount).map((row) => (row || []).slice(0, displayColCount));
            const rowsTruncated = totalRows > DISPLAY_MAX_ROWS;
            const colsTruncated = totalCols > DISPLAY_MAX_COLS;
            const previewDims = `${displayRowCount}×${displayColCount}`;
            const fullDims = `${totalRows}×${totalCols}`;
            const previewLabel = rowsTruncated || colsTruncated
                ? `preview ${previewDims} of ${fullDims} (download CSV for full table)`
                : `all ${previewDims}`;
            const footer = [];
            if (savedPath) footer.push(`<span style="font-size: 12px; color: #666;">Saved to: ${integrationSavedPathLink(savedPath)}</span>`);
            footer.push(`<button type="button" onclick="downloadIntegrationTableAsCsv('${downloadId}')" style="margin-left: 10px; padding: 6px 12px; font-size: 13px; background: #28a745; color: white; border: none; border-radius: 6px; cursor: pointer;">📥 Download full CSV (${totalRows} rows)</button>`);
            const infoRows = buildTableInfo(table, displayColCount);
            const infoHeaderRow = '<th style="border:1px solid #dee2e6;padding:4px 8px; background: #e9ecef; font-size: 11px;"> </th>' + infoRows.map(({ col }) => `<th style="border:1px solid #dee2e6;padding:4px 8px; background: #e9ecef; font-size: 11px; white-space: nowrap;">${normalizeIntegrationText(col)}</th>`).join('');
            const infoNonNullRow = '<td style="border:1px solid #dee2e6;padding:4px 8px; font-size: 11px; font-weight: 600;">Non-Null %</td>' + infoRows.map(({ nonNullPct }) => `<td style="border:1px solid #dee2e6;padding:4px 8px; font-size: 11px;">${nonNullPct}</td>`).join('');
            const infoDtypeRow = '<td style="border:1px solid #dee2e6;padding:4px 8px; font-size: 11px; font-weight: 600;">Dtype</td>' + infoRows.map(({ dtype }) => `<td style="border:1px solid #dee2e6;padding:4px 8px; font-size: 11px;">${dtype}</td>`).join('');
            let html = `<div style="padding: 15px; background: #fff; border-radius: 4px; border: 1px solid #dee2e6;">
                ${titleHtml}
                <div style="margin-bottom: 10px; font-size: 13px;">Input: ${stats.input_tables} tables, ${stats.input_rows} rows → Output: ${stats.output_rows} rows, ${stats.output_columns} cols (${previewLabel})</div>
                ${extraHtml}
                <div style="position: relative;">
                    <div style="${INTEGRATION_TABLE_VIEWPORT_STYLE}" id="table-viewport-${downloadId}" title="Integrated table preview (max ${DISPLAY_MAX_ROWS} rows × ${DISPLAY_MAX_COLS} cols); download CSV for full table">
                        <table style="width: max-content; min-width: 100%; border-collapse: collapse; font-size: 12px;">
                            <thead><tr style="background: #f8f9fa; position: sticky; top: 0; z-index: 10;">
                                ${displayCols.map(col => `<th style="border: 1px solid #dee2e6; padding: 6px; text-align: left; background: #f8f9fa; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${normalizeIntegrationText(col)}</th>`).join('')}
                            </tr></thead>
                            <tbody>
                                ${displayRows.map(row => `<tr>${row.map(cell => `<td style="border: 1px solid #dee2e6; padding: 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${formatIntegrationCell(cell)}</td>`).join('')}</tr>`).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
                <div style="margin-top: 12px;">
                    <div style="font-size: 13px; color: #495057; margin-bottom: 6px;">📋 Table info</div>
                    <div style="overflow-x: auto;">
                        <table style="border-collapse: collapse; font-size: 11px;">
                            <thead><tr>${infoHeaderRow}</tr></thead>
                            <tbody>
                                <tr>${infoNonNullRow}</tr>
                                <tr>${infoDtypeRow}</tr>
                            </tbody>
                        </table>
                    </div>
                </div>
                <div style="margin-top: 10px; display: flex; align-items: center; flex-wrap: wrap; gap: 8px;">
                    ${footer.join('')}
                    <button type="button" onclick="showTableAsImage('${downloadId}', '${downloadId}')" style="margin-left: 8px; padding: 6px 12px; font-size: 13px; background: #17a2b8; color: white; border: none; border-radius: 6px; cursor: pointer;">🖼️ View as Image (Drag & Zoom)</button>
                </div>
            </div>`;
            return html;
        }
        
        async function runBothIntegrations(jobId) {
            const integrationType = (document.getElementById('integration_type') || {}).value || 'alite';
            const retrievalMode = (document.getElementById('integration_q2m_mode') || {}).value || 'dense';
            const k = parseInt((document.getElementById('integration_k') || {}).value, 10) || 10;
            const maxModels = parseInt((document.getElementById('integration_max_models') || {}).value, 10) || 10;
            const searchType = (document.getElementById('integration_search_type') || {}).value || 'single_column';
            
            const btn = document.getElementById('integrationRunBothBtn');
            const container = document.getElementById('integrationResultsContainer');
            const leftDiv = document.getElementById('integrationModelSearchResults');
            const rightDiv = document.getElementById('integrationResults');
            if (!btn || !container || !leftDiv || !rightDiv) return;
            
            btn.disabled = true;
            btn.textContent = '⏳ Integrating...';
            container.style.display = 'block';
            leftDiv.innerHTML = '<div style="padding: 8px; background: #f8f9fa; border-radius: 4px; font-size: 12px;">⏳ Waiting for 1 Query2Card Results integration...</div>';
            rightDiv.innerHTML = '<div style="padding: 8px; background: #f8f9fa; border-radius: 4px; font-size: 12px;">⏳ Waiting for 2 Query2Tab2Card Results integration...</div>';
            
            try {
                const tablesSource = (document.getElementById('integration_tables_source') || {}).value || 'intermediate';
                const modelReq = fetch('{{BACKEND_URL}}/api/integrate-model-search', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ job_id: jobId, integration_type: integrationType, query2modelcard_retrieval_mode: retrievalMode, k, max_models: maxModels })
                }).then(r => r.json());
                const tableReq = fetch('{{BACKEND_URL}}/api/integrate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ job_id: jobId, search_type: searchType, integration_type: integrationType, k, max_models: maxModels, tables_source: tablesSource })
                }).then(r => r.json());
                const [modelRes, tableRes] = await Promise.all([modelReq, tableReq]);

                if (modelRes.status === 'success') {
                    const stats = modelRes.stats || {};
                    const table = modelRes.integrated_table;
                    const modelIds = modelRes.models_with_tables || [];
                    let extra = '';
                    if (modelIds.length > 0) {
                        const modelToTables = modelRes.model_to_table_paths || {};
                        const linesHtml = formatQuery2CardModelTableLinesHtml(modelIds, modelToTables);
                        const modelsCount = stats.models_with_tables != null ? stats.models_with_tables : modelIds.length;
                        const tablesCount = stats.total_unique_tables != null ? stats.total_unique_tables : (table && table.data ? table.data.length : 0);
                        extra = `<div style="margin-bottom: 10px; padding: 8px; background: #e7f3ff; border-radius: 4px; font-size: 12px;">
                            <details style="margin: 0;">
                                <summary style="cursor: pointer; list-style: none; outline: none;">
                                    <span style="font-weight: 600;">${INT_MODEL_IDS_C2C}</span>
                                    <span style="color:#004085;">(${modelsCount} models, ${tablesCount} rows)</span>
                                </summary>
                                <div style="margin-top: 6px; font-size: 11px;">
                                    ${linesHtml}
                                </div>
                            </details>
                        </div>`;
                    } else {
                        extra = '<div style="margin-bottom: 10px; padding: 8px; background: #f8f9fa; border-radius: 4px; font-size: 12px; color: #6c757d;">Model IDs (1 Query2Card Results): — (none or not available)</div>';
                    }
                    leftDiv.innerHTML = renderIntegrationTable(table, stats, { title: INT_TITLE_C2C_HTML, successColor: '#007bff', extraHtml: extra, savedPath: modelRes.saved_path || '', downloadId: 'model-search' });
                } else {
                    let debugHtml = '';
                    if (modelRes.model_to_table_paths) {
                        const mtp = modelRes.model_to_table_paths || {};
                        const mids = (Array.isArray(modelRes.model_ids) && modelRes.model_ids.length) ? modelRes.model_ids : Object.keys(mtp);
                        const linesDbg = formatQuery2CardModelTableLinesHtml(mids.slice(0, 12), mtp);
                        debugHtml = `<div style="margin-top: 10px; padding: 8px; border: 1px solid #f1b0b7; background: #fff6f7; border-radius: 4px;">
                            <div style="font-size: 11px; color:#b02a37;"><strong>Debug: model → related table</strong> (first 12 models)</div>
                            <div style="margin-top: 6px; font-size: 11px;">${linesDbg}</div>
                        </div>`;
                    }
                    leftDiv.innerHTML = `<div style="padding: 10px; border-radius: 4px; border: 1px solid #dc3545; color: #dc3545; font-size: 12px;">
                        ❌ ${modelRes.message || 'Integration failed'}
                        ${debugHtml}
                    </div>`;
                }
                initTablePanZoom(leftDiv);

                if (tableRes.status === 'success') {
                    const stats = tableRes.stats || {};
                    const groupedTables = getGroupedIntegratedTables(tableRes);
                    let tableExtra = '';
                    const tableModelIds = tableRes.models_with_tables || [];
                    if (tableModelIds.length > 0) {
                        const modelToTables = tableRes.model_to_table_paths || {};
                        const tplist = tableRes.table_paths || [];
                        let rtRows = tableRes.retrieved_table_model_rows;
                        if (!Array.isArray(rtRows) || !rtRows.length) {
                            rtRows = buildRetrievedTableModelRowsFallback(modelToTables, tplist);
                        }
                        const tsTr = tableRes.tables_source || (tableRes.stats && tableRes.stats.tables_source) || tablesSource;
                        const traceHtml = formatQuery2Tab2CardTraceHtml(tableRes.query_tables || [], rtRows, {
                            tablesSource: tsTr,
                            modelToTablePaths: tableRes.model_to_table_paths || {},
                            rankedModelIds: tableModelIds,
                            pipelineTrace: tableRes.pipeline_trace,
                            tab2tabTraceRows: tableRes.tab2tab_trace_rows,
                            afterModelCapTraceRows: tableRes.after_model_cap_trace_rows,
                        });
                        const modelsCount = (tableRes.stats && tableRes.stats.models_with_tables != null) ? tableRes.stats.models_with_tables : tableModelIds.length;
                        tableExtra = `<div style="margin-bottom: 10px; padding: 8px; background: #d4edda; border-radius: 4px; font-size: 12px;">
                            <details style="margin: 0;">
                                <summary style="cursor: pointer; list-style: none; outline: none;">
                                    <span style="font-weight: 600;">${INT_MODEL_IDS_C2T2C}</span>
                                    <span style="color:#155724;">(${modelsCount} models, ${describeTableSearchCount(tableRes, tableRes.stats || {})})</span>
                                </summary>
                                <div style="margin-top: 6px; font-size: 11px;">
                                    ${traceHtml}
                                </div>
                            </details>
                        </div>`;
                    } else {
                        tableExtra = '<div style="margin-bottom: 10px; padding: 8px; background: #f8f9fa; border-radius: 4px; font-size: 12px; color: #6c757d;">Model IDs (2 Query2Tab2Card Results): — (none or not available)</div>';
                    }
                    rightDiv.innerHTML = renderGroupedIntegrationTables(groupedTables, { title: INT_TITLE_C2T2C_HTML, successColor: '#28a745', extraHtml: tableExtra, downloadPrefix: 'table-search' });
                } else {
                    let debugHtml = '';
                    if (tableRes.model_to_table_paths) {
                        const mtp = tableRes.model_to_table_paths || {};
                        const tplist = tableRes.table_paths || [];
                        let rtRows = tableRes.retrieved_table_model_rows;
                        if (!Array.isArray(rtRows) || !rtRows.length) {
                            rtRows = buildRetrievedTableModelRowsFallback(mtp, tplist);
                        }
                        const tsDbg = tableRes.tables_source || (tableRes.stats && tableRes.stats.tables_source) || tablesSource;
                        const dbgMids = tableRes.models_with_tables || [];
                        const traceDbg = formatQuery2Tab2CardTraceHtml(tableRes.query_tables || [], rtRows, {
                            tablesSource: tsDbg,
                            modelToTablePaths: mtp,
                            rankedModelIds: dbgMids,
                            pipelineTrace: tableRes.pipeline_trace,
                            tab2tabTraceRows: tableRes.tab2tab_trace_rows,
                            afterModelCapTraceRows: tableRes.after_model_cap_trace_rows,
                        });
                        debugHtml = `<div style="margin-top: 10px; padding: 8px; border: 1px solid #f1b0b7; background: #fff6f7; border-radius: 4px;">
                            <div style="font-size: 11px; color:#b02a37;"><strong>Debug: query / retrieved tables → models</strong></div>
                            <div style="margin-top: 6px; font-size: 11px;">${traceDbg}</div>
                        </div>`;
                    }
                    rightDiv.innerHTML = `<div style="padding: 10px; border-radius: 4px; border: 1px solid #dc3545; color: #dc3545; font-size: 12px;">
                        ❌ ${tableRes.message || 'Integration failed'}
                        ${debugHtml}
                    </div>`;
                }
                initTablePanZoom(rightDiv);
                const modelKey = getModelSearchKey(integrationType, retrievalMode);
                const tableKey = getTableSearchKey(integrationType, searchType, tablesSource);
                const hasModelRun = modelRes.status === 'success' || modelRes.status === 'no_result';
                const hasTableRun = tableRes.status === 'success' || tableRes.status === 'no_result';
                if (hasModelRun || hasTableRun) {
                    if (hasModelRun) {
                        let runs = window.__modelSearchRuns || [];
                        const mPayload = { key: modelKey, integration_type: integrationType, k, max_models: maxModels, ...modelRes };
                        const idx = runs.findIndex(r => modelRunMatchesKey(r, modelKey));
                        if (idx >= 0) runs[idx] = mPayload; else runs = [...runs, mPayload];
                        window.__modelSearchRuns = runs;
                    }
                    if (hasTableRun) {
                        let runs = window.__tableSearchRuns || [];
                        const tPayload = { key: tableKey, integration_type: integrationType, search_type: searchType, tables_source: tablesSource, k, max_models: maxModels, ...tableRes };
                        const idx = runs.findIndex(r => (r.key || getTableSearchKey(r.integration_type, r.search_type, r.tables_source)) === tableKey);
                        if (idx >= 0) runs[idx] = tPayload; else runs = [...runs, tPayload];
                        window.__tableSearchRuns = runs;
                    }
                    syncBothIntegrationDisplays();
                }
            } catch (err) {
                leftDiv.innerHTML = `<div style="padding: 10px; border: 1px solid #dc3545; color: #dc3545; font-size: 12px;">❌ ${err.message}</div>`;
                rightDiv.innerHTML = `<div style="padding: 10px; border: 1px solid #dc3545; color: #dc3545; font-size: 12px;">❌ ${err.message}</div>`;
            } finally {
                btn.disabled = false;
                btn.textContent = 'Integrated';
            }
        }

        async function runAllTableIntegrations(jobId) {
            const k = parseInt((document.getElementById('integration_k') || {}).value, 10) || 10;
            const maxModels = parseInt((document.getElementById('integration_max_models') || {}).value, 10) || 10;
            // For run-all we fix tablesSource to "intermediate" to avoid very heavy all_from_modelcards scans.
            const tablesSource = 'intermediate';
            const btn = document.getElementById('integrationRunAllTableBtn');
            if (!btn) return;
            const integrationTypes = ['alite'];
            const tableSearchTypes = ['single_column','keyword','multi_column','unionable','complex','correlation','imputation','augmentation','dependent_data','feature_for_ml','multi_column_collinearity','negative_example'];
            btn.disabled = true;
            const originalText = btn.textContent;
            btn.textContent = '⏳ Running all...';
            try {
                const tasks = [];
                // All Model Search integrations: one per integration type (neighbors = dense from search_results)
                for (const it of integrationTypes) {
                    tasks.push(fetch('{{BACKEND_URL}}/api/integrate-model-search', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({ job_id: jobId, integration_type: it, k, max_models: maxModels })
                    }).then(r => r.json()).catch(() => null));
                }
                // All Table Search integrations: all integrationTypes × tableSearchTypes, tablesSource fixed
                for (const it of integrationTypes) {
                    for (const st of tableSearchTypes) {
                        tasks.push(fetch('{{BACKEND_URL}}/api/integrate', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({ job_id: jobId, search_type: st, integration_type: it, k, max_models: maxModels, tables_source: tablesSource })
                        }).then(r => r.json()).catch(() => null));
                    }
                }
                await Promise.allSettled(tasks);
                btn.textContent = '✅ All done';
                // Load saved runs from server so switching dropdowns shows them without reload
                try {
                    const r = await fetch('{{BACKEND_URL}}/api/integration-runs/' + jobId);
                    const apiData = await r.json();
                    if (apiData.status === 'success') {
                        const modelRuns = apiData.model_search_runs || [];
                        const tableRuns = apiData.table_search_runs || [];
                        window.__modelSearchRuns = modelRuns;
                        window.__tableSearchRuns = tableRuns;
                        syncBothIntegrationDisplays();
                    }
                } catch (e) { console.warn('Run all: could not refresh integration runs', e); }
                setTimeout(() => { btn.textContent = originalText; }, 2000);
            } catch (e) {
                console.error('runAllTableIntegrations error', e);
                btn.textContent = '❌ Error';
                setTimeout(() => { btn.textContent = originalText; }, 2000);
            } finally {
                btn.disabled = false;
            }
        }
        
        async function runIntegration(jobId) {
            const searchType = (document.getElementById('integration_search_type') || {}).value || 'single_column';
            const integrationType = (document.getElementById('integration_type') || {}).value || 'alite';
            const k = parseInt((document.getElementById('integration_k') || {}).value, 10) || 10;
            const maxModels = parseInt((document.getElementById('integration_max_models') || {}).value, 10) || 10;
            
            const resultsDiv = document.getElementById('integrationResults');
            if (!resultsDiv) return;
            resultsDiv.innerHTML = '<div style="padding: 15px;">⏳ Running integration...</div>';
            try {
                const tsSingle = (document.getElementById('integration_tables_source') || {}).value || 'intermediate';
                const response = await fetch('{{BACKEND_URL}}/api/integrate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ job_id: jobId, search_type: searchType, integration_type: integrationType, k, max_models: maxModels, tables_source: tsSingle })
                });
                const data = await response.json();
                if (data.status === 'success') {
                    const groupedTables = getGroupedIntegratedTables(data);
                    const modelIds = data.models_with_tables || [];
                    let extra = '';
                    if (modelIds.length > 0) {
                        const mtp = data.model_to_table_paths || {};
                        const tplist = data.table_paths || [];
                        let rtRows = data.retrieved_table_model_rows;
                        if (!Array.isArray(rtRows) || !rtRows.length) {
                            rtRows = buildRetrievedTableModelRowsFallback(mtp, tplist);
                        }
                        const tsD = data.tables_source || (data.stats && data.stats.tables_source) || tsSingle;
                        const traceHtml = formatQuery2Tab2CardTraceHtml(data.query_tables || [], rtRows, {
                            tablesSource: tsD,
                            modelToTablePaths: data.model_to_table_paths || {},
                            rankedModelIds: modelIds,
                            pipelineTrace: data.pipeline_trace,
                            tab2tabTraceRows: data.tab2tab_trace_rows,
                            afterModelCapTraceRows: data.after_model_cap_trace_rows,
                        });
                        extra = `<div style="margin-bottom: 10px; padding: 8px; background: #d4edda; border-radius: 4px; font-size: 12px;"><details style="margin:0;"><summary style="cursor:pointer;font-weight:600;">${INT_MODEL_IDS_C2T2C}</summary><div style="margin-top:6px;font-size:11px;">${traceHtml}</div></details></div>`;
                    }
                    resultsDiv.innerHTML = renderGroupedIntegrationTables(groupedTables, { title: INT_TITLE_C2T2C_HTML, successColor: '#28a745', extraHtml: extra, downloadPrefix: 'table-search-single' });
                    initTablePanZoom(resultsDiv);
                } else {
                    resultsDiv.innerHTML = `<div style="padding: 15px; border: 1px solid #dc3545; color: #dc3545;">❌ ${data.message || 'Unknown error'}</div>`;
                }
            } catch (error) {
                resultsDiv.innerHTML = `<div style="padding: 15px; border: 1px solid #dc3545; color: #dc3545;">❌ ${error.message}</div>`;
            }
        }
        
        async function runModelSearchIntegration(jobId) {
            const integrationType = (document.getElementById('integration_type') || {}).value || 'alite';
            const retrievalMode = (document.getElementById('integration_q2m_mode') || {}).value || 'dense';
            const k = parseInt((document.getElementById('integration_k') || {}).value, 10) || 10;
            const maxModels = parseInt((document.getElementById('integration_max_models') || {}).value, 10) || 10;
            const resultsDiv = document.getElementById('integrationModelSearchResults');
            if (!resultsDiv) return;
            resultsDiv.style.display = 'block';
            resultsDiv.innerHTML = '<div style="padding: 15px;">⏳ Running integration...</div>';
            try {
                const response = await fetch('{{BACKEND_URL}}/api/integrate-model-search', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ job_id: jobId, integration_type: integrationType, query2modelcard_retrieval_mode: retrievalMode, k, max_models: maxModels })
                });
                const data = await response.json();
                if (data.status === 'success') {
                    const stats = data.stats || {};
                    const table = data.integrated_table;
                    const modelIds = data.models_with_tables || [];
                    let extra = '';
                    if (modelIds.length > 0) {
                        const mtp = data.model_to_table_paths || {};
                        const linesHtml = formatQuery2CardModelTableLinesHtml(modelIds, mtp);
                        extra = `<div style="margin-bottom: 10px; padding: 8px; background: #e7f3ff; border-radius: 4px; font-size: 12px;"><details style="margin:0;"><summary style="cursor:pointer;font-weight:600;">${INT_MODEL_IDS_C2C}</summary><div style="margin-top:6px;font-size:11px;">${linesHtml}</div></details></div>`;
                    } else {
                        extra = '<div style="margin-bottom: 10px; padding: 8px; background: #f8f9fa; border-radius: 4px; font-size: 12px; color: #6c757d;">Model IDs (1 Query2Card Results): — (none or not available)</div>';
                    }
                    resultsDiv.innerHTML = renderIntegrationTable(table, stats, { title: INT_TITLE_C2C_HTML, successColor: '#007bff', extraHtml: extra, savedPath: data.saved_path || '', downloadId: 'model-search-single' });
                    initTablePanZoom(resultsDiv);
                } else {
                    resultsDiv.innerHTML = `<div style="padding: 15px; border: 1px solid #dc3545; color: #dc3545;">❌ ${data.message || 'Unknown error'}</div>`;
                }
            } catch (error) {
                resultsDiv.innerHTML = `<div style="padding: 15px; border: 1px solid #dc3545; color: #dc3545;">❌ ${error.message}</div>`;
            }
        }
        
        let fakeResponseFile = null;
        let qaFakeResponseFile = null;
        
        function toggleQAMode() {}
        
        function handleQAFakeFileSelect() {
            const fileInput = document.getElementById('qa_fake_file');
            const fileNameSpan = document.getElementById('qa_fake_file_name');
            if (fileInput && fileInput.files.length > 0) {
                qaFakeResponseFile = fileInput.files[0];
                if (fileNameSpan) {
                    fileNameSpan.textContent = qaFakeResponseFile.name;
                    fileNameSpan.style.color = '#28a745';
                }
            }
        }
        
        function toggleEvaluationMode() {}
        
        function handleFakeFileSelect() {
            const fileInput = document.getElementById('evaluation_fake_file2');
            const fileNameSpan = document.getElementById('fake_file_name2');
            if (fileInput && fileInput.files.length > 0) {
                fakeResponseFile = fileInput.files[0];
                if (fileNameSpan) {
                    fileNameSpan.textContent = fakeResponseFile.name;
                    fileNameSpan.style.color = '#28a745';
                }
            }
        }
        
        function displayEvaluationResults(eval_result, resultsDiv, table1Data, table2Data) {
            if (!resultsDiv) return;
            resultsDiv.style.display = 'block';
            const escTpl = (s) => String(s == null ? '' : s).replace(/\\\\/g, '\\\\\\\\').replace(/`/g, '\\\\`').replace(/\\\$/g, '\\\\$').replace(/\{/g, '\\\\{').replace(/\}/g, '\\\\}');
            const totalScore = eval_result.total_quality_score || {};
            const comparisonScore = eval_result.comparison_score || {};
            const modelSearchScore = (totalScore.model_search != null ? totalScore.model_search : (comparisonScore.model_search_quality != null ? comparisonScore.model_search_quality : (eval_result.model_search_quality != null ? eval_result.model_search_quality : 'N/A')));
            const tableSearchScore = (totalScore.table_search != null ? totalScore.table_search : (comparisonScore.table_search_quality != null ? comparisonScore.table_search_quality : (eval_result.table_search_quality != null ? eval_result.table_search_quality : 'N/A')));
            const winner = totalScore.winner || comparisonScore.winner || eval_result.winner || 'N/A';
            const subScores = eval_result.sub_scores || [];
            const qualityAnalysis = eval_result.quality_analysis || {};
            const modelSearchAnalysis = qualityAnalysis.model_search || {};
            const tableSearchAnalysis = qualityAnalysis.table_search || {};
            const keyDiffs = eval_result.key_differences || [];
            const evidenceForDiffs = escTpl(eval_result.evidence_for_differences || '');
            // Calculate average of sub-scores for comparison
            let avgModelSearch = null, avgTableSearch = null;
            if (subScores.length > 0) {
                const modelSum = subScores.reduce((sum, ss) => sum + (ss.model_search != null ? ss.model_search : 0), 0);
                const tableSum = subScores.reduce((sum, ss) => sum + (ss.table_search != null ? ss.table_search : 0), 0);
                avgModelSearch = Math.round(modelSum / subScores.length);
                avgTableSearch = Math.round(tableSum / subScores.length);
            }
            let html = `
                <div style="padding: 15px; background: #fff; border-radius: 4px; border: 1px solid #dee2e6;">
                    <h4 style="margin-top: 0; color: #856404; margin-bottom: 15px;">📊 Quality Comparison (on user question)</h4>
                    <div style="margin-bottom: 20px; padding: 15px; background: #f8f9fa; border-radius: 4px;">
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                            <div style="padding: 15px; background: ${winner === 'model_search' ? '#d4edda' : '#e7f3ff'}; border-radius: 4px; border: 2px solid ${winner === 'model_search' ? '#28a745' : '#007bff'};">
                                <div style="font-size: 14px; color: #666;">1 Query2Card Results</div>
                                <div style="font-size: 28px; font-weight: bold; color: ${winner === 'model_search' ? '#28a745' : '#004085'};">${modelSearchScore}/100</div>
                                ${winner === 'model_search' ? '<div style="font-size: 11px; color: #28a745;">🏆 Winner</div>' : ''}
                                ${avgModelSearch != null ? `<div style="font-size: 10px; color: #666; margin-top: 4px;">Avg of sub-scores: ${avgModelSearch}/100</div>` : ''}
                            </div>
                            <div style="padding: 15px; background: ${winner === 'table_search' ? '#d4edda' : '#fff3cd'}; border-radius: 4px; border: 2px solid ${winner === 'table_search' ? '#28a745' : '#ffc107'};">
                                <div style="font-size: 14px; color: #666;">2 Query2Tab2Card Results</div>
                                <div style="font-size: 28px; font-weight: bold; color: ${winner === 'table_search' ? '#28a745' : '#856404'};">${tableSearchScore}/100</div>
                                ${winner === 'table_search' ? '<div style="font-size: 11px; color: #28a745;">🏆 Winner</div>' : ''}
                                ${avgTableSearch != null ? `<div style="font-size: 10px; color: #666; margin-top: 4px;">Avg of sub-scores: ${avgTableSearch}/100</div>` : ''}
                            </div>
                        </div>
                        ${subScores.length > 0 ? `<div style="margin-top: 10px; padding: 8px; background: #fff; border-radius: 4px; font-size: 11px; color: #666; text-align: center;">Total Quality Score is composed of three sub-scores: <strong>Relevance</strong>, <strong>Coverage</strong>, and <strong>Diversity</strong> (shown below).</div>` : ''}
                    </div>
                    ${subScores.length > 0 ? `
                    <div style="margin-bottom: 20px;">
                        <h5 style="margin: 0 0 10px 0; color: #856404; font-size: 13px;">Sub-scores (compose the Total Quality Score above)</h5>
                        <div style="display: flex; flex-direction: column; gap: 10px;">
                            ${subScores.map(ss => `
                                <div style="padding: 10px; background: #f8f9fa; border-radius: 4px; border-left: 4px solid #ffc107;">
                                    <div style="font-weight: 600; font-size: 12px;">${escTpl(ss.name)}</div>
                                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 12px; margin-top: 4px;">
                                        <span>1 Query2Card: <strong>${ss.model_search != null ? ss.model_search : '–'}/100</strong></span>
                                        <span>2 Query2Tab2Card: <strong>${ss.table_search != null ? ss.table_search : '–'}/100</strong></span>
                                    </div>
                                    ${ss.evidence ? '<div style="font-size: 11px; color: #666; margin-top: 6px;">Evidence: ' + escTpl(ss.evidence) + '</div>' : ''}
                                </div>
                            `).join('')}
                        </div>
                    </div>
                    ` : ''}
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 20px;">
                        <div style="padding: 15px; background: #e7f3ff; border-radius: 4px; border-left: 4px solid #007bff;">
                            <h5 style="margin-top: 0; color: #004085; font-size: 13px;">1 Query2Card Results</h5>
                            ${modelSearchAnalysis.strengths && modelSearchAnalysis.strengths.length > 0 ? '<div style="margin-top: 8px;"><strong style="font-size: 12px; color: #28a745;">Strengths:</strong><ul style="font-size: 11px; margin: 4px 0 0 0; padding-left: 20px;">' + modelSearchAnalysis.strengths.map(s => '<li>' + escTpl(s) + '</li>').join('') + '</ul></div>' : ''}
                            ${modelSearchAnalysis.weaknesses && modelSearchAnalysis.weaknesses.length > 0 ? '<div style="margin-top: 8px;"><strong style="font-size: 12px; color: #dc3545;">Weaknesses:</strong><ul style="font-size: 11px; margin: 4px 0 0 0; padding-left: 20px;">' + modelSearchAnalysis.weaknesses.map(w => '<li>' + escTpl(w) + '</li>').join('') + '</ul></div>' : ''}
                        </div>
                        <div style="padding: 15px; background: #fff3cd; border-radius: 4px; border-left: 4px solid #ffc107;">
                            <h5 style="margin-top: 0; color: #856404; font-size: 13px;">2 Query2Tab2Card Results</h5>
                            ${tableSearchAnalysis.strengths && tableSearchAnalysis.strengths.length > 0 ? '<div style="margin-top: 8px;"><strong style="font-size: 12px; color: #28a745;">Strengths:</strong><ul style="font-size: 11px; margin: 4px 0 0 0; padding-left: 20px;">' + tableSearchAnalysis.strengths.map(s => '<li>' + escTpl(s) + '</li>').join('') + '</ul></div>' : ''}
                            ${tableSearchAnalysis.weaknesses && tableSearchAnalysis.weaknesses.length > 0 ? '<div style="margin-top: 8px;"><strong style="font-size: 12px; color: #dc3545;">Weaknesses:</strong><ul style="font-size: 11px; margin: 4px 0 0 0; padding-left: 20px;">' + tableSearchAnalysis.weaknesses.map(w => '<li>' + escTpl(w) + '</li>').join('') + '</ul></div>' : ''}
                        </div>
                    </div>
                    ${keyDiffs.length > 0 ? '<div style="margin-top: 15px; padding: 12px; background: #f8f9fa; border-radius: 4px;"><strong style="font-size: 13px;">Key differences</strong><ul style="margin: 8px 0 0 0; padding-left: 20px; font-size: 13px;">' + keyDiffs.map(d => '<li>' + escTpl(d) + '</li>').join('') + '</ul></div>' : ''}
                    ${evidenceForDiffs ? '<div style="margin-top: 12px; padding: 12px; background: #e7f3ff; border-radius: 4px; border-left: 4px solid #007bff;"><strong style="font-size: 12px;">Evidence for the above</strong><p style="margin: 6px 0 0 0; font-size: 12px; line-height: 1.5;">' + evidenceForDiffs + '</p></div>' : ''}
                    ${eval_result.source ? '<div style="margin-top: 10px; font-size: 11px; color: #999;">Source: ' + escTpl(eval_result.source) + '</div>' : ''}
                </div>
            `;
            resultsDiv.innerHTML = html;
        }
        
        async function runEvaluationAndQABoth(jobId) {
            const btn = document.getElementById('evalQaBothBtn');
            const evaluationBtn = document.getElementById('evaluationBtn');
            const qaBtn = document.getElementById('qaBtn');
            const evalResultsDiv = document.getElementById('evaluationResults');
            const qaAfterClick = document.getElementById('qa_after_click');
            const qaResultsTable = document.getElementById('qaResultsTableSearch');
            const qaResultsModel = document.getElementById('qaResultsModelSearch');
            if (!btn || !evalResultsDiv || !qaResultsTable || !qaResultsModel) return;
            if (qaAfterClick) qaAfterClick.style.display = 'block';
            refreshQAIntegratedPaths();
            evalResultsDiv.style.display = 'block';
            evalResultsDiv.innerHTML = '<div style="padding: 15px;">⏳ Running evaluation...</div>';
            qaResultsTable.innerHTML = '<div style="padding: 12px;">⏳ Running QA...</div>';
            qaResultsModel.innerHTML = '<div style="padding: 12px;">⏳ Running QA...</div>';
            [btn, evaluationBtn, qaBtn].forEach(b => { if (b) { b.disabled = true; } });
            try {
                const evalBody = { job_id: jobId };
                if (window.__selectedIntegrationKey) evalBody.integration_run_key = window.__selectedIntegrationKey;
                const evalPromise = sendEvaluationRequest(evalBody, evalResultsDiv, null);
                const qaTablePromise = sendQARequest({ job_id: jobId, use_table_search: true }, qaResultsTable, null);
                const qaModelPromise = sendQARequest({ job_id: jobId, use_table_search: false }, qaResultsModel, null);
                const [, qaTableRes, qaModelRes] = await Promise.all([evalPromise, qaTablePromise, qaModelPromise]);
                if (jobId && (qaTableRes || qaModelRes)) {
                    fetch('{{BACKEND_URL}}/api/save-qa', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ job_id: jobId, table_search: qaTableRes || null, model_search: qaModelRes || null }) }).catch(() => {});
                }
            } catch (err) {
                if (evalResultsDiv.innerHTML.indexOf('❌') < 0) evalResultsDiv.innerHTML = `<div style="padding: 15px; color: #dc3545;">❌ ${err.message}</div>`;
                if (qaResultsTable.innerHTML.indexOf('❌') < 0) qaResultsTable.innerHTML = `<div style="padding: 12px; color: #dc3545;">❌ ${err.message}</div>`;
                if (qaResultsModel.innerHTML.indexOf('❌') < 0) qaResultsModel.innerHTML = `<div style="padding: 12px; color: #dc3545;">❌ ${err.message}</div>`;
            } finally {
                [btn, evaluationBtn, qaBtn].forEach(b => { if (b) { b.disabled = false; } });
                if (btn) btn.textContent = '📊💬 Both (parallel)';
            }
        }

        async function runEvaluation(jobId) {
            const evaluationBtn = document.getElementById('evaluationBtn');
            const resultsDiv = document.getElementById('evaluationResults');
            if (!evaluationBtn || !resultsDiv) return;
            evaluationBtn.disabled = true;
            evaluationBtn.textContent = '⏳ Evaluating...';
            resultsDiv.style.display = 'block';
            resultsDiv.innerHTML = '<div style="padding: 15px; background: #fff; border-radius: 4px;">⏳ Running evaluation...</div>';
            try {
                const body = { job_id: jobId };
                if (window.__selectedIntegrationKey) body.integration_run_key = window.__selectedIntegrationKey;
                await sendEvaluationRequest(body, resultsDiv, evaluationBtn);
            } catch (error) {
                resultsDiv.innerHTML = `<div style="padding: 15px; background: #fff; border-radius: 4px; border: 1px solid #dc3545; color: #dc3545;"><strong>❌ Error:</strong> ${error.message}</div>`;
                evaluationBtn.disabled = false;
                evaluationBtn.textContent = '📊 Generate Evaluation';
            }
        }
        
        async function sendEvaluationRequest(requestBody, resultsDiv, evaluationBtn) {
            try {
                const response = await fetch('{{BACKEND_URL}}/api/evaluate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(requestBody)
                });
                
                // Check if response is ok (status 200-299)
                if (!response.ok) {
                    // Try to parse error response
                    let errorMessage = 'Unknown error';
                    try {
                        const errorData = await response.json();
                        errorMessage = errorData.error || errorData.message || `HTTP ${response.status}: ${response.statusText}`;
                    } catch (e) {
                        errorMessage = `HTTP ${response.status}: ${response.statusText}`;
                    }
                    resultsDiv.innerHTML = `
                        <div style="padding: 15px; background: #fff; border-radius: 4px; border: 1px solid #dc3545; color: #dc3545;">
                            <strong>❌ Evaluation Failed:</strong> ${errorMessage}
                        </div>
                    `;
                    if (evaluationBtn) { evaluationBtn.disabled = false; evaluationBtn.textContent = '📊 Generate Evaluation'; }
                    return;
                }
                
                const data = await response.json();
                
                if (data.status === 'success') {
                    const eval_result = data.evaluation;
                    const table1Data = data.table1 || null;
                    const table2Data = data.table2 || null;
                    displayEvaluationResults(eval_result, resultsDiv, table1Data, table2Data);
                    const jid = requestBody.job_id || currentJobId;
                    if (jid) {
                        fetch('{{BACKEND_URL}}/api/save-evaluation', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ job_id: jid, evaluation: data.evaluation, table1: data.table1, table2: data.table2 }) }).catch(() => {});
                    }
                } else {
                    // Handle error status in response
                    const errorMessage = data.error || data.message || 'Unknown error';
                    resultsDiv.innerHTML = `
                        <div style="padding: 15px; background: #fff; border-radius: 4px; border: 1px solid #dc3545; color: #dc3545;">
                            <strong>❌ Evaluation Failed:</strong> ${errorMessage}
                        </div>
                    `;
                }
            } catch (error) {
                resultsDiv.innerHTML = `
                    <div style="padding: 15px; background: #fff; border-radius: 4px; border: 1px solid #dc3545; color: #dc3545;">
                        <strong>❌ Error:</strong> ${error.message || 'Failed to connect to server'}
                    </div>
                `;
            } finally {
                if (evaluationBtn) { evaluationBtn.disabled = false; evaluationBtn.textContent = '📊 Generate Evaluation'; }
            }
        }
        
        function showError(message) {
            const errorDiv = document.getElementById('errorMsg');
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
        }
        
        function toggleExpand(elementId, toggleElement) {
            const element = document.getElementById(elementId);
            if (element) {
                const isCurrentlyExpanded = element.classList.contains('expanded') || element.style.display === 'block';
                
                if (isCurrentlyExpanded) {
                    // Collapse: hide content, show ▶ (right)
                    element.classList.remove('expanded');
                    element.style.display = 'none';
                    toggleElement.textContent = '▶';
                    toggleElement.classList.remove('expanded');
                } else {
                    // Expand: show content, show ▼ (down)
                    element.classList.add('expanded');
                    element.style.display = 'block';
                    toggleElement.textContent = '▼';
                    toggleElement.classList.add('expanded');
                }
                
                // Handle "Show more" / "Hide more" text if present
                const currentText = toggleElement.textContent;
                if (currentText.includes('Show')) {
                    const count = currentText.match(/Show (\\d+)/)[1];
                    toggleElement.textContent = `Hide ${count} more`;
                } else if (currentText.includes('Hide')) {
                    const count = currentText.match(/Hide (\\d+)/)[1];
                    toggleElement.textContent = `Show ${count} more`;
                }
            }
        }
        
        function toggleSearchType(sectionId, headerElement) {
            const element = document.getElementById(sectionId);
            if (element) {
                const isCurrentlyExpanded = element.classList.contains('expanded');
                
                if (isCurrentlyExpanded) {
                    // Collapse: hide content, triangle points right (▶)
                    element.classList.remove('expanded');
                    headerElement.classList.remove('expanded');
                } else {
                    // Expand: show content, triangle points down (▼)
                    element.classList.add('expanded');
                    headerElement.classList.add('expanded');
                }
            }
        }
        
        async function runQABoth(jobId) {
            const qaBtn = document.getElementById('qaBtn');
            const afterClickDiv = document.getElementById('qa_after_click');
            const resultsDivTable = document.getElementById('qaResultsTableSearch');
            const resultsDivModel = document.getElementById('qaResultsModelSearch');
            if (!qaBtn || !resultsDivTable || !resultsDivModel) return;
            if (afterClickDiv) afterClickDiv.style.display = 'block';
            refreshQAIntegratedPaths();
            qaBtn.disabled = true;
            qaBtn.textContent = '⏳ Generating...';
            resultsDivTable.innerHTML = '<div style="padding: 12px;">⏳ Running QA...</div>';
            resultsDivModel.innerHTML = '<div style="padding: 12px;">⏳ Running QA...</div>';
            try {
                const bodyTable = { job_id: jobId, use_table_search: true };
                const bodyModel = { job_id: jobId, use_table_search: false };
                const [tableRes, modelRes] = await Promise.all([
                    sendQARequest(bodyTable, resultsDivTable, null),
                    sendQARequest(bodyModel, resultsDivModel, null)
                ]);
                if (jobId && (tableRes || modelRes)) {
                    fetch('{{BACKEND_URL}}/api/save-qa', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ job_id: jobId, table_search: tableRes || null, model_search: modelRes || null }) }).catch(() => {});
                }
            } catch (error) {
                resultsDivTable.innerHTML = resultsDivTable.innerHTML.indexOf('❌') >= 0 ? resultsDivTable.innerHTML : '<div style="padding: 12px; color: #dc3545;">❌ ' + error.message + '</div>';
                resultsDivModel.innerHTML = resultsDivModel.innerHTML.indexOf('❌') >= 0 ? resultsDivModel.innerHTML : '<div style="padding: 12px; color: #dc3545;">❌ ' + error.message + '</div>';
            }
            qaBtn.disabled = false;
            qaBtn.textContent = '💬 Generate Answer';
        }
        
        async function sendQARequest(requestBody, resultsDiv, qaBtn) {
            try {
                const response = await fetch('{{BACKEND_URL}}/api/qa', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(requestBody)
                });
                
                if (!response.ok) {
                    let errorMessage = 'Unknown error';
                    try {
                        const errorData = await response.json();
                        errorMessage = errorData.error || errorData.message || `HTTP ${response.status}: ${response.statusText}`;
                    } catch (e) {
                        errorMessage = `HTTP ${response.status}: ${response.statusText}`;
                    }
                    resultsDiv.innerHTML = `
                        <div style="padding: 15px; background: #fff; border-radius: 4px; border: 1px solid #dc3545; color: #dc3545;">
                            <strong>❌ QA Failed:</strong> ${errorMessage}
                        </div>
                    `;
                    if (qaBtn) { qaBtn.disabled = false; qaBtn.textContent = '💬 Generate Answer'; }
                    return null;
                }
                
                const data = await response.json();
                
                if (data.status === 'success') {
                    displayQAResults(data.qa, data.query, resultsDiv);
                    if (qaBtn) { qaBtn.disabled = false; qaBtn.textContent = '💬 Generate Answer'; }
                    return { qa: data.qa, query: data.query };
                } else {
                    resultsDiv.innerHTML = `
                        <div style="padding: 15px; background: #fff; border-radius: 4px; border: 1px solid #dc3545; color: #dc3545;">
                            <strong>❌ QA Failed:</strong> ${data.message || 'Unknown error'}
                        </div>
                    `;
                    if (qaBtn) { qaBtn.disabled = false; qaBtn.textContent = '💬 Generate Answer'; }
                    return null;
                }
            } catch (error) {
                resultsDiv.innerHTML = `
                    <div style="padding: 15px; background: #fff; border-radius: 4px; border: 1px solid #dc3545; color: #dc3545;">
                        <strong>❌ Error:</strong> ${error.message}
                    </div>
                `;
                if (qaBtn) { qaBtn.disabled = false; qaBtn.textContent = '💬 Generate Answer'; }
                return null;
            }
        }
        
        function displayQAResults(qaResult, query, resultsDiv) {
            if (!resultsDiv) return;
            
            resultsDiv.style.display = 'block';
            
            // Handle both: flat {answer:"text",model_ranking:[]} and nested {answer:{answer:"text",...}}
            const answer = (typeof qaResult.answer === 'object' && qaResult.answer !== null)
                ? (qaResult.answer || {}) : (qaResult || {});
            const answerText = answer.answer || qaResult.answer || 'No answer provided';
            const modelRanking = answer.model_ranking || [];
            const summary = answer.summary || {};
            const confidence = answer.confidence || 'unknown';
            const limitations = answer.limitations || [];
            
            let html = `
                <div style="padding: 15px; background: #fff; border-radius: 4px; border: 1px solid #dee2e6;">
                    <h4 style="margin-top: 0; color: #17a2b8; margin-bottom: 15px;">🏆 Model Ranking & Recommendations</h4>
                    
                    <div style="margin-bottom: 15px; padding: 10px; background: #f8f9fa; border-radius: 4px; border-left: 4px solid #17a2b8;">
                        <strong>Query Requirements:</strong> ${query}
                    </div>
                    
                    <div style="margin-bottom: 20px; padding: 15px; background: #e7f3ff; border-radius: 4px;">
                        <div style="font-size: 14px; color: #666; margin-bottom: 8px;">Summary:</div>
                        <div style="font-size: 15px; line-height: 1.6; color: #333;">${answerText}</div>
                    </div>
                    
                    ${modelRanking.length > 0 ? `
                        <div style="margin-bottom: 20px;">
                            <h5 style="color: #17a2b8; margin-bottom: 10px;">Model Rankings:</h5>
                            <div style="display: flex; flex-direction: column; gap: 15px;">
                                ${modelRanking.map(model => `
                                    <div style="padding: 15px; background: ${model.rank <= 3 ? '#e7f3ff' : '#f8f9fa'}; border-radius: 4px; border-left: 4px solid ${model.rank === 1 ? '#28a745' : model.rank === 2 ? '#17a2b8' : model.rank === 3 ? '#ffc107' : '#6c757d'};">
                                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                            <div>
                                                <span style="font-size: 18px; font-weight: bold; color: ${model.rank === 1 ? '#28a745' : model.rank === 2 ? '#17a2b8' : model.rank === 3 ? '#ffc107' : '#6c757d'};">
                                                    #${model.rank}
                                                </span>
                                                <div style="margin-left: 10px;">
                                                    <div style="font-size: 16px; font-weight: bold;">${model.model_name || model.model_id || 'Unknown Model'}</div>
                                                    ${model.model_id && model.model_id !== model.model_name ? `
                                                        <div style="font-size: 12px; color: #666; font-family: monospace; margin-top: 2px;">${model.model_id}</div>
                                                    ` : ''}
                                                </div>
                                            </div>
                                            <div style="padding: 5px 12px; background: ${model.suitability_score >= 80 ? '#d4edda' : model.suitability_score >= 60 ? '#fff3cd' : '#f8d7da'}; border-radius: 12px; font-weight: bold; color: ${model.suitability_score >= 80 ? '#155724' : model.suitability_score >= 60 ? '#856404' : '#721c24'};">
                                                Score: ${model.suitability_score || 'N/A'}/100
                                            </div>
                                        </div>
                                        
                                        ${model.analysis ? `
                                            <div style="margin-bottom: 10px; padding: 10px; background: #f8f9fa; border-radius: 3px;">
                                                <strong style="color: #17a2b8; font-size: 13px;">Analysis:</strong>
                                                <div style="margin-top: 5px; font-size: 13px; line-height: 1.5;">${model.analysis}</div>
                                            </div>
                                        ` : ''}
                                        
                                        ${model.supporting_evidence && model.supporting_evidence.length > 0 ? `
                                            <div style="margin-bottom: 10px;">
                                                <strong style="color: #17a2b8; font-size: 13px;">Supporting Evidence:</strong>
                                                <div style="margin-top: 5px;">
                                                    ${model.supporting_evidence.map(evidence => `
                                                        <div style="margin-bottom: 8px; padding: 8px; background: white; border-left: 3px solid ${evidence.source === 'table_cell' ? '#28a745' : '#17a2b8'}; border-radius: 3px; font-size: 12px;">
                                                            <div style="font-weight: bold; margin-bottom: 3px;">
                                                                <span style="color: ${evidence.source === 'table_cell' ? '#28a745' : '#17a2b8'};">
                                                                    ${evidence.source === 'table_cell' ? '📊 Table Cell' : '📄 Model Card'}
                                                                </span>
                                                            </div>
                                                            <div style="color: #333; margin-bottom: 3px;"><strong>Claim:</strong> ${evidence.claim || 'N/A'}</div>
                                                            <div style="color: #666; font-family: monospace; font-size: 11px; margin-bottom: 3px;"><strong>Evidence:</strong> ${evidence.evidence || 'N/A'}</div>
                                                            ${evidence.relevance ? `<div style="color: #666; font-size: 11px;"><strong>Relevance:</strong> ${evidence.relevance}</div>` : ''}
                                                        </div>
                                                    `).join('')}
                                                </div>
                                            </div>
                                        ` : ''}
                                        
                                        ${model.reasons && model.reasons.length > 0 ? `
                                            <div style="margin-bottom: 10px;">
                                                <strong style="color: #17a2b8; font-size: 13px;">Why this model:</strong>
                                                <ul style="margin: 5px 0 0 0; padding-left: 20px; font-size: 13px;">
                                                    ${model.reasons.map(reason => `<li>${reason}</li>`).join('')}
                                                </ul>
                                            </div>
                                        ` : ''}
                                        
                                        ${model.strengths && model.strengths.length > 0 ? `
                                            <div style="margin-bottom: 10px;">
                                                <strong style="color: #28a745; font-size: 13px;">Strengths:</strong>
                                                <ul style="margin: 5px 0 0 0; padding-left: 20px; font-size: 13px;">
                                                    ${model.strengths.map(strength => `<li>${strength}</li>`).join('')}
                                                </ul>
                                            </div>
                                        ` : ''}
                                        
                                        ${model.limitations && model.limitations.length > 0 ? `
                                            <div style="margin-bottom: 10px;">
                                                <strong style="color: #dc3545; font-size: 13px;">Limitations:</strong>
                                                <ul style="margin: 5px 0 0 0; padding-left: 20px; font-size: 13px;">
                                                    ${model.limitations.map(lim => `<li>${lim}</li>`).join('')}
                                                </ul>
                                            </div>
                                        ` : ''}
                                        
                                        ${model.key_metrics ? `
                                            <div style="margin-top: 10px; padding: 8px; background: white; border-radius: 3px; font-size: 12px;">
                                                <strong>Key Metrics:</strong>
                                                <div style="margin-top: 5px; font-family: monospace; color: #666;">
                                                    ${Object.entries(model.key_metrics).map(([key, value]) => `${key}: ${value}`).join(', ')}
                                                </div>
                                            </div>
                                        ` : model.key_metrics_from_table ? `
                                            <div style="margin-top: 10px; padding: 8px; background: white; border-radius: 3px; font-size: 12px;">
                                                <strong>Key Metrics from Table:</strong>
                                                <div style="margin-top: 5px; font-family: monospace; color: #666;">
                                                    ${Object.entries(model.key_metrics_from_table).map(([key, value]) => `${key}: ${value}`).join(', ')}
                                                </div>
                                            </div>
                                        ` : ''}
                                        
                                        ${model.use_case ? `
                                            <div style="margin-top: 10px; padding: 8px; background: white; border-radius: 3px; font-size: 12px; color: #666;">
                                                <strong>Best Use Case:</strong> ${model.use_case}
                                            </div>
                                        ` : ''}
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    ` : ''}
                    
                    ${summary.total_models_analyzed ? `
                        <div style="margin-bottom: 20px; padding: 12px; background: #f8f9fa; border-radius: 4px;">
                            <strong style="color: #17a2b8;">Analysis Summary:</strong>
                            <div style="margin-top: 8px; font-size: 14px;">
                                <div>Total Models Analyzed: <strong>${summary.total_models_analyzed}</strong></div>
                                ${summary.top_recommendations && summary.top_recommendations.length > 0 ? `
                                    <div style="margin-top: 5px;">Top Recommendations: <strong>${summary.top_recommendations.join(', ')}</strong></div>
                                ` : ''}
                                ${summary.key_criteria_used && summary.key_criteria_used.length > 0 ? `
                                    <div style="margin-top: 5px;">Key Criteria Used: ${summary.key_criteria_used.join(', ')}</div>
                                ` : ''}
                                ${summary.evidence_sources ? `
                                    <div style="margin-top: 5px;">
                                        <strong>Evidence Sources:</strong>
                                        <div style="margin-top: 3px; font-size: 12px;">
                                            ${summary.evidence_sources.table_cells_used ? '✅ Table Cells' : '❌ Table Cells'} | 
                                            ${summary.evidence_sources.model_cards_used ? '✅ Model Cards' : '❌ Model Cards'}
                                            ${summary.evidence_sources.data_quality ? ` | Quality: ${summary.evidence_sources.data_quality}` : ''}
                                        </div>
                                    </div>
                                ` : ''}
                                ${summary.table_analysis ? `
                                    <div style="margin-top: 5px; font-style: italic; color: #666;">Table Analysis: ${summary.table_analysis}</div>
                                ` : ''}
                            </div>
                        </div>
                    ` : ''}
                    
                    <div style="margin-bottom: 15px; padding: 10px; background: ${confidence === 'high' ? '#d4edda' : confidence === 'medium' ? '#fff3cd' : '#f8d7da'}; border-radius: 4px;">
                        <strong>Confidence:</strong> <span style="text-transform: capitalize; font-weight: bold;">${confidence}</span>
                    </div>
                    
                    ${limitations.length > 0 ? `
                        <div style="margin-top: 15px; padding: 12px; background: #fff3cd; border-radius: 4px; border-left: 4px solid #ffc107;">
                            <strong>Limitations:</strong>
                            <ul style="margin: 8px 0 0 0; padding-left: 20px; font-size: 13px;">
                                ${limitations.map(lim => `<li>${lim}</li>`).join('')}
                            </ul>
                        </div>
                    ` : ''}
                    
                    ${qaResult.source ? `
                        <div style="margin-top: 10px; font-size: 11px; color: #999; font-style: italic;">
                            Source: ${qaResult.source}
                        </div>
                    ` : ''}
                </div>
            `;
            resultsDiv.innerHTML = html;
        }
