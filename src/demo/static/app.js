
        let currentJobId = null;
        let eventSource = null;
        
        // Preset queries (loaded from backend); index in select value
        let presetQueriesList = [];
        
        async function loadPresetQueries() {
            const sel = document.getElementById('preset_query_select');
            if (!sel) return;
            try {
                const response = await fetch('{{BACKEND_URL}}/api/preset-queries');
                const data = await response.json();
                if (data.status === 'success' && data.queries && data.queries.length > 0) {
                    presetQueriesList = data.queries;
                    sel.innerHTML = '<option value="">— custom —</option>';
                    data.queries.forEach(function(q, i) {
                        const opt = document.createElement('option');
                        opt.value = String(i);
                        opt.textContent = (q.title || q.id || ('Query ' + (i + 1)));
                        sel.appendChild(opt);
                    });
                }
            } catch (e) {
                console.warn('Preset queries load failed:', e);
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
            
            // Initialize diagram based on default selected mode (query is default)
            toggleMode();
            
            loadPresetQueries();
        });
        
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
        function updateTableSearchKSlider(value) {
            const slider = document.getElementById('table_search_k_slider');
            const num = document.getElementById('table_search_k');
            const v = parseInt(value, 10);
            if (slider && num && v >= parseInt(slider.min) && v <= parseInt(slider.max)) {
                slider.value = v;
                num.value = v;
            }
        }
        function updateTableSearchKDefault() {
            const topK = parseInt(document.getElementById('top_k').value, 10) || 20;
            const suggested = Math.min(Math.max(Math.round(topK * 1.5), 20), 20);
            const current = parseInt(document.getElementById('table_search_k').value, 10) || 20;
            const oldTopK = Math.floor(current / 1.5);
            if (Math.abs(current - Math.round(oldTopK * 1.5)) <= 5) {
                const newVal = Math.min(Math.max(suggested, 1), 20);
                document.getElementById('table_search_k').value = newVal;
                document.getElementById('table_search_k_slider').value = newVal;
            }
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
            const listContainer = document.getElementById('saved_searches_list');
            listContainer.innerHTML = '<div style="text-align: center; color: #666; padding: 20px;">Loading...</div>';
            
            try {
                const response = await fetch('{{BACKEND_URL}}/api/saved-searches');
                const data = await response.json();
                
                if (data.status === 'success') {
                    let html = '';
                    const searches = data.searches || [];
                    const selectEl = document.getElementById('saved_search_select');
                    if (selectEl) {
                        selectEl.innerHTML = '<option value="">— select folder —</option>';
                        searches.forEach(search => {
                            const opt = document.createElement('option');
                            opt.value = search.folder_name || search.id || '';
                            const settingsPart = [search.use_by_type ? 'by_type✓' : 'by_type✗', 'K=' + (search.top_k || '-'), search.card2card_retrieval_mode || ''].filter(Boolean).join(' ');
                            const label = (search.timestamp_str || '') + ' ' + (search.query ? search.query.substring(0, 40) + (search.query.length > 40 ? '...' : '') : search.model_id || search.folder_name || opt.value) + (settingsPart ? ' | ' + settingsPart : '');
                            opt.textContent = label;
                            selectEl.appendChild(opt);
                        });
                    }
                    if (searches.length === 0) {
                        html = '<div style="text-align: center; color: #666; padding: 20px;">No saved searches found. Run a new search to create one.</div>';
                    } else {
                        const escapeHtml = (s) => String(s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
                        searches.forEach(search => {
                            const byTypeLabel = search.use_by_type ? 'by_type ✓' : 'by_type ✗';
                            const settingsLabel = [byTypeLabel, 'K=' + (search.top_k || '-'), (search.card2card_retrieval_mode || '')].filter(Boolean).join(', ');
                            const qShort = search.query ? (search.query.length > 45 ? search.query.substring(0, 45) + '...' : search.query) : '';
                            const oneLine = search.query
                                ? `${search.timestamp_str || ''} - ${qShort} | K: ${search.top_k || '-'} | ${settingsLabel}`
                                : `${search.timestamp_str || ''} - ${search.model_id || ''} | K: ${search.top_k || '-'} | ${settingsLabel}`;
                            const folder = (search.folder_name || search.id || '').replace(/'/g, "\\'").replace(/\\/g, '\\\\');
                            const titleText = [search.query || search.model_id || '', 'by_type: ' + (search.use_by_type ? 'ON' : 'OFF'), 'top_k: ' + (search.top_k || '-'), 'card2card: ' + (search.card2card_retrieval_mode || '-')].join(' | ');
                            const titleSafe = escapeHtml(titleText.substring(0, 200));
                            const oneLineSafe = escapeHtml(oneLine);
                            html += `
                                <div class="saved-search-item" onclick="loadSavedSearchFolder('${folder}')" 
                                     style="padding: 6px 8px; margin-bottom: 4px; background: white; border: 1px solid #ddd; border-radius: 4px; cursor: pointer; font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"
                                     title="${titleSafe}"
                                     onmouseover="this.style.background='#e7f3ff'" 
                                     onmouseout="this.style.background='white'">${oneLineSafe}</div>
                            `;
                        });
                    }
                    
                    listContainer.innerHTML = html;
                } else {
                    listContainer.innerHTML = '<div style="text-align: center; color: #dc3545; padding: 20px;">Error loading saved searches</div>';
                }
            } catch (error) {
                listContainer.innerHTML = `<div style="text-align: center; color: #dc3545; padding: 20px;">Error: ${error.message}</div>`;
            }
        }
        
        async function loadSavedSearchFolder(folderName) {
            // Reset UI
            document.getElementById('progressSection').classList.add('active');
            document.getElementById('resultsSection').classList.remove('active');
            document.getElementById('errorMsg').style.display = 'none';
            document.getElementById('logContainer').innerHTML = '';
            
            try {
                const response = await fetch('{{BACKEND_URL}}/api/search', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        search_mode: 'mimic',
                        folder_name: folderName
                    })
                });
                
                const data = await response.json();
                
                if (data.status === 'completed' || data.status === 'success') {
                    currentJobId = data.job_id;
                    displayResults({ ...(data.results || data), job_id: data.job_id });
                    await restoreIntegrationEvaluationQA(data);
                    document.getElementById('progressSection').classList.remove('active');
                } else {
                    showError(data.message || 'Failed to load saved search');
                }
            } catch (error) {
                showError('Error: ' + error.message);
            }
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
                setSelect('integration_model_search_mode', modelRes.card2card_retrieval_mode);
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
                syncBothIntegrationDisplays();
            } else {
            const hasModel = data.integration_model_search && data.integration_model_search.status === 'success' && data.integration_model_search.integrated_table;
            const hasTable = data.integration_table_search && data.integration_table_search.status === 'success' && data.integration_table_search.integrated_table;
            if (hasModel || hasTable) {
                container.style.display = 'block';
                if (hasModel) {
                    const m = data.integration_model_search;
                    const key = getModelSearchKey(m.integration_type, m.card2card_retrieval_mode);
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
        
        async function loadDemoExample() {
            // Load the demo example from template folder
            const loadBtn = event.target;
            loadBtn.disabled = true;
            loadBtn.textContent = '⏳ Loading Template...';
            
            // Reset UI
            document.getElementById('progressSection').classList.add('active');
            document.getElementById('resultsSection').classList.remove('active');
            document.getElementById('errorMsg').style.display = 'none';
            document.getElementById('logContainer').innerHTML = '';
            
            try {
                const response = await fetch('{{BACKEND_URL}}/api/search', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        search_mode: 'mimic',
                        folder_name: 'template'
                    })
                });
                
                const data = await response.json();
                
                if (data.status === 'completed' || data.status === 'success') {
                    currentJobId = data.job_id;
                    displayResults({ ...(data.results || data), job_id: data.job_id });
                    restoreIntegrationEvaluationQA(data);
                    document.getElementById('progressSection').classList.remove('active');
                } else {
                    showError(data.message || 'Failed to load template');
                }
            } catch (error) {
                showError('Error: ' + error.message);
            } finally {
                loadBtn.disabled = false;
                loadBtn.textContent = '🎨 Load Demo Example (Template)';
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
            document.getElementById('errorMsg').style.display = 'none';
            document.getElementById('logContainer').innerHTML = '';
            
            try {
                const response = await fetch('{{BACKEND_URL}}/api/search', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        search_mode: 'mimic',
                        folder_name: folderName
                    })
                });
                
                const data = await response.json();
                
                if (data.status === 'completed' || data.status === 'success') {
                    currentJobId = data.job_id;
                    displayResults({ ...(data.results || data), job_id: data.job_id });
                    await restoreIntegrationEvaluationQA(data);
                    document.getElementById('progressSection').classList.remove('active');
                } else {
                    showError(data.message || 'Failed to load saved search');
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
            
            const mode = (document.getElementById('search_mode_select') || {}).value || 'query';
            const query = document.getElementById('query').value.trim();
            const modelId = document.getElementById('model_id').value.trim();
            const topK = parseInt((document.getElementById('top_k') || {}).value, 10) || 100;  // Left aligns to right; high default
            const tableSearchK = parseInt(document.getElementById('table_search_k').value, 10) || 10;
            // Table retrieval: always run search (no load-from-JSON option)
            const tab2tabMode = 'search';
            // One-click: always run all; primary display uses dense
            const card2cardRetrievalMode = 'dense';
            
            // Validate input based on mode
            if (mode === 'query' && !query) {
                showError('Please enter a query');
                return;
            }
            if (mode === 'modelid' && !modelId) {
                showError('Please enter a model ID');
                return;
            }
            
            // Reset UI
            document.getElementById('searchBtn').disabled = true;
            document.getElementById('progressSection').classList.add('active');
            document.getElementById('resultsSection').classList.remove('active');
            document.getElementById('errorMsg').style.display = 'none';
            document.getElementById('logContainer').innerHTML = '';
            
            try {
                // Start search
                const requestBody = {
                    search_mode: 'new',
                    mode: mode,
                    top_k: topK,
                    tab2tab_mode: 'search',
                    table_search_k: tableSearchK,
                    card2card_retrieval_mode: card2cardRetrievalMode,
                    use_by_type: document.getElementById('use_by_type').checked
                };
                
                if (mode === 'query') {
                    requestBody.query = query;
                    const requireSeedEl = document.getElementById('require_seed_has_tables');
                    requestBody.require_seed_has_tables = !!(requireSeedEl && requireSeedEl.value === '1');
                } else {
                    requestBody.model_id = modelId;
                }
                
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
        
        function toggleMode() {
            const modeEl = document.getElementById('search_mode_select');
            const mode = modeEl ? modeEl.value : 'query';
            const queryInput = document.getElementById('query-input');
            const modelIdInput = document.getElementById('modelid-input');
            const diagramImg = document.getElementById('search-diagram');
            
            if (mode === 'query') {
                queryInput.classList.add('active');
                modelIdInput.classList.remove('active');
                if (diagramImg) diagramImg.src = '/static/fig/modelsearch_wquery.png';
            } else {
                queryInput.classList.remove('active');
                modelIdInput.classList.add('active');
                if (diagramImg) diagramImg.src = '/static/fig/modelsearch.png';
            }
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
                        displayResults(data.results);
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
            const container = document.getElementById('resultsContent');
            // Pipeline error (e.g. query2modelcard failed or no model from query) - shown at top of results
            const errorBlock = results.error
                ? `<div style="padding: 12px; margin-bottom: 15px; color: #721c24; background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 6px;"><strong>❌ Pipeline error:</strong> ${results.error}</div>`
                : '';
            // Seed model + Query tables: same row, two columns (aligned with the two result cards below)
            const seedModelId = results.model_id || null;
            const headerStyle = 'font-size: 12px; color: #666;';
            const seedModelCell = results.error ? '' : (seedModelId
                ? `<span style="${headerStyle}"><strong>Seed model (from query):</strong> <a href="https://huggingface.co/${seedModelId}" target="_blank">${seedModelId}</a></span>`
                : `<span style="font-size: 12px; color: #856404;">⚠️ Model ID missing</span>`);
            // Query table path(s) from seed model card — used to run table search; result items below are model cards hit by that search
            let queryTables = [];
            let searchedTables = [];
            const c2t2c = results.card2tab2card_results || {};
            Object.keys(c2t2c).forEach(type => {
                const data = c2t2c[type];
                if (data && Array.isArray(data.query_tables) && data.query_tables.length > 0) {
                    data.query_tables.forEach(p => { if (p && !queryTables.includes(p)) queryTables.push(p); });
                }
                if (data && Array.isArray(data.searched_tables) && data.searched_tables.length > 0) {
                    data.searched_tables.forEach(p => { if (p && !searchedTables.includes(p)) searchedTables.push(p); });
                }
            });
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
            const tablesNoteCell = results.error ? '<span style="' + headerStyle + '">—</span>' : (queryTables.length > 0 || searchedTables.length > 0
                ? `<span style="${headerStyle}"><strong>Query table(s):</strong> <span style="font-size: 10px; font-family: monospace;">${queryTables.length ? queryTables.map(basename).join(' ') : '—'}</span><br><strong>Searched table(s):</strong> <span style="font-size: 10px; font-family: monospace;">${searchedTables.length ? searchedTables.map(basename).join(' ') : '—'}</span></span>`
                : `<span style="${headerStyle}"><strong>Query table(s):</strong> —<br><strong>Searched table(s):</strong> —</span>`);
            const headerRowHtml = `<div class="results-grid" style="margin-bottom: 6px; align-items: center;">
                <div>${seedModelCell}</div>
                <div>${tablesNoteCell}</div>
            </div>`;
            
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
            const card2cardMoreId = 'card2card-more-' + Date.now();
            const card2tab2cardIds = {};
            const card2tab2cardResults = results.card2tab2card_results || {};
            Object.keys(card2tab2cardResults).forEach((type, idx) => {
                card2tab2cardIds[type] = 'card2tab2card-' + type + '-' + Date.now() + '-' + idx;
            });
            
            // Get Card2Card results for all modes
            const card2cardAllModes = results.card2card_all_modes || {};
            const retrievalModes = [
                { key: 'dense', label: 'Dense', desc: 'Semantic similarity using embeddings' },
                { key: 'sparse', label: 'Sparse', desc: 'Sparse retrieval via Pyserini Lucene BM25' },
                { key: 'hybrid', label: 'Hybrid', desc: 'Pyserini sparse + FAISS dense, then combine' }
            ];
            const currentMode = results.card2card_retrieval_mode || 'dense';
            
            let html = `
                ${errorBlock}
                ${headerRowHtml}
                <div class="results-grid">
                    <div class="result-card" style="min-width: 0;">
                        <h3 style="margin-top: 0; margin-bottom: 8px; font-size: 14px; color: #495057;">
                            <span class="number-badge">1</span> Card2Card Results
                        </h3>
                        ${retrievalModes.map((modeInfo, idx) => {
                            const modeKey = modeInfo.key;
                            const modeResults = card2cardAllModes[modeKey] || [];
                            const isError = modeResults.error !== undefined;
                            const resultList = isError ? [] : (Array.isArray(modeResults) ? modeResults : []);
                            const sectionId = `card2card-${modeKey}-${Date.now()}-${idx}`;
                            const isCurrentMode = modeKey === currentMode;
                            
                            return `
                                <div class="search-type-section" style="margin-bottom: 15px; ${isCurrentMode ? 'border: 2px solid #007bff;' : ''}">
                                    <div class="search-type-header ${isCurrentMode ? 'expanded' : ''}" onclick="toggleSearchType('${sectionId}', this)" style="${isCurrentMode ? 'background: #e7f3ff;' : ''}">
                                        <h4 style="margin: 0; display: flex; align-items: center; gap: 8px;">
                                            ${modeInfo.label}
                                            <span style="font-size: 12px; color: #666; font-weight: normal;">${isError ? 'Error' : resultList.length + ' models'}</span>
                                        </h4>
                                    </div>
                                    <div class="collapsible-content ${isCurrentMode ? 'expanded' : ''}" id="${sectionId}">
                                        ${isError ? `
                                            <div style="padding: 10px; color: #dc3545; background: #f8d7da; border-radius: 4px; margin: 10px 0;">
                                                ❌ Error: ${modeResults.error || 'Unknown error'}
                                            </div>
                                        ` : resultList.length > 0 ? `
                                            <ul class="result-list" style="list-style: none; padding: 0;">
                                                ${resultList.slice(0, 10).map(m => `<li class="result-item">${formatModel(m)}</li>`).join('')}
                                                ${resultList.length > 10 ? `
                                                    <li class="collapsible-content" id="${sectionId}-more">
                                                        ${resultList.slice(10).map(m => `<div class="result-item">${formatModel(m)}</div>`).join('')}
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
                        <h3 style="margin-top: 0; margin-bottom: 6px; font-size: 14px; font-weight: 600; color: #343a40;"><span class="number-badge">2</span> Card2Tab2Card Results</h3>
                        ${(() => {
                            // Filter: show keyword, unionable, joinable, and by_type (table type classification)
                            const allowedTypes = ['keyword', 'unionable', 'single_column', 'multi_column', 'by_type'];
                            // Map joinable types to display names
                            const typeDisplayNames = {
                                'single_column': 'joinable (single_column)',
                                'multi_column': 'joinable (multi_column)',
                                'keyword': 'keyword',
                                'unionable': 'unionable',
                                'by_type': 'by type (classification)'
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
                                <div class="search-type-section">
                                    <div class="search-type-header" onclick="toggleSearchType('${sectionId}', this)">
                                        <h4 style="margin: 0; display: flex; align-items: center; gap: 8px; font-size: 14px;">
                                            ${displayName}
                                            <span style="font-size: 12px; color: #666; font-weight: normal;">
                                                ${models.length} models${realTableCount ? ` from ${realTableCount} tables` : ''}
                                            </span>
                                        </h4>
                                    </div>
                                    <div class="collapsible-content" id="${sectionId}">
                                        <ul class="result-list" style="list-style: none; padding: 0;">
                                            ${sortedModels.length > 0 ? sortedModels.map((m, idx) => {
                                                let modelId = typeof m === 'string' ? m : (m.model_id || m);
                                                modelId = String(modelId).trim();
                                                const modelUrl = typeof m === 'string' ? `https://huggingface.co/${modelId}` : (m.url || `https://huggingface.co/${modelId}`);
                                                const modelTables = modelToTables[modelId] || [];
                                                const hasTables = modelTables.length > 0;
                                                
                                                const tableLine = hasTables ? modelTables.map(t => String(t).split('/').pop()).join(' ') : '';
                                                return `
                                                    <li class="result-item" style="margin-bottom: 4px;">
                                                        <div style="display: flex; align-items: baseline; gap: 4px; flex-wrap: wrap;">
                                                            <a href="${modelUrl}" target="_blank" style="color: #007bff; text-decoration: none; font-weight: 500; font-size: 13px;">${modelId}</a>
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
            `;
            
            // Comparison HTML: will be merged into Evaluation card
            const comparisonHtml = results.comparison ? `
                <h4 style="margin: 0 0 10px 0; font-size: 14px; color: #856404;">Comparison</h4>
                <p style="font-size: 14px; color: #666; margin-bottom: 15px;">
                    Overlap analysis between Card2Card and Card2Tab2Card search results
                </p>
                <div style="overflow-x: auto; margin-bottom: 20px;">
                    <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
                        <thead>
                            <tr style="background: #007bff; color: white;">
                                <th style="padding: 8px; text-align: left; border: 1px solid #0056b3;">Search Type</th>
                                <th style="padding: 8px; text-align: center; border: 1px solid #0056b3;">Card2Card Count</th>
                                <th style="padding: 8px; text-align: center; border: 1px solid #0056b3;">Card2Tab2Card Count</th>
                                <th style="padding: 8px; text-align: center; border: 1px solid #0056b3;">Overlap Count</th>
                                <th style="padding: 8px; text-align: center; border: 1px solid #0056b3;">Overlap Ratio</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${Object.entries(results.comparison).map(([type, comp], idx) => `
                                <tr style="${idx % 2 === 0 ? 'background: #f8f9fa;' : 'background: white;'}">
                                    <td style="padding: 6px; border: 1px solid #dee2e6; font-weight: 500;">${type}</td>
                                    <td style="padding: 6px; border: 1px solid #dee2e6; text-align: center;">${comp.card2card_count || 0}</td>
                                    <td style="padding: 6px; border: 1px solid #dee2e6; text-align: center;">${comp.card2tab2card_count || 0}</td>
                                    <td style="padding: 6px; border: 1px solid #dee2e6; text-align: center; font-weight: bold; color: #28a745;">${comp.overlap_count || 0}</td>
                                    <td style="padding: 6px; border: 1px solid #dee2e6; text-align: center; font-weight: bold;">${((comp.overlap_ratio || 0) * 100).toFixed(1)}%</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            ` : '';
            
            // Table Integration: Model Search and Table Search SEPARATE - each has its own params and dropdown switching
            const integrationCardStyle = 'padding: 12px; background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%); border-radius: 8px; border: 1px solid #dee2e6; font-size: 13px; color: #212529; min-width: 0;';
            const integrationTitleStyle = 'margin-top: 0; margin-bottom: 6px; font-size: 15px; font-weight: 600; color: #1a1d21;';
            const topKLabelStyle = 'display: block; margin-bottom: 2px; font-size: 11px; font-weight: 500; color: #212529;';
            const defaultIntegrationK = results.table_search_k || 10;
            const defaultIntegrationMaxModels = results.top_k || 10;
            html += `
                <div class="integration-section" style="${integrationCardStyle}; margin-top: 16px;">
                    <h3 style="${integrationTitleStyle}">Table Integration</h3>
                    <p style="font-size: 12px; color: #5a6268; margin-bottom: 10px;">Model Search and Table Search are saved separately. Switch via dropdowns to view different saved results. One <strong>Integrated</strong> button runs both.</p>
                    <div style="display: flex; gap: 10px; align-items: flex-end; flex-wrap: wrap; margin-bottom: 6px;">
                        <div style="flex: 0 0 auto;"><label style="${topKLabelStyle}">integration method:</label><select id="integration_type" class="form-control" onchange="syncBothIntegrationDisplays();" style="width: 100px; box-sizing: border-box; padding: 4px 6px; font-size: 12px;">
                            <option value="union">Union</option>
                            <option value="intersection">Intersection</option>
                            <option value="alite">ALITE</option>
                            <option value="outer_join">Outer Join</option>
                        </select></div>
                        <!-- top k tables/models: commented out - use defaults; integrate prints #tables/#models -->
                        <div style="display: none;"><input type="number" id="integration_k" value="${defaultIntegrationK}" min="1" max="50"><input type="number" id="integration_max_models" value="${defaultIntegrationMaxModels}" min="1" max="50"></div>
                        <button id="integrationRunBothBtn" onclick="runBothIntegrations('${results.job_id || currentJobId}')" style="padding: 6px 14px; font-size: 13px; font-weight: 600;">Integrated</button>
                    </div>
                    <div id="integrationResultsContainer" style="margin-top: 12px;">
                        <div style="margin-bottom: 16px; padding: 10px; background: #e7f3ff; border-radius: 6px; border-left: 4px solid #007bff;">
                            <h4 style="margin: 0 0 6px 0; font-size: 13px; color: #004085;">Model Search</h4>
                            <div style="display: flex; gap: 8px; align-items: flex-end; flex-wrap: wrap; margin-bottom: 8px;">
                                <div style="flex: 0 0 auto;"><label style="${topKLabelStyle}">retrieval mode:</label><select id="integration_model_search_mode" class="form-control" onchange="syncModelSearchDisplay()" style="width: 90px; box-sizing: border-box; padding: 4px 6px; font-size: 12px;">
                                    <option value="dense">Dense</option>
                                    <option value="sparse">Sparse</option>
                                    <option value="hybrid">Hybrid</option>
                                </select></div>
                            </div>
                            <div id="integrationModelSearchResults"></div>
                        </div>
                        <div style="padding: 10px; background: #d4edda; border-radius: 6px; border-left: 4px solid #28a745;">
                            <h4 style="margin: 0 0 6px 0; font-size: 13px; color: #155724;">Table Search</h4>
                            <div style="display: flex; gap: 8px; align-items: flex-end; flex-wrap: wrap; margin-bottom: 8px;">
                                <div style="flex: 0 0 auto;"><label style="${topKLabelStyle}">tables source:</label><select id="integration_tables_source" class="form-control" onchange="syncTableSearchDisplay()" style="width: 175px; box-sizing: border-box; padding: 4px 6px; font-size: 12px;" title="Intermediate: from search. All from Modelcards: parquet (DuckDB).">
                                    <option value="intermediate" selected>Intermediate tables</option>
                                    <option value="all_from_modelcards">All tables from Modelcards</option>
                                </select></div>
                                <div style="flex: 0 0 auto;"><label style="${topKLabelStyle}">search type:</label><select id="integration_search_type" class="form-control" onchange="syncTableSearchDisplay()" style="width: 110px; box-sizing: border-box; padding: 4px 6px; font-size: 12px;">
                                    <option value="single_column">Single Column</option>
                                    <option value="keyword">Keyword</option>
                                    <option value="by_type">By Type</option>
                                    <option value="multi_column">Multi Column</option>
                                    <option value="unionable">Unionable</option>
                                    <option value="complex">Complex</option>
                                    <option value="correlation">Correlation</option>
                                    <option value="imputation">Imputation</option>
                                    <option value="augmentation">Augmentation</option>
                                    <option value="dependent_data">Dependent Data</option>
                                    <option value="feature_for_ml">Feature for ML</option>
                                    <option value="multi_column_collinearity">Multi-Column Collinearity</option>
                                    <option value="negative_example">Negative Example</option>
                                </select></div>
                            </div>
                            <div id="integrationResults"></div>
                        </div>
                    </div>
                </div>
            `;
            
            html += `
                <div id="integrationShortAnalysis" class="integration-summary-section" style="margin-top: 16px; padding: 14px; background: #e2e3e5; border-radius: 6px; border: 2px solid #6c757d; display: none;">
                    <h4 style="margin: 0 0 6px 0; font-size: 14px; color: #383d41;">Summary (between Table Integration and Evaluation)</h4>
                    <p style="font-size: 11px; color: #6c757d; margin: 0 0 10px 0;">Deterministic comparison: column overlap, Jaccard, containment, coverage. No LLM.</p>
                    <div id="integrationShortAnalysisContent"></div>
                </div>
                <div class="evaluation-section" style="margin-top: 16px; padding: 12px; background: #fff3cd; border-radius: 6px; border: 2px solid #ffc107;">
                    ${comparisonHtml}
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                        <div>
                            <h3 style="margin: 0 0 4px 0; color: #856404; font-size: 15px;">📊 Evaluation on Integrated Tables</h3>
                            <p style="font-size: 12px; color: #666; margin: 0;">Evaluate diversity between Table Search and Model Search integration results using LLM.</p>
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
                                <div><strong style="font-size: 13px; color: #0c5460;">Table Search Integration</strong></div>
                                <div><strong style="font-size: 13px; color: #0c5460;">Model Search Integration</strong></div>
                            </div>
                            <div id="qaIntegratedPaths" style="font-size: 11px; color: #555; margin-bottom: 12px;"></div>
                            <h4 style="margin: 0 0 10px 0; font-size: 14px; color: #0c5460;">Answers compare</h4>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                                <div>
                                    <strong style="font-size: 13px; color: #0c5460;">Table Search</strong>
                                    <div id="qaResultsTableSearch" style="margin-top: 8px;"></div>
                                </div>
                                <div>
                                    <strong style="font-size: 13px; color: #0c5460;">Model Search</strong>
                                    <div id="qaResultsModelSearch" style="margin-top: 8px;"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            
            container.innerHTML = html;
            window.__modelSearchRuns = [];
            window.__tableSearchRuns = [];
            document.getElementById('resultsSection').classList.add('active');
            syncBothIntegrationDisplays();
        }

        function refreshQAIntegratedPaths() {
            const target = document.getElementById('qaIntegratedPaths');
            if (!target) return;
            // Use run data so paths show even when DOM structure differs; same key logic as sync*Display
            const integrationType = (document.getElementById('integration_type') || {}).value || 'union';
            const searchType = (document.getElementById('integration_search_type') || {}).value || 'single_column';
            const tablesSource = (document.getElementById('integration_tables_source') || {}).value || 'intermediate';
            const card2cardMode = (document.getElementById('integration_model_search_mode') || {}).value || 'dense';
            const tableKey = getTableSearchKey(integrationType, searchType, tablesSource);
            const modelKey = getModelSearchKey(integrationType, card2cardMode);
            const tableRun = (window.__tableSearchRuns || []).find(r => (r.key || getTableSearchKey(r.integration_type, r.search_type, r.tables_source)) === tableKey);
            const modelRun = (window.__modelSearchRuns || []).find(r => (r.key || getModelSearchKey(r.integration_type, r.card2card_retrieval_mode)) === modelKey);
            const tablePath = (tableRun && tableRun.saved_path) ? tableRun.saved_path : '';
            const modelPath = (modelRun && modelRun.saved_path) ? modelRun.saved_path : '';
            const tableHtml = tablePath
                ? `<code style="background: #f1f3f5; padding: 2px 6px; border-radius: 4px;">${tablePath}</code>`
                : '<span style="color:#999;">N/A</span>';
            const modelHtml = modelPath
                ? `<code style="background: #f1f3f5; padding: 2px 6px; border-radius: 4px;">${modelPath}</code>`
                : '<span style="color:#999;">N/A</span>';
            target.innerHTML = `
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                    <div>Table Search CSV: ${tableHtml}</div>
                    <div>Model Search CSV: ${modelHtml}</div>
                </div>
            `;
        }
        
        const INTEGRATION_TABLE_VIEWPORT_STYLE = 'height: 320px; width: 100%; max-width: 100%; overflow-x: auto; overflow-y: auto; border: 1px solid #dee2e6; border-radius: 6px; background: #fff;';
        const DISPLAY_MAX_ROWS = 20;
        window.__integrationTables = window.__integrationTables || {};
        window.__modelSearchRuns = window.__modelSearchRuns || [];
        window.__tableSearchRuns = window.__tableSearchRuns || [];
        
        function getModelSearchKey(integrationType, card2cardMode) {
            return [(integrationType || 'union'), (card2cardMode || 'dense')].map(s => String(s).toLowerCase().replace(/[^a-z0-9_]/g, '_')).join('_');
        }
        function getTableSearchKey(integrationType, searchType, tablesSource) {
            const src = (tablesSource || 'intermediate').toLowerCase().replace(/-/g, '_').replace(/[^a-z0-9_]/g, '_');
            return [(integrationType || 'union'), (searchType || 'single_column'), src].map(s => String(s).toLowerCase().replace(/[^a-z0-9_]/g, '_')).join('_');
        }
        
        function syncBothIntegrationDisplays() {
            try { syncModelSearchDisplay(); } catch (e) { console.error('syncModelSearchDisplay error:', e); }
            try { syncTableSearchDisplay(); } catch (e) { console.error('syncTableSearchDisplay error:', e); }
        }
        
        function syncModelSearchDisplay() {
            const leftDiv = document.getElementById('integrationModelSearchResults');
            const container = document.getElementById('integrationResultsContainer');
            if (!leftDiv || !container) return;
            container.style.display = 'block';
            const integrationType = (document.getElementById('integration_type') || {}).value || 'union';
            const card2cardMode = (document.getElementById('integration_model_search_mode') || {}).value || 'dense';
            const key = getModelSearchKey(integrationType, card2cardMode);
            const runs = window.__modelSearchRuns || [];
            const run = runs.find(r => (r.key || getModelSearchKey(r.integration_type, r.card2card_retrieval_mode)) === key);
            const placeholder = '<div style="padding: 12px; background: #f8f9fa; border: 1px dashed #dee2e6; border-radius: 6px; color: #6c757d; font-size: 13px;">No result for this combination. Click <strong>Integrated</strong> to run.</div>';
            const noResultMsg = placeholder;
                if (run && run.status === 'success' && run.integrated_table) {
                const stats = run.stats || {};
                let extra = '';
                const modelIds = run.models_with_tables || [];
                if (modelIds.length > 0) {
                    const links = modelIds.map(m => `<a href="https://huggingface.co/${m}" target="_blank">${m}</a>`).join(', ');
                    extra = `<div style="margin-bottom: 10px; padding: 8px; background: #e7f3ff; border-radius: 4px; font-size: 12px;"><strong>Model IDs (Model Search):</strong> ${links} <span style="color:#004085;">(${modelIds.length} models)</span></div>`;
                } else {
                    extra = '<div style="margin-bottom: 10px; padding: 8px; background: #f8f9fa; border-radius: 4px; font-size: 12px; color: #6c757d;">Model IDs (Model Search): — (none or not available)</div>';
                }
                leftDiv.innerHTML = renderIntegrationTable(run.integrated_table, stats, { title: 'Model Search integration', successColor: '#007bff', extraHtml: extra, savedPath: run.saved_path || '', downloadId: 'model-search-' + key });
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
            const integrationType = (document.getElementById('integration_type') || {}).value || 'union';
            const searchType = (document.getElementById('integration_search_type') || {}).value || 'single_column';
            const tablesSource = (document.getElementById('integration_tables_source') || {}).value || 'intermediate';
            const card2cardMode = (document.getElementById('integration_model_search_mode') || {}).value || 'dense';
            const tableKey = getTableSearchKey(integrationType, searchType, tablesSource);
            const modelKey = getModelSearchKey(integrationType, card2cardMode);
            const tableRun = (window.__tableSearchRuns || []).find(r => (r.key || getTableSearchKey(r.integration_type, r.search_type, r.tables_source)) === tableKey);
            const modelRun = (window.__modelSearchRuns || []).find(r => (r.key || getModelSearchKey(r.integration_type, r.card2card_retrieval_mode)) === modelKey);
            const tableT = tableRun && tableRun.integrated_table;
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
                    missingHtml += `<li><code>${col}</code>: Table Search ${pctA}%, Model Search ${pctB}%</li>`;
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
                            <li>Only in Table Search: <strong>${colsOnlyInTable}</strong>${onlyTable.length ? ' (' + onlyTable.slice(0, 3).join(', ') + (onlyTable.length > 3 ? '…' : '') + ')' : ''}</li>
                            <li>Only in Model Search: <strong>${colsOnlyInModel}</strong>${onlyModel.length ? ' (' + onlyModel.slice(0, 3).join(', ') + (onlyModel.length > 3 ? '…' : '') + ')' : ''}</li>
                        </ul>
                    </div>
                    <div style="padding: 8px; background: #fff; border-radius: 4px; border: 1px solid #dee2e6;">
                        <strong>Deterministic metrics (no LLM)</strong>
                        <ul style="margin: 6px 0 0 0; padding-left: 18px;">
                            <li>Column Jaccard: <strong>${jaccardCols}</strong> (|A∩B|/|A∪B|)</li>
                            <li>Containment (Table→Model): <strong>${containmentTableInModel}</strong> (|A∩B|/|A|)</li>
                            <li>Containment (Model→Table): <strong>${containmentModelInTable}</strong> (|A∩B|/|B|)</li>
                        </ul>
                    </div>
                </div>
                <div style="margin-top: 8px; padding: 8px; background: #fff; border-radius: 4px; border: 1px solid #dee2e6; font-size: 12px;">
                    <strong>Row counts</strong>: Table Search <strong>${totalRowsA}</strong> rows, ${tableT.columns.length} cols; Model Search <strong>${totalRowsB}</strong> rows, ${modelT.columns.length} cols.
                </div>
                ${missingHtml}
                <p style="margin: 8px 0 0 0; font-size: 10px; color: #6c757d;">Schema + data-consistency metrics only (overlap, Jaccard, containment, coverage). Deterministic; no LLM.</p>
            `;
            container.style.display = 'block';
        }
        
        function syncTableSearchDisplay() {
            const rightDiv = document.getElementById('integrationResults');
            const container = document.getElementById('integrationResultsContainer');
            if (!rightDiv || !container) return;
            container.style.display = 'block';
            const integrationType = (document.getElementById('integration_type') || {}).value || 'union';
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
            if (run && run.status === 'success' && run.integrated_table) {
                let extra = '';
                const modelIds = run.models_with_tables || [];
                if (modelIds.length > 0) {
                    const links = modelIds.map(m => `<a href="https://huggingface.co/${m}" target="_blank">${m}</a>`).join(', ');
                    extra = `<div style="margin-bottom: 10px; padding: 8px; background: #d4edda; border-radius: 4px; font-size: 12px;"><strong>Model IDs (Table Search):</strong> ${links} <span style="color:#155724;">(${modelIds.length} models)</span></div>`;
                } else {
                    extra = '<div style="margin-bottom: 10px; padding: 8px; background: #f8f9fa; border-radius: 4px; font-size: 12px; color: #6c757d;">Model IDs (Table Search): — (none or not available)</div>';
                }
                rightDiv.innerHTML = renderIntegrationTable(run.integrated_table, run.stats || {}, { title: 'Table Search integration', successColor: '#28a745', extraHtml: extra, savedPath: run.saved_path || '', downloadId: 'table-search-' + key });
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
                        ${t.columns.map(col => `<th style="border: 1px solid #dee2e6; padding: 8px; text-align: left; background: #f8f9fa; white-space: nowrap;">${col}</th>`).join('')}
                    </tr></thead>
                    <tbody>
                        ${t.data.map(row => `<tr>${row.map(cell => `<td style="border: 1px solid #dee2e6; padding: 6px; white-space: nowrap;">${cell != null && cell !== '' ? cell : ''}</td>`).join('')}</tr>`).join('')}
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
        
        function buildTableInfo(table) {
            const totalRows = (table.data || []).length;
            const cols = table.columns || [];
            return cols.map((col, i) => {
                let nonNull = 0;
                let hasNum = true;
                for (let r = 0; r < table.data.length; r++) {
                    const v = table.data[r][i];
                    if (v != null && v !== '') {
                        nonNull++;
                        if (hasNum && isNaN(Number(v))) hasNum = false;
                    }
                }
                const pct = totalRows ? (nonNull / totalRows * 100).toFixed(1) : '0';
                const nonNullPct = pct + '%';
                const dtype = nonNull === 0 ? 'object' : (hasNum ? 'number' : 'object');
                return { col, nonNullPct, dtype };
            });
        }
        function renderIntegrationTable(table, stats, options) {
            const { title = 'Integration', successColor = '#28a745', extraHtml = '', savedPath = '', downloadId = '' } = options || {};
            if (!table || (stats && stats.output_rows === 0)) {
                return `<div style="padding: 15px; background: #fff; border-radius: 4px; border: 1px solid #dee2e6;">
                    <h4 style="margin-top: 0; color: ${successColor};">✅ ${title}</h4>
                    <div style="margin-bottom: 15px;">${(stats && stats.output_columns === 0) ? '⚠️ No common columns. Intersection result is empty.' : '⚠️ No common rows. Intersection result is empty.'}</div></div>`;
            }
            if (downloadId) window.__integrationTables[downloadId] = table;
            const totalRows = (table.data || []).length;
            const displayRows = table.data.slice(0, DISPLAY_MAX_ROWS);
            const showRowsLabel = totalRows <= DISPLAY_MAX_ROWS ? totalRows : `${DISPLAY_MAX_ROWS} of ${totalRows}`;
            const footer = [];
            if (savedPath) footer.push(`<span style="font-size: 12px; color: #666;">Saved to: <code style="background: #f1f3f5; padding: 2px 6px; border-radius: 4px;">${savedPath}</code></span>`);
            footer.push(`<button type="button" onclick="downloadIntegrationTableAsCsv('${downloadId}')" style="margin-left: 10px; padding: 6px 12px; font-size: 13px; background: #28a745; color: white; border: none; border-radius: 6px; cursor: pointer;">📥 Download full CSV (${totalRows} rows)</button>`);
            const infoRows = buildTableInfo(table);
            const infoHeaderRow = '<th style="border:1px solid #dee2e6;padding:4px 8px; background: #e9ecef; font-size: 11px;"> </th>' + infoRows.map(({ col }) => `<th style="border:1px solid #dee2e6;padding:4px 8px; background: #e9ecef; font-size: 11px; white-space: nowrap;">${String(col)}</th>`).join('');
            const infoNonNullRow = '<td style="border:1px solid #dee2e6;padding:4px 8px; font-size: 11px; font-weight: 600;">Non-Null %</td>' + infoRows.map(({ nonNullPct }) => `<td style="border:1px solid #dee2e6;padding:4px 8px; font-size: 11px;">${nonNullPct}</td>`).join('');
            const infoDtypeRow = '<td style="border:1px solid #dee2e6;padding:4px 8px; font-size: 11px; font-weight: 600;">Dtype</td>' + infoRows.map(({ dtype }) => `<td style="border:1px solid #dee2e6;padding:4px 8px; font-size: 11px;">${dtype}</td>`).join('');
            let html = `<div style="padding: 15px; background: #fff; border-radius: 4px; border: 1px solid #dee2e6;">
                <h4 style="margin-top: 0; color: ${successColor};">✅ ${title}</h4>
                <div style="margin-bottom: 10px; font-size: 13px;">Input: ${stats.input_tables} tables, ${stats.input_rows} rows → Output: ${stats.output_rows} rows, ${stats.output_columns} cols (showing first ${showRowsLabel} rows)</div>
                ${extraHtml}
                <div style="position: relative;">
                    <div style="${INTEGRATION_TABLE_VIEWPORT_STYLE}" id="table-viewport-${downloadId}" title="Showing first ${displayRows.length} rows; download CSV for full table">
                        <table style="width: max-content; min-width: 100%; border-collapse: collapse; font-size: 12px;">
                            <thead><tr style="background: #f8f9fa; position: sticky; top: 0; z-index: 10;">
                                ${table.columns.map(col => `<th style="border: 1px solid #dee2e6; padding: 6px; text-align: left; background: #f8f9fa; white-space: nowrap;">${col}</th>`).join('')}
                            </tr></thead>
                            <tbody>
                                ${displayRows.map(row => `<tr>${row.map(cell => `<td style="border: 1px solid #dee2e6; padding: 6px; white-space: nowrap;">${cell != null && cell !== '' ? cell : ''}</td>`).join('')}</tr>`).join('')}
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
            const integrationType = (document.getElementById('integration_type') || {}).value || 'union';
            const k = parseInt((document.getElementById('integration_k') || {}).value, 10) || 10;
            const maxModels = parseInt((document.getElementById('integration_max_models') || {}).value, 10) || 10;
            const modelSearchMode = (document.getElementById('integration_model_search_mode') || {}).value || 'dense';
            const searchType = (document.getElementById('integration_search_type') || {}).value || 'single_column';
            
            const btn = document.getElementById('integrationRunBothBtn');
            const container = document.getElementById('integrationResultsContainer');
            const leftDiv = document.getElementById('integrationModelSearchResults');
            const rightDiv = document.getElementById('integrationResults');
            if (!btn || !container || !leftDiv || !rightDiv) return;
            
            btn.disabled = true;
            btn.textContent = '⏳ Integrating...';
            container.style.display = 'block';
            leftDiv.innerHTML = '<div style="padding: 8px; background: #f8f9fa; border-radius: 4px; font-size: 12px;">⏳ Waiting for Model Search integration...</div>';
            rightDiv.innerHTML = '<div style="padding: 8px; background: #f8f9fa; border-radius: 4px; font-size: 12px;">⏳ Waiting for Table Search integration...</div>';
            
            try {
                const tablesSource = (document.getElementById('integration_tables_source') || {}).value || 'intermediate';
                const modelReq = fetch('{{BACKEND_URL}}/api/integrate-model-search', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ job_id: jobId, integration_type: integrationType, k, max_models: maxModels, card2card_retrieval_mode: modelSearchMode })
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
                        const links = modelIds.map(m => `<a href="https://huggingface.co/${m}" target="_blank">${m}</a>`).join(', ');
                        extra = `<div style="margin-bottom: 10px; padding: 8px; background: #e7f3ff; border-radius: 4px; font-size: 12px;"><strong>Model IDs (Model Search):</strong> ${links} (${modelIds.length} models)</div>`;
                    } else {
                        extra = '<div style="margin-bottom: 10px; padding: 8px; background: #f8f9fa; border-radius: 4px; font-size: 12px; color: #6c757d;">Model IDs (Model Search): — (none or not available)</div>';
                    }
                    leftDiv.innerHTML = renderIntegrationTable(table, stats, { title: 'Model Search integration', successColor: '#007bff', extraHtml: extra, savedPath: modelRes.saved_path || '', downloadId: 'model-search' });
                } else {
                    leftDiv.innerHTML = `<div style="padding: 10px; border-radius: 4px; border: 1px solid #dc3545; color: #dc3545; font-size: 12px;">❌ ${modelRes.message || 'Integration failed'}</div>`;
                }
                initTablePanZoom(leftDiv);

                if (tableRes.status === 'success') {
                    const stats = tableRes.stats || {};
                    const table = tableRes.integrated_table;
                    let tableExtra = '';
                    const tableModelIds = tableRes.models_with_tables || [];
                    if (tableModelIds.length > 0) {
                        const links = tableModelIds.map(m => `<a href="https://huggingface.co/${m}" target="_blank">${m}</a>`).join(', ');
                        tableExtra = `<div style="margin-bottom: 10px; padding: 8px; background: #d4edda; border-radius: 4px; font-size: 12px;"><strong>Model IDs (Table Search):</strong> ${links} <span style="color:#155724;">(${tableModelIds.length} models)</span></div>`;
                    } else {
                        tableExtra = '<div style="margin-bottom: 10px; padding: 8px; background: #f8f9fa; border-radius: 4px; font-size: 12px; color: #6c757d;">Model IDs (Table Search): — (none or not available)</div>';
                    }
                    rightDiv.innerHTML = renderIntegrationTable(table, stats, { title: 'Table Search integration', successColor: '#28a745', extraHtml: tableExtra, savedPath: tableRes.saved_path || '', downloadId: 'table-search' });
                } else {
                    rightDiv.innerHTML = `<div style="padding: 10px; border-radius: 4px; border: 1px solid #dc3545; color: #dc3545; font-size: 12px;">❌ ${tableRes.message || 'Integration failed'}</div>`;
                }
                initTablePanZoom(rightDiv);
                const modelKey = getModelSearchKey(integrationType, modelSearchMode);
                const tableKey = getTableSearchKey(integrationType, searchType, tablesSource);
                const hasModelRun = modelRes.status === 'success' || modelRes.status === 'no_result';
                const hasTableRun = tableRes.status === 'success' || tableRes.status === 'no_result';
                if (hasModelRun || hasTableRun) {
                    if (hasModelRun) {
                        let runs = window.__modelSearchRuns || [];
                        const mPayload = { key: modelKey, integration_type: integrationType, card2card_retrieval_mode: modelSearchMode, k, max_models: maxModels, ...modelRes };
                        const idx = runs.findIndex(r => (r.key || getModelSearchKey(r.integration_type, r.card2card_retrieval_mode)) === modelKey);
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
        
        async function runIntegration(jobId) {
            const searchType = (document.getElementById('integration_search_type') || {}).value || 'single_column';
            const integrationType = (document.getElementById('integration_type') || {}).value || 'union';
            const k = parseInt((document.getElementById('integration_k') || {}).value, 10) || 10;
            const maxModels = parseInt((document.getElementById('integration_max_models') || {}).value, 10) || 10;
            
            const resultsDiv = document.getElementById('integrationResults');
            if (!resultsDiv) return;
            resultsDiv.innerHTML = '<div style="padding: 15px;">⏳ Running integration...</div>';
            try {
                const response = await fetch('{{BACKEND_URL}}/api/integrate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ job_id: jobId, search_type: searchType, integration_type: integrationType, k, max_models: maxModels })
                });
                const data = await response.json();
                if (data.status === 'success') {
                    const stats = data.stats || {};
                    const table = data.integrated_table;
                    const modelIds = data.models_with_tables || [];
                    let extra = '';
                    if (modelIds.length > 0) {
                        const links = modelIds.map(m => `<a href="https://huggingface.co/${m}" target="_blank">${m}</a>`).join(', ');
                        extra = `<div style="margin-bottom: 10px; padding: 8px; background: #d4edda; border-radius: 4px; font-size: 12px;"><strong>Model IDs (Table Search):</strong> ${links} (${modelIds.length} models)</div>`;
                    }
                    resultsDiv.innerHTML = renderIntegrationTable(table, stats, { title: 'Integration Successful', successColor: '#28a745', extraHtml: extra, savedPath: data.saved_path || '', downloadId: 'table-search-single' });
                    initTablePanZoom(resultsDiv);
                } else {
                    resultsDiv.innerHTML = `<div style="padding: 15px; border: 1px solid #dc3545; color: #dc3545;">❌ ${data.message || 'Unknown error'}</div>`;
                }
            } catch (error) {
                resultsDiv.innerHTML = `<div style="padding: 15px; border: 1px solid #dc3545; color: #dc3545;">❌ ${error.message}</div>`;
            }
        }
        
        async function runModelSearchIntegration(jobId) {
            const integrationType = (document.getElementById('integration_type') || {}).value || 'union';
            const k = parseInt((document.getElementById('integration_k') || {}).value, 10) || 10;
            const maxModels = parseInt((document.getElementById('integration_max_models') || {}).value, 10) || 10;
            const modelSearchMode = (document.getElementById('integration_model_search_mode') || {}).value || 'dense';
            const resultsDiv = document.getElementById('integrationModelSearchResults');
            if (!resultsDiv) return;
            resultsDiv.style.display = 'block';
            resultsDiv.innerHTML = '<div style="padding: 15px;">⏳ Running integration...</div>';
            try {
                const response = await fetch('{{BACKEND_URL}}/api/integrate-model-search', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ job_id: jobId, integration_type: integrationType, k, max_models: maxModels, card2card_retrieval_mode: modelSearchMode })
                });
                const data = await response.json();
                if (data.status === 'success') {
                    const stats = data.stats || {};
                    const table = data.integrated_table;
                    const modelIds = data.models_with_tables || [];
                    let extra = '';
                    if (modelIds.length > 0) {
                        const links = modelIds.map(m => `<a href="https://huggingface.co/${m}" target="_blank">${m}</a>`).join(', ');
                        extra = `<div style="margin-bottom: 10px; padding: 8px; background: #e7f3ff; border-radius: 4px; font-size: 12px;"><strong>Model IDs (Model Search):</strong> ${links} (${modelIds.length} models)</div>`;
                    } else {
                        extra = '<div style="margin-bottom: 10px; padding: 8px; background: #f8f9fa; border-radius: 4px; font-size: 12px; color: #6c757d;">Model IDs (Model Search): — (none or not available)</div>';
                    }
                    resultsDiv.innerHTML = renderIntegrationTable(table, stats, { title: 'Model Search integration', successColor: '#007bff', extraHtml: extra, savedPath: data.saved_path || '', downloadId: 'model-search-single' });
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
                                <div style="font-size: 14px; color: #666;">Model Search</div>
                                <div style="font-size: 28px; font-weight: bold; color: ${winner === 'model_search' ? '#28a745' : '#004085'};">${modelSearchScore}/100</div>
                                ${winner === 'model_search' ? '<div style="font-size: 11px; color: #28a745;">🏆 Winner</div>' : ''}
                                ${avgModelSearch != null ? `<div style="font-size: 10px; color: #666; margin-top: 4px;">Avg of sub-scores: ${avgModelSearch}/100</div>` : ''}
                            </div>
                            <div style="padding: 15px; background: ${winner === 'table_search' ? '#d4edda' : '#fff3cd'}; border-radius: 4px; border: 2px solid ${winner === 'table_search' ? '#28a745' : '#ffc107'};">
                                <div style="font-size: 14px; color: #666;">Table Search</div>
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
                                        <span>Model Search: <strong>${ss.model_search != null ? ss.model_search : '–'}/100</strong></span>
                                        <span>Table Search: <strong>${ss.table_search != null ? ss.table_search : '–'}/100</strong></span>
                                    </div>
                                    ${ss.evidence ? '<div style="font-size: 11px; color: #666; margin-top: 6px;">Evidence: ' + escTpl(ss.evidence) + '</div>' : ''}
                                </div>
                            `).join('')}
                        </div>
                    </div>
                    ` : ''}
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 20px;">
                        <div style="padding: 15px; background: #e7f3ff; border-radius: 4px; border-left: 4px solid #007bff;">
                            <h5 style="margin-top: 0; color: #004085; font-size: 13px;">Model Search</h5>
                            ${modelSearchAnalysis.strengths && modelSearchAnalysis.strengths.length > 0 ? '<div style="margin-top: 8px;"><strong style="font-size: 12px; color: #28a745;">Strengths:</strong><ul style="font-size: 11px; margin: 4px 0 0 0; padding-left: 20px;">' + modelSearchAnalysis.strengths.map(s => '<li>' + escTpl(s) + '</li>').join('') + '</ul></div>' : ''}
                            ${modelSearchAnalysis.weaknesses && modelSearchAnalysis.weaknesses.length > 0 ? '<div style="margin-top: 8px;"><strong style="font-size: 12px; color: #dc3545;">Weaknesses:</strong><ul style="font-size: 11px; margin: 4px 0 0 0; padding-left: 20px;">' + modelSearchAnalysis.weaknesses.map(w => '<li>' + escTpl(w) + '</li>').join('') + '</ul></div>' : ''}
                        </div>
                        <div style="padding: 15px; background: #fff3cd; border-radius: 4px; border-left: 4px solid #ffc107;">
                            <h5 style="margin-top: 0; color: #856404; font-size: 13px;">Table Search</h5>
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
                const evalBody = { job_id: jobId, use_fake: false };
                if (window.__selectedIntegrationKey) evalBody.integration_run_key = window.__selectedIntegrationKey;
                const evalPromise = sendEvaluationRequest(evalBody, evalResultsDiv, null);
                const qaTablePromise = sendQARequest({ job_id: jobId, use_table_search: true, use_fake: false }, qaResultsTable, null);
                const qaModelPromise = sendQARequest({ job_id: jobId, use_table_search: false, use_fake: false }, qaResultsModel, null);
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
                const body = { job_id: jobId, use_fake: false };
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
        
        async function copyTablePath(escapedTablePath, buttonElement) {
            // Decode HTML entities to get the actual path
            const decodedPath = escapedTablePath
                .replace(/&gt;/g, '>')
                .replace(/&lt;/g, '<')
                .replace(/&quot;/g, '"')
                .replace(/&#39;/g, "'")
                .replace(/&amp;/g, '&');
            
            try {
                // Use Clipboard API if available
                if (navigator.clipboard && navigator.clipboard.writeText) {
                    await navigator.clipboard.writeText(decodedPath);
                    const originalHTML = buttonElement.innerHTML;
                    buttonElement.innerHTML = '✓';
                    buttonElement.style.background = '#28a745';
                    setTimeout(() => {
                        buttonElement.innerHTML = originalHTML;
                        buttonElement.style.background = '#6c757d';
                    }, 1500);
                    console.log('✅ Copied to clipboard:', decodedPath);
                } else {
                    // Fallback for older browsers
                    const textArea = document.createElement('textarea');
                    textArea.value = decodedPath;
                    textArea.style.position = 'fixed';
                    textArea.style.opacity = '0';
                    document.body.appendChild(textArea);
                    textArea.select();
                    try {
                        document.execCommand('copy');
                        const originalHTML = buttonElement.innerHTML;
                        buttonElement.innerHTML = '✓';
                        buttonElement.style.background = '#28a745';
                        setTimeout(() => {
                            buttonElement.innerHTML = originalHTML;
                            buttonElement.style.background = '#6c757d';
                        }, 1500);
                        console.log('✅ Copied to clipboard (fallback):', decodedPath);
                    } catch (err) {
                        console.error('❌ Failed to copy:', err);
                        alert('Failed to copy. Path: ' + decodedPath);
                    }
                    document.body.removeChild(textArea);
                }
            } catch (error) {
                console.error('❌ Error copying to clipboard:', error);
                alert('Failed to copy. Path: ' + decodedPath);
            }
        }
        
        async function toggleTablePreview(tableExpandId, toggleElement) {
            const element = document.getElementById(tableExpandId);
            if (!element) {
                console.error('❌ Element not found:', tableExpandId);
                return;
            }
            
            // Get table path from data attribute
            const tablePath = toggleElement.getAttribute('data-table-path');
            if (!tablePath) {
                console.error('❌ No table path found in data-table-path attribute');
                console.error('   Toggle element:', toggleElement);
                console.error('   Available attributes:', Array.from(toggleElement.attributes).map(a => `${a.name}="${a.value}"`));
                return;
            }
            
            // Decode HTML entities if any (from &quot; etc.)
            // Decode &amp; last since other entities contain &
            const decodedPath = tablePath
                .replace(/&gt;/g, '>')
                .replace(/&lt;/g, '<')
                .replace(/&quot;/g, '"')
                .replace(/&#39;/g, "'")
                .replace(/&amp;/g, '&');  // Must be last
            
            console.log('🔄 Toggle table preview:', tableExpandId);
            console.log('   Raw path from attribute:', tablePath);
            console.log('   Decoded path:', decodedPath);
            console.log('   Current display:', element.style.display);
            
            // Check current state - simpler logic
            const isCurrentlyVisible = element.style.display === 'block' || (element.style.display === '' && element.offsetHeight > 0);
            const contentDiv = element.querySelector('div');
            const hasLoadedContent = contentDiv && 
                                   contentDiv.innerHTML && 
                                   !contentDiv.innerHTML.includes('Loading preview...') &&
                                   !contentDiv.innerHTML.includes('⏳') &&
                                   (contentDiv.innerHTML.includes('<table') || contentDiv.innerHTML.includes('Preview:'));
            
            console.log('   Is visible:', isCurrentlyVisible, 'Has content:', hasLoadedContent);
            
            if (isCurrentlyVisible && hasLoadedContent) {
                // Collapse if already loaded and visible
                console.log('   ➖ Collapsing...');
                element.style.display = 'none';
                toggleElement.textContent = '▶';
                toggleElement.classList.remove('expanded');
                toggleElement.style.transform = 'none';  // Reset any CSS rotation
            } else {
                // Expand and load
                console.log('   ➕ Expanding and loading...');
                element.style.display = 'block';
                toggleElement.textContent = '▼';
                toggleElement.classList.add('expanded');
                toggleElement.style.transform = 'none';  // Reset any CSS rotation (we use text, not rotation)
                
                // Get or create content div
                let div = element.querySelector('div');
                if (!div) {
                    div = document.createElement('div');
                    div.style.cssText = 'padding: 8px; background: white; border-radius: 4px; border: 1px solid #dee2e6;';
                    element.appendChild(div);
                }
                
                // Check if already loaded (has HTML table content, not just loading message)
                const currentHTML = div.innerHTML || '';
                const hasRealContent = currentHTML && 
                                     !currentHTML.includes('Loading preview...') && 
                                     !currentHTML.includes('⏳') &&
                                     (currentHTML.includes('<table') || currentHTML.includes('Preview:'));
                
                if (!hasRealContent) {
                    // Show loading state
                    div.innerHTML = '<div style="font-size: 11px; color: #999;">⏳ Loading preview...</div>';
                    
                    // Load preview
                    try {
                        // Use decoded path for the API call
                        const pathToSend = decodedPath || tablePath;
                        console.log('📊 Loading table preview for:', pathToSend);
                        const url = `{{BACKEND_URL}}/api/table-preview?path=${encodeURIComponent(pathToSend)}`;
                        console.log('📡 Request URL:', url);
                        
                        const response = await fetch(url);
                        
                        if (!response.ok) {
                            const errorText = await response.text();
                            console.error('❌ HTTP Error:', response.status, errorText);
                            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                        }
                        
                        const data = await response.json();
                        console.log('✅ Table preview response:', data);
                        
                        if (data.status === 'success') {
                            // Use HTML directly from backend
                            div.innerHTML = `
                                <div style="font-size: 10px; color: #999; margin-bottom: 5px;">
                                    Preview: ${data.rows} rows × ${data.columns} columns (first 5 rows, first 5 columns)
                                </div>
                                <div style="max-height: 200px; overflow: auto;">
                                    ${data.html || ''}
                                </div>
                            `;
                            console.log('✅ Table preview loaded successfully');
                        } else {
                            console.error('❌ API Error:', data.message);
                            div.innerHTML = `<div style="color: #dc3545; font-size: 11px;">❌ Error: ${data.message || 'Failed to load preview'}</div>`;
                        }
                    } catch (error) {
                        console.error('❌ Error loading table preview:', error);
                        div.innerHTML = `<div style="color: #dc3545; font-size: 11px;">❌ Error: ${error.message}</div>`;
                    }
                } else {
                    console.log('✅ Table preview already loaded, skipping fetch');
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
                const bodyTable = { job_id: jobId, use_table_search: true, use_fake: false };
                const bodyModel = { job_id: jobId, use_table_search: false, use_fake: false };
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
