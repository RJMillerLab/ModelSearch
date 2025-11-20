"""
Frontend for ModelSearch Demo

Simple web interface to compare Card2Card vs Card2Tab2Card search pipelines.
"""

import os
import sys
import json
import requests
from flask import Flask, render_template_string, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Backend API URL
BACKEND_URL = "http://localhost:5000"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>ModelSearch Demo</title>
    <meta charset="utf-8">
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            margin-bottom: 30px;
        }
        .input-section {
            margin-bottom: 30px;
        }
        .mode-selector {
            margin-bottom: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 4px;
        }
        .mode-option {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }
        .mode-option input[type="radio"] {
            margin-right: 8px;
        }
        .mode-option label {
            margin: 0;
            font-weight: normal;
            cursor: pointer;
        }
        .mode-input {
            margin-top: 15px;
            display: none;
        }
        .mode-input.active {
            display: block;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: bold;
            color: #555;
        }
        input[type="text"], input[type="number"] {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
            box-sizing: border-box;
        }
        button {
            background: #007bff;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            margin-top: 10px;
        }
        button:hover {
            background: #0056b3;
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .progress-section {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 4px;
            display: none;
        }
        .progress-section.active {
            display: block;
        }
        .log-container {
            max-height: 400px;
            overflow-y: auto;
            background: #1e1e1e;
            color: #d4d4d4;
            padding: 15px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            line-height: 1.6;
        }
        .log-entry {
            margin-bottom: 5px;
        }
        .log-timestamp {
            color: #858585;
            margin-right: 8px;
        }
        .log-message {
            color: #d4d4d4;
        }
        .results-section {
            margin-top: 30px;
            display: none;
        }
        .results-section.active {
            display: block;
        }
        .results-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .result-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 4px;
            border: 1px solid #dee2e6;
        }
        .result-card h3 {
            margin-top: 0;
            color: #495057;
        }
        .result-list {
            list-style: none;
            padding: 0;
        }
        .result-item {
            padding: 8px;
            margin: 5px 0;
            background: white;
            border-radius: 3px;
            border-left: 3px solid #007bff;
        }
        .comparison-section {
            margin-top: 20px;
            padding: 20px;
            background: #e7f3ff;
            border-radius: 4px;
        }
        .comparison-item {
            margin: 10px 0;
            padding: 10px;
            background: white;
            border-radius: 4px;
        }
        .expand-toggle {
            cursor: pointer;
            color: #007bff;
            font-weight: bold;
            padding: 2px 5px;
            margin-right: 5px;
            display: inline-block;
            user-select: none;
            transition: transform 0.2s;
            font-size: 12px;
        }
        .expand-toggle:hover {
            color: #0056b3;
        }
        .expand-toggle.expanded {
            transform: rotate(90deg);
        }
        .collapsible-content {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
        }
        .collapsible-content.expanded {
            max-height: 5000px;
        }
        .search-type-section {
            margin: 15px 0;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 4px;
        }
        .search-type-header {
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px;
            background: white;
            border-radius: 4px;
            margin-bottom: 10px;
        }
        .search-type-header:hover {
            background: #e9ecef;
        }
        .search-type-header::after {
            content: '▶';
            transition: transform 0.2s;
        }
        .search-type-header.expanded::after {
            transform: rotate(90deg);
        }
        .error {
            color: #dc3545;
            padding: 10px;
            background: #f8d7da;
            border-radius: 4px;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 ModelSearch Demo</h1>
        <p>Compare Card2Card (dense semantic) vs Card2Tab2Card (table-based) search</p>
        
        <div class="input-section">
            <div class="mode-selector">
                <label style="margin-bottom: 10px;">Search Mode:</label>
                <div class="mode-option">
                    <input type="radio" id="mode_query" name="search_mode" value="query" checked onchange="toggleMode()">
                    <label for="mode_query">Query → ModelCard → Search</label>
                </div>
                <div class="mode-option">
                    <input type="radio" id="mode_modelid" name="search_mode" value="modelid" onchange="toggleMode()">
                    <label for="mode_modelid">ModelID → Search (direct)</label>
                </div>
            </div>
            
            <div class="mode-input active" id="query-input">
                <label for="query">Query Text:</label>
                <input type="text" id="query" placeholder="e.g., transformer model for code generation" value="transformer model for code generation">
            </div>
            
            <div class="mode-input" id="modelid-input">
                <label for="model_id">Model ID:</label>
                <input type="text" id="model_id" value="Salesforce/codet5-base" placeholder="Enter HuggingFace model ID">
                <p style="font-size: 12px; color: #666; margin-top: 5px;">
                    Default CSV: data_citationlake/processed/deduped_github_csvs/0021c79d4e1a37579ca87328864d67a5_table_0.csv
                </p>
            </div>
            
            <label for="top_k" style="margin-top: 15px;">Top K Results (Final ModelCard Count):</label>
            <input type="number" id="top_k" value="20" min="1" max="100" oninput="updateTableSearchKDefault()">
            <p style="font-size: 11px; color: #666; margin-top: 3px;">
                Both pipelines will return this many model cards (Card2Card and Card2Tab2Card)
            </p>
            
            <label for="table_search_k" style="margin-top: 15px;">Table Search Top K (Card2Tab2Card Intermediate Step):</label>
            <div style="display: flex; align-items: center; gap: 10px;">
                <input type="range" id="table_search_k_slider" min="10" max="200" value="40" step="5" 
                       style="flex: 1;" oninput="updateTableSearchKValue(this.value)">
                <input type="number" id="table_search_k" value="40" min="10" max="200" 
                       style="width: 80px;" oninput="updateTableSearchKSlider(this.value)">
            </div>
            <p style="font-size: 11px; color: #666; margin-top: 3px;">
                Controls how many tables to retrieve in Card2Tab2Card search (intermediate step). 
                Final modelcard count is still controlled by "Top K Results" above.
            </p>
            
            <button id="searchBtn" onclick="startSearch()">Start Search</button>
        </div>
        
        <div id="progressSection" class="progress-section">
            <h3>Progress Logs</h3>
            <div id="logContainer" class="log-container"></div>
        </div>
        
        <div id="errorMsg" class="error" style="display: none;"></div>
        
        <div id="resultsSection" class="results-section">
            <h2>Results</h2>
            <div id="resultsContent"></div>
        </div>
    </div>
    
    <script>
        let currentJobId = null;
        let eventSource = null;
        
        function updateTableSearchKValue(value) {
            document.getElementById('table_search_k').value = value;
        }
        
        function updateTableSearchKSlider(value) {
            const slider = document.getElementById('table_search_k_slider');
            const numValue = parseInt(value);
            if (numValue >= parseInt(slider.min) && numValue <= parseInt(slider.max)) {
                slider.value = numValue;
            }
        }
        
        function updateTableSearchKDefault() {
            // When top_k changes, suggest a default table_search_k (2x top_k)
            const topK = parseInt(document.getElementById('top_k').value) || 20;
            const suggestedTableSearchK = topK * 2;
            // Only update if current value is close to old default (within 5)
            const currentTableSearchK = parseInt(document.getElementById('table_search_k').value) || 40;
            const oldTopK = Math.floor(currentTableSearchK / 2);
            if (Math.abs(currentTableSearchK - oldTopK * 2) <= 5) {
                // User hasn't manually adjusted much, update to new default
                const newValue = Math.min(Math.max(suggestedTableSearchK, 10), 200);
                document.getElementById('table_search_k').value = newValue;
                document.getElementById('table_search_k_slider').value = newValue;
            }
        }
        
        async function startSearch() {
            const mode = document.querySelector('input[name="search_mode"]:checked').value;
            const query = document.getElementById('query').value.trim();
            const modelId = document.getElementById('model_id').value.trim();
            const topK = parseInt(document.getElementById('top_k').value);
            const tableSearchK = parseInt(document.getElementById('table_search_k').value);
            
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
                    mode: mode,
                    top_k: topK,
                    table_search_k: tableSearchK
                };
                if (mode === 'query') {
                    requestBody.query = query;
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
            const mode = document.querySelector('input[name="search_mode"]:checked').value;
            const queryInput = document.getElementById('query-input');
            const modelIdInput = document.getElementById('modelid-input');
            
            if (mode === 'query') {
                queryInput.classList.add('active');
                modelIdInput.classList.remove('active');
            } else {
                queryInput.classList.remove('active');
                modelIdInput.classList.add('active');
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
            Object.keys(results.card2tab2card_results).forEach((type, idx) => {
                card2tab2cardIds[type] = 'card2tab2card-' + type + '-' + Date.now() + '-' + idx;
            });
            
            let html = `
                <div class="results-grid">
                    <div class="result-card">
                        <h3>Card2Card Results (${results.card2card_results.length})</h3>
                        <ul class="result-list">
                            ${results.card2card_results.slice(0, 10).map(m => `<li class="result-item">${formatModel(m)}</li>`).join('')}
                            ${results.card2card_results.length > 10 ? `
                                <li class="collapsible-content" id="${card2cardMoreId}">
                                    ${results.card2card_results.slice(10).map(m => `<div class="result-item">${formatModel(m)}</div>`).join('')}
                                </li>
                                <li>
                                    <span class="expand-toggle" onclick="toggleExpand('${card2cardMoreId}', this)">
                                        Show ${results.card2card_results.length - 10} more
                                    </span>
                                </li>
                            ` : ''}
                        </ul>
                    </div>
                    <div class="result-card">
                        <h3>Card2Tab2Card Results</h3>
                        ${Object.entries(results.card2tab2card_results).map(([type, data]) => {
                            const sectionId = card2tab2cardIds[type];
                            // Handle both old format (array) and new format (object with model_ids and intermediate)
                            const models = Array.isArray(data) ? data : (data.model_ids || []);
                            const intermediate = data.intermediate || {};
                            const tableToModels = intermediate.table_to_models || {};
                            
                            // Build reverse mapping: model_id -> list of tables
                            const modelToTables = {};
                            Object.entries(tableToModels).forEach(([table, modelList]) => {
                                modelList.forEach(modelId => {
                                    if (!modelToTables[modelId]) {
                                        modelToTables[modelId] = [];
                                    }
                                    modelToTables[modelId].push(table);
                                });
                            });
                            
                            return `
                                <div class="search-type-section">
                                    <div class="search-type-header" onclick="toggleSearchType('${sectionId}', this)">
                                        <h4>${type} (${models.length} models)</h4>
                                    </div>
                                    <div class="collapsible-content expanded" id="${sectionId}">
                                        <ul class="result-list" style="list-style: none; padding: 0;">
                                            ${models.length > 0 ? models.map((m, idx) => {
                                                const modelId = typeof m === 'string' ? m : (m.model_id || m);
                                                const modelUrl = typeof m === 'string' ? `https://huggingface.co/${modelId}` : (m.url || `https://huggingface.co/${modelId}`);
                                                const modelTables = modelToTables[modelId] || [];
                                                const modelExpandId = `${sectionId}-model-${idx}`;
                                                const hasTables = modelTables.length > 0;
                                                
                                                return `
                                                    <li class="result-item" style="margin-bottom: 8px;">
                                                        <div style="display: flex; align-items: center;">
                                                            <span class="expand-toggle" onclick="toggleExpand('${modelExpandId}', this)" style="margin-right: 8px; ${hasTables ? '' : 'display: none;'}">
                                                                ▶
                                                            </span>
                                                            <a href="${modelUrl}" target="_blank" style="color: #007bff; text-decoration: none; font-weight: 500;">
                                                                ${modelId}
                                                            </a>
                                                        </div>
                                                        ${hasTables ? `
                                                            <div class="collapsible-content" id="${modelExpandId}" style="margin-left: 20px; margin-top: 5px;">
                                                                <div style="font-size: 12px; color: #666;">
                                                                    <strong>From Tables (${modelTables.length}):</strong>
                                                                    <div style="margin-top: 5px; padding: 8px; background: #f8f9fa; border-radius: 4px; max-height: 300px; overflow-y: auto;">
                                                                        ${modelTables.map((table, tableIdx) => {
                                                                            const tableBasename = table.split('/').pop();
                                                                            const tableExpandId = `${modelExpandId}-table-${tableIdx}`;
                                                                            // Use data attribute to store table path safely
                                                                            return `
                                                                                <div style="padding: 4px 0; border-bottom: 1px solid #dee2e6;">
                                                                                    <div style="display: flex; align-items: center;">
                                                                                        <span class="expand-toggle" data-table-path="${table.replace(/"/g, '&quot;')}" onclick="toggleTablePreview('${tableExpandId}', this)" style="margin-right: 5px; font-size: 10px; cursor: pointer;">
                                                                                            ▶
                                                                                        </span>
                                                                                        <span style="font-weight: 500; color: #495057;">📄 ${tableBasename}</span>
                                                                                    </div>
                                                                                    <div class="collapsible-content" id="${tableExpandId}" style="margin-left: 20px; margin-top: 5px; display: none;">
                                                                                        <div style="padding: 8px; background: white; border-radius: 4px; border: 1px solid #dee2e6;">
                                                                                            <div style="font-size: 11px; color: #999;">Loading preview...</div>
                                                                                        </div>
                                                                                    </div>
                                                                                </div>
                                                                            `;
                                                                        }).join('')}
                                                                    </div>
                                                                </div>
                                                            </div>
                                                        ` : ''}
                                                    </li>
                                                `;
                                            }).join('') : '<li>No results</li>'}
                                        </ul>
                                    </div>
                                </div>
                            `;
                        }).join('')}
                    </div>
                </div>
            `;
            
            if (results.comparison) {
                html += '<div class="comparison-section"><h3>Comparison</h3>';
                html += Object.entries(results.comparison).map(([type, comp]) => `
                    <div class="comparison-item">
                        <strong>${type}:</strong> Overlap: ${comp.overlap_count} (${(comp.overlap_ratio * 100).toFixed(1)}%)
                    </div>
                `).join('');
                html += '</div>';
            }
            
            container.innerHTML = html;
            document.getElementById('resultsSection').classList.add('active');
        }
        
        function showError(message) {
            const errorDiv = document.getElementById('errorMsg');
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
        }
        
        function toggleExpand(elementId, toggleElement) {
            const element = document.getElementById(elementId);
            if (element) {
                element.classList.toggle('expanded');
                const isExpanded = element.classList.contains('expanded');
                const currentText = toggleElement.textContent;
                if (currentText.includes('Show')) {
                    const count = currentText.match(/Show (\d+)/)[1];
                    toggleElement.textContent = `Hide ${count} more`;
                } else {
                    const count = currentText.match(/Hide (\d+)/)[1];
                    toggleElement.textContent = `Show ${count} more`;
                }
            }
        }
        
        function toggleSearchType(sectionId, headerElement) {
            const element = document.getElementById(sectionId);
            if (element) {
                element.classList.toggle('expanded');
                headerElement.classList.toggle('expanded');
            }
        }
        
        async function toggleTablePreview(tableExpandId, toggleElement) {
            const element = document.getElementById(tableExpandId);
            if (!element) return;
            
            // Get table path from data attribute
            const tablePath = toggleElement.getAttribute('data-table-path');
            if (!tablePath) return;
            
            const isExpanded = element.style.display !== 'none';
            
            if (isExpanded) {
                // Collapse
                element.style.display = 'none';
                toggleElement.textContent = '▶';
                toggleElement.classList.remove('expanded');
            } else {
                // Expand
                element.style.display = 'block';
                toggleElement.textContent = '▼';
                toggleElement.classList.add('expanded');
                
                // Check if already loaded
                const contentDiv = element.querySelector('div');
                if (contentDiv && contentDiv.textContent.trim() === 'Loading preview...') {
                    // Load preview
                    try {
                        const response = await fetch(`{{BACKEND_URL}}/api/table-preview?path=${encodeURIComponent(tablePath)}`);
                        const data = await response.json();
                        
                        if (data.status === 'success') {
                            // Use HTML directly from backend
                            contentDiv.innerHTML = `
                                <div style="font-size: 10px; color: #999; margin-bottom: 5px;">
                                    Preview: ${data.rows} rows × ${data.columns} columns (first 5 rows, first 5 columns)
                                </div>
                                <div style="max-height: 200px; overflow: auto;">
                                    ${data.html || ''}
                                </div>
                            `;
                        } else {
                            contentDiv.innerHTML = '<div style="color: #dc3545; font-size: 11px;">Error: ' + (data.message || 'Failed to load preview') + '</div>';
                        }
                    } catch (error) {
                        contentDiv.innerHTML = '<div style="color: #dc3545; font-size: 11px;">Error: ' + error.message + '</div>';
                    }
                }
            }
        }
    </script>
</body>
</html>
""".replace('{{BACKEND_URL}}', BACKEND_URL)


@app.route('/')
def index():
    """Serve frontend HTML"""
    return render_template_string(HTML_TEMPLATE)


if __name__ == '__main__':
    print("Starting ModelSearch Frontend...")
    print("Open http://localhost:5001 in your browser")
    app.run(host='0.0.0.0', port=5001, debug=True)
