"""
Frontend for ModelSearch Demo

Simple web interface to compare Card2Card vs Card2Tab2Card search pipelines.
"""

import os
import sys
import json
import requests
from flask import Flask, render_template_string, jsonify, send_from_directory
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Get the project root directory (assuming frontend.py is in src/demo/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Backend API URL
BACKEND_URL = "http://localhost:5002"

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
        .form-row {
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
            margin-bottom: 10px;
        }
        .form-row label {
            margin-bottom: 0;
            min-width: 140px;
        }
        .form-row .form-control {
            flex: 1;
            min-width: 160px;
            padding: 6px 10px;
            font-size: 13px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .form-row select.form-control { width: auto; max-width: 320px; }
        .form-row input[type="number"].form-control { width: 80px; flex: none; }
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
            gap: 15px;
            align-items: start;
        }
        .result-card {
            background: #f8f9fa;
            padding: 12px;
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
            padding: 4px 6px;
            margin: 3px 0;
            background: white;
            border-radius: 3px;
            border-left: 3px solid #007bff;
            font-size: 13px;
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
            margin: 8px 0;
            padding: 0;
            background: #fff;
            border-radius: 8px;
            border: 1px solid #e9ecef;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            overflow: hidden;
        }
        .search-type-header {
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 12px;
            background: linear-gradient(180deg, #fafbfc 0%, #f1f3f5 100%);
            border-bottom: 1px solid #e9ecef;
            margin-bottom: 0;
        }
        .search-type-header:hover {
            background: linear-gradient(180deg, #f1f3f5 0%, #e9ecef 100%);
        }
        .search-type-section .collapsible-content {
            padding: 12px;
        }
        .search-type-header::before {
            content: '▶';
            margin-right: 5px;
            transition: transform 0.2s;
            display: inline-block;
        }
        .search-type-header.expanded::before {
            transform: rotate(90deg);
        }
        .error {
            color: #dc3545;
            padding: 10px;
            background: #f8d7da;
            border-radius: 4px;
            margin-top: 10px;
        }
        .number-badge {
            display: inline-block;
            width: 20px;
            height: 20px;
            background: black;
            color: white;
            border-radius: 50%;
            text-align: center;
            line-height: 20px;
            font-size: 12px;
            font-weight: bold;
            margin-right: 5px;
            vertical-align: middle;
        }
        .pdf-section {
            margin-bottom: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #dee2e6;
            text-align: center;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .pdf-section img {
            height: 240px;
            width: auto;
            max-width: 100%;
            object-fit: contain;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            background: white;
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 ModelSearch Demo</h1>
        <p>Compare <span class="number-badge">1</span> Card2Card (dense semantic) vs <span class="number-badge">2</span> Card2Tab2Card (table-based) search</p>
        
        <div class="input-section">
            <div style="margin-bottom: 15px; padding: 12px; background: #f8f9fa; border-radius: 4px; border: 1px solid #ddd;">
                <label style="display: flex; align-items: center; cursor: pointer;">
                    <input type="checkbox" id="load_previous_search" onchange="toggleLoadPrevious()" style="margin-right: 8px; width: 18px; height: 18px;">
                    <span style="font-weight: 500;">Load Previous Search</span>
                </label>
                <p style="font-size: 11px; color: #666; margin-top: 5px; margin-left: 26px;">
                    Check to load a previously saved search result instead of running a new search
                </p>
            </div>
            
            <div id="previous-search-section" style="display: none; margin-bottom: 20px;">
                <div style="margin-bottom: 15px; padding: 10px; background: #e7f3ff; border-radius: 4px;">
                    <button onclick="loadDemoExample()" style="padding: 8px 16px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; width: 100%; font-weight: 500; margin-bottom: 10px;">
                        🎨 Load Demo Example (Template)
                    </button>
                    <p style="font-size: 11px; color: #666; margin: 0;">
                        Or select from saved searches below:
                    </p>
                </div>
                
                <div style="margin-top: 15px;">
                    <label style="margin-bottom: 10px; display: block; font-weight: bold;">Saved Searches:</label>
                    <div id="saved_searches_list" style="max-height: 300px; overflow-y: auto; border: 1px solid #ddd; border-radius: 4px; padding: 10px; background: #f8f9fa;">
                        <div style="text-align: center; color: #666; padding: 20px;">
                            Loading saved searches...
                        </div>
                    </div>
                </div>
                <p style="margin-top: 12px; font-size: 13px;">
                    <a href="#" onclick="document.getElementById('load_previous_search').checked = false; toggleLoadPrevious(); return false;">Or run a new search</a>
                </p>
            </div>
            
            <div id="new-search-inputs">
            <div id="diagram-section" class="pdf-section" style="margin-bottom: 20px;">
                <img id="search-diagram" src="/static/fig/modelsearch.png" alt="ModelSearch Overview" />
            </div>
            
            <div class="form-row" style="margin-top: 15px;">
                <label for="search_mode_select">Search Mode:</label>
                <select id="search_mode_select" class="form-control" onchange="toggleMode()">
                    <option value="query" selected>Query → ModelCard → Search</option>
                    <option value="modelid">ModelID → Search (direct)</option>
                </select>
            </div>
            
            <div class="mode-input active" id="query-input">
                <div class="form-row">
                    <label for="query">Query (preset / fill in):</label>
                    <select id="preset_query_select" onchange="onPresetQueryChange()" class="form-control" style="width: 200px; flex: none;">
                        <option value="">— custom —</option>
                    </select>
                    <input type="text" id="query" class="form-control" placeholder="Type or pick preset" value="transformer model for code generation">
                </div>
                <div class="form-row" id="require-seed-has-tables-row" style="margin-top: 8px;">
                    <label for="require_seed_has_tables">Table Search Seed Model (Model with Tables / Model without Tables):</label>
                    <select id="require_seed_has_tables" class="form-control" style="width: 200px; flex: none;" title="Only applies in Query mode: pick which seed model to use for Table Search.">
                        <option value="0">Use top-1 result</option>
                        <option value="1" selected>Pick first model with tables (from top-20)</option>
                    </select>
                    <span style="font-size: 11px; color: #666; margin-left: 8px;">Only in Query mode. If "Pick first model with tables" and none of top-20 have tables → Table Search skipped.</span>
                </div>
            </div>
            
            <div class="mode-input" id="modelid-input">
                <div class="form-row">
                    <label for="model_id">Model ID (direct):</label>
                    <input type="text" id="model_id" class="form-control" value="Salesforce/codet5-base" placeholder="HuggingFace model ID">
                </div>
            </div>
            
            <div class="form-row one-click-info" style="background: #f0f7ff; border: 1px solid #b8d4e8; border-radius: 6px; padding: 10px 12px; margin-bottom: 8px;">
                <div style="font-size: 13px;">
                    <strong>One-click runs:</strong><br>
                    • Card2Card: Dense, Sparse, Hybrid<br>
                    • Card2Tab2Card: single_column, keyword, multi_column, unionable, complex, correlation, imputation, augmentation, dependent_data, feature_for_ml, multi_column_collinearity, negative_example<br>
                    <span style="color: #555;">Each method logs its elapsed time ⏱️ in the progress log when done.</span>
                </div>
            </div>
            
            <div class="form-row" style="display: flex; gap: 20px; align-items: flex-end; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 200px;">
                    <label for="top_k">Model Card Top K:</label>
                    <div style="display: flex; gap: 8px; align-items: center;">
                        <input type="range" id="top_k_slider" min="1" max="100" value="50" step="1" style="flex: 1; max-width: 200px;" oninput="updateTopKValue(this.value)">
                        <input type="number" id="top_k" class="form-control" value="50" min="1" max="100" oninput="updateTopKSlider(this.value)" style="width: 80px;">
                    </div>
                </div>
                <div style="flex: 1; min-width: 200px;">
                    <label for="table_search_k">Table Search Top K:</label>
                    <div style="display: flex; gap: 8px; align-items: center;">
                        <input type="range" id="table_search_k_slider" min="1" max="20" value="10" step="1" style="flex: 1; max-width: 200px;" oninput="updateTableSearchKValue(this.value)">
                        <input type="number" id="table_search_k" class="form-control" value="10" min="1" max="20" oninput="updateTableSearchKSlider(this.value)" style="width: 80px;">
                    </div>
                </div>
                <div>
                    <button id="searchBtn" onclick="startSearch()" style="padding: 10px 24px; font-size: 15px; font-weight: 600;">Start Search</button>
                </div>
            </div>
            </div>
        </div>
        
        <div id="progressSection" class="progress-section">
            <h3>Progress Logs</h3>
            <div id="logContainer" class="log-container"></div>
        </div>
        
        <div id="errorMsg" class="error" style="display: none;"></div>
        
        <div id="resultsSection" class="results-section">
            <h3 style="margin-top: 0; margin-bottom: 15px; font-size: 16px; font-weight: bold;">Retrieval results</h3>
            <div id="resultsContent"></div>
        </div>
    </div>
    
    <script>
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
            // Ensure "Load Previous Search" is unchecked by default
            const loadPreviousCheckbox = document.getElementById('load_previous_search');
            if (loadPreviousCheckbox) {
                loadPreviousCheckbox.checked = false;
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
                    const templateAvailable = !!data.template_available;
                    if (searches.length === 0 && !templateAvailable) {
                        html = '<div style="text-align: center; color: #666; padding: 20px;">No saved searches found. Run a new search to create one.</div>';
                    } else {
                        // Display each saved search as a clickable card
                        searches.forEach(search => {
                            const label = search.query 
                                ? `${search.timestamp_str} - ${search.query.substring(0, 50)}${search.query.length > 50 ? '...' : ''}`
                                : `${search.timestamp_str} - ${search.model_id}`;
                            
                            html += `
                                <div class="saved-search-item" onclick="loadSavedSearchFolder('${search.folder_name}')" 
                                     style="padding: 12px; margin-bottom: 8px; background: white; border: 1px solid #ddd; border-radius: 4px; cursor: pointer; transition: background 0.2s;"
                                     onmouseover="this.style.background='#e7f3ff'" 
                                     onmouseout="this.style.background='white'">
                                    <div style="font-weight: 500; color: #333; margin-bottom: 4px;">${label}</div>
                                    <div style="font-size: 11px; color: #666;">
                                        ${search.query ? `Query: ${search.query.substring(0, 60)}${search.query.length > 60 ? '...' : ''}` : `Model: ${search.model_id}`}
                                        ${search.top_k ? ` | Top K: ${search.top_k}` : ''}
                                    </div>
                                    <div style="font-size: 10px; color: #999; margin-top: 4px;">
                                        ${search.timestamp_str || search.timestamp || ''}
                                    </div>
                                </div>
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
                    // Display results directly
                    displayResults(data.results || data);
                    document.getElementById('progressSection').classList.remove('active');
                } else {
                    showError(data.message || 'Failed to load saved search');
                }
            } catch (error) {
                showError('Error: ' + error.message);
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
                    // Display results directly
                    displayResults(data.results || data);
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
                    // Display results directly
                    displayResults(data.results || data);
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
                showError('Please select a saved search from the list above, or use "Load Demo Example" button');
                return;
            }
            
            // Continue with new search logic
            
            const mode = (document.getElementById('search_mode_select') || {}).value || 'query';
            const query = document.getElementById('query').value.trim();
            const modelId = document.getElementById('model_id').value.trim();
            const topK = parseInt(document.getElementById('top_k').value, 10) || 50;
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
                    card2card_retrieval_mode: card2cardRetrievalMode
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
            // Seed model = from query step or user input (modelid mode). This is the ID used for this search, not a fixed placeholder.
            const seedModelId = results.model_id || null;
            const seedModelLine = seedModelId
                ? `<p style="margin: 0 0 12px 0; font-size: 14px;"><strong>Seed model (from query):</strong> <a href="https://huggingface.co/${seedModelId}" target="_blank">${seedModelId}</a></p>`
                : (results.error ? '' : `<p style="margin: 0 0 12px 0; font-size: 14px; color: #856404; background: #fff3cd; padding: 8px; border-radius: 4px;">⚠️ Model ID missing (something went wrong)</p>`);
            
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
                ${seedModelLine}
                <div class="results-grid">
                    <div class="result-card" style="min-width: 0;">
                        <h3 style="margin-top: 0; margin-bottom: 15px; font-size: 16px; color: #495057;">
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
                                            <p style="font-size: 11px; color: #666; margin: 10px 0 5px 0;">${modeInfo.desc}</p>
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
                    <div class="result-card" style="min-width: 0; box-shadow: 0 2px 8px rgba(0,0,0,0.06); border-radius: 8px;">
                        <h3 style="margin-top: 0; margin-bottom: 15px; font-size: 17px; font-weight: 600; color: #343a40;"><span class="number-badge">2</span> Card2Tab2Card Results</h3>
                        ${(() => {
                            // Filter: show keyword, unionable, and joinable types (single_column and multi_column are joinable)
                            const allowedTypes = ['keyword', 'unionable', 'single_column', 'multi_column'];
                            // Map joinable types to display names
                            const typeDisplayNames = {
                                'single_column': 'joinable (single_column)',
                                'multi_column': 'joinable (multi_column)',
                                'keyword': 'keyword',
                                'unionable': 'unionable'
                            };
                            const entries = Object.entries(results.card2tab2card_results)
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
                            
                            // Build reverse mapping: model_id -> list of tables
                            const modelToTables = {};
                            Object.entries(tableToModels).forEach(([table, modelList]) => {
                                // Handle both string arrays and object arrays
                                const normalizedModelList = Array.isArray(modelList) ? modelList : [];
                                normalizedModelList.forEach(modelIdOrObj => {
                                    // Extract model_id: handle both string and object formats
                                    let modelId = typeof modelIdOrObj === 'string' 
                                        ? modelIdOrObj 
                                        : (modelIdOrObj?.model_id || modelIdOrObj);
                                    
                                    // Normalize: trim whitespace and ensure it's a string
                                    if (modelId) {
                                        modelId = String(modelId).trim();
                                        if (modelId) {
                                            if (!modelToTables[modelId]) {
                                                modelToTables[modelId] = [];
                                            }
                                            modelToTables[modelId].push(table);
                                        }
                                    }
                                });
                            });
                            
                            // Debug: log the mapping
                            console.log(`[${type}] Built modelToTables mapping:`, Object.keys(modelToTables).length, 'models with tables');
                            console.log(`[${type}] Sample modelToTables:`, Object.entries(modelToTables).slice(0, 2));
                            console.log(`[${type}] tableToModels keys:`, Object.keys(tableToModels));
                            console.log(`[${type}] tableToModels sample:`, Object.entries(tableToModels).slice(0, 1));
                            console.log(`[${type}] models count:`, models.length);
                            console.log(`[${type}] models sample:`, models.slice(0, 2));
                            
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
                                            ${models.length > 0 ? models.map((m, idx) => {
                                                let modelId = typeof m === 'string' ? m : (m.model_id || m);
                                                // Normalize modelId: trim whitespace and ensure it's a string
                                                modelId = String(modelId).trim();
                                                const modelUrl = typeof m === 'string' ? `https://huggingface.co/${modelId}` : (m.url || `https://huggingface.co/${modelId}`);
                                                const modelTables = modelToTables[modelId] || [];
                                                const modelExpandId = `${sectionId}-model-${idx}`;
                                                const hasTables = modelTables.length > 0;
                                                
                                                // Debug: log if model has tables
                                                if (hasTables) {
                                                    console.log(`[${type}] Model ${modelId} has ${modelTables.length} tables:`, modelTables.slice(0, 2));
                                                } else {
                                                    console.log(`[${type}] Model ${modelId} has NO tables. modelToTables keys:`, Object.keys(modelToTables));
                                                    console.log(`[${type}] Looking for modelId: "${modelId}" in modelToTables`);
                                                }
                                                
                                                return `
                                                    <li class="result-item" style="margin-bottom: 4px;">
                                                        <div style="display: flex; align-items: center;">
                                                            <span class="expand-toggle" onclick="toggleExpand('${modelExpandId}', this)" style="margin-right: 6px; ${hasTables ? '' : 'display: none;'}">
                                                                ▶
                                                            </span>
                                                            <a href="${modelUrl}" target="_blank" style="color: #007bff; text-decoration: none; font-weight: 500; font-size: 13px;">
                                                                ${modelId}
                                                            </a>
                                                        </div>
                                                        ${hasTables ? `
                                                            <div class="collapsible-content" id="${modelExpandId}" style="margin-left: 15px; margin-top: 2px; display: none;">
                                                                <div style="font-size: 10px; color: #666;">
                                                                    <strong>From Tables (${modelTables.length}):</strong>
                                                                    <div style="margin-top: 2px; padding: 4px; background: #f8f9fa; border-radius: 4px; max-height: 200px; overflow-y: auto;">
                                                                        ${modelTables.map((table, tableIdx) => {
                                                                            const tableBasename = table.split('/').pop();
                                                                            const tableExpandId = `${modelExpandId}-table-${tableIdx}`;
                                                                            // Escape the table path for HTML attribute (handle quotes and other special chars)
                                                                            // First escape HTML entities, then escape quotes for attribute
                                                                            const escapedTablePath = String(table)
                                                                                .replace(/&/g, '&amp;')
                                                                                .replace(/"/g, '&quot;')
                                                                                .replace(/'/g, '&#39;')
                                                                                .replace(/</g, '&lt;')
                                                                                .replace(/>/g, '&gt;');
                                                                            return `
                                                                                <div style="padding: 1px 0; border-bottom: 1px solid #dee2e6;">
                                                                                    <div style="display: flex; align-items: center; gap: 5px;">
                                                                                        <span style="font-size: 8px; color: #999; font-family: monospace; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; line-height: 1.2;" title="${table}">
                                                                                            ${table}
                                                                                        </span>
                                                                                        <button onclick="copyTablePath('${escapedTablePath}', this)" 
                                                                                                style="padding: 2px 4px; font-size: 11px; background: #6c757d; color: white; border: none; border-radius: 3px; cursor: pointer; min-width: 20px; height: 20px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;"
                                                                                                title="Copy full path to clipboard">
                                                                                            📋
                                                                                        </button>
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
            
            // Single merged Table Integration section: one set of settings, one button runs both in parallel
            const integrationCardStyle = 'padding: 24px; background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%); border-radius: 10px; border: 1px solid #dee2e6; box-shadow: 0 4px 12px rgba(0,0,0,0.06); font-family: inherit; font-size: 14px; color: #212529; min-width: 0;';
            const integrationTitleStyle = 'margin-top: 0; font-size: 18px; font-weight: 600; color: #1a1d21; letter-spacing: -0.02em;';
            const integrationDescStyle = 'font-size: 14px; color: #5a6268; margin-bottom: 18px; line-height: 1.5;';
            const topKLabelStyle = 'display: block; margin-bottom: 4px; font-weight: 500; color: #212529;';
            html += `
                <div class="integration-section" style="${integrationCardStyle}; margin-top: 30px;">
                    <h3 style="${integrationTitleStyle}">Table Integration</h3>
                    <p style="${integrationDescStyle}">Run both: <span class="number-badge">1</span> Card2Card (model search) + <span class="number-badge">2</span> Card2Tab2Card (table search). Shared settings below; one button runs both in parallel. Table search returns <strong>all models</strong> that contain the retrieved tables (no model cap).</p>
                    <div style="margin-bottom: 12px;">
                        <div style="display: flex; gap: 16px; align-items: flex-end; flex-wrap: wrap; margin-bottom: 10px;">
                            <div style="flex: 0 0 auto;"><label style="${topKLabelStyle}">integration method:</label><select id="integration_type" class="form-control" style="width: 140px; box-sizing: border-box;">
                                <option value="union">Union</option>
                                <option value="intersection">Intersection</option>
                                <option value="alite">ALITE</option>
                                <option value="outer_join">Outer Join</option>
                            </select></div>
                            <div style="flex: 0 0 auto;"><label for="integration_k" style="${topKLabelStyle}">top k tables:</label><input type="number" id="integration_k" class="form-control" value="10" min="1" max="50" style="width: 140px; box-sizing: border-box;"></div>
                            <div style="flex: 0 0 auto;"><label for="integration_max_models" style="${topKLabelStyle}">top k models:</label><input type="number" id="integration_max_models" class="form-control" value="10" min="1" max="50" style="width: 140px; box-sizing: border-box;"></div>
                        </div>
                        <div style="display: flex; gap: 16px; align-items: flex-end; flex-wrap: wrap;">
                            <div style="flex: 0 0 auto;"><label style="${topKLabelStyle}">Model Search option:</label><select id="integration_model_search_mode" class="form-control" style="width: 140px; box-sizing: border-box;">
                                <option value="dense">Dense</option>
                                <option value="sparse">Sparse</option>
                                <option value="hybrid">Hybrid</option>
                            </select></div>
                            <div style="flex: 0 0 auto;"><label style="${topKLabelStyle}">Table Search option:</label><select id="integration_search_type" class="form-control" style="width: 140px; box-sizing: border-box;">
                                <option value="single_column">Single Column</option>
                                <option value="keyword">Keyword</option>
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
                    </div>
                    <div style="margin-top: 12px;">
                        <button id="integrationRunBothBtn" onclick="runBothIntegrations('${results.job_id || currentJobId}')" style="padding: 10px 24px; font-size: 15px; font-weight: 600;">Run Both Integrations</button>
                    </div>
                    <div id="integrationResultsContainer" style="margin-top: 20px; display: none;">
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                            <div><h4 style="margin: 0 0 8px 0; font-size: 14px; color: #495057;">From Model Search</h4><div id="integrationModelSearchResults"></div></div>
                            <div><h4 style="margin: 0 0 8px 0; font-size: 14px; color: #495057;">From Table Search</h4><div id="integrationResults"></div></div>
                        </div>
                    </div>
                </div>
            `;
            
            // Add Evaluation Section - comparison merged into same card, then evaluation
            html += `
                <div class="evaluation-section" style="margin-top: 30px; padding: 20px; background: #fff3cd; border-radius: 8px; border: 2px solid #ffc107; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    ${comparisonHtml}
                    <h3 style="margin-top: 0; color: #856404; font-size: 16px;">📊 Evaluation on Integrated Tables</h3>
                    <p style="font-size: 14px; color: #666; margin-bottom: 15px;">
                        Evaluate diversity between Table Search Integration and Model Search Integration results using LLM.
                    </p>
                    <div style="display: flex; gap: 15px; align-items: center; flex-wrap: wrap; margin-bottom: 15px;">
                        <label style="display: flex; align-items: center; gap: 5px; font-weight: 500;">
                            <input type="radio" name="evaluation_mode" value="generate" id="eval_mode_generate" checked onchange="toggleEvaluationMode()" style="width: 18px; height: 18px;">
                            <span>Generate New Response</span>
                        </label>
                        <label style="display: flex; align-items: center; gap: 5px; font-weight: 500;">
                            <input type="radio" name="evaluation_mode" value="use_fake" id="eval_mode_fake" onchange="toggleEvaluationMode()" style="width: 18px; height: 18px;">
                            <span>Use Fake Response for testing/demo</span>
                        </label>
                    </div>
                    <div id="evaluation_generate_options" style="display: block;">
                        <button id="evaluationBtn" onclick="runEvaluation('${results.job_id || currentJobId}')" 
                                style="padding: 8px 16px; background: #ffc107; color: #000; border: none; border-radius: 4px; cursor: pointer; font-weight: 500; width: 100%;">
                            📊 Generate Evaluation
                        </button>
                    </div>
                    <div id="evaluation_use_fake_options" style="display: none;">
                        <div style="margin-bottom: 10px;">
                            <input type="file" id="evaluation_fake_file2" accept=".json" style="display: none;" onchange="handleFakeFileSelect()">
                            <button type="button" onclick="document.getElementById('evaluation_fake_file2').click()" 
                                    style="padding: 8px 16px; background: #6c757d; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: 500; width: 100%;">
                                📁 Load Fake Response File
                            </button>
                            <span id="fake_file_name2" style="font-size: 11px; color: #666;"></span>
                        </div>
                        <button id="evaluationBtnFake" onclick="runEvaluation('${results.job_id || currentJobId}')" 
                                style="padding: 8px 16px; background: #6c757d; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: 500; width: 100%;">
                            📊 Use Fake Response
                        </button>
                    </div>
                    <div id="evaluationResults" style="margin-top: 20px; display: none;"></div>
                    </div>
                    
                    <!-- QA Section: fake choice + one button; after click show integrated tables + answers (two columns) -->
                    <div class="qa-section" style="margin-top: 30px; padding: 20px; background: #d1ecf1; border-radius: 8px; border: 2px solid #17a2b8; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        <h3 style="margin-top: 0; color: #0c5460;">💬 Question Answering (QA)</h3>
                        <p style="font-size: 14px; color: #666; margin-bottom: 15px;">
                            QA on both integrated tables. One button generates two answers for comparison.
                        </p>
                        <div style="display: flex; gap: 15px; align-items: center; flex-wrap: wrap; margin-bottom: 15px;">
                            <label style="display: flex; align-items: center; gap: 5px; font-weight: 500;">
                                <input type="radio" name="qa_mode" value="generate" id="qa_mode_generate" checked onchange="toggleQAMode()" style="width: 18px; height: 18px;">
                                <span>Generate New Answer</span>
                            </label>
                            <label style="display: flex; align-items: center; gap: 5px; font-weight: 500;">
                                <input type="radio" name="qa_mode" value="use_fake" id="qa_mode_fake" onchange="toggleQAMode()" style="width: 18px; height: 18px;">
                                <span>Use Fake Response for testing/demo</span>
                            </label>
                            <div id="qa_use_fake_options" style="display: none; width: 100%; margin-top: 8px;">
                                <input type="file" id="qa_fake_file" accept=".json" onchange="handleQAFakeFileSelect()" style="font-size: 13px;">
                                <span id="qa_fake_file_name" style="font-size: 12px; color: #666;"></span>
                            </div>
                        </div>
                        <button id="qaBtn" onclick="runQABoth('${results.job_id || currentJobId}')" 
                                style="padding: 8px 16px; background: #17a2b8; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: 500; width: 100%;">
                            💬 Generate Answer
                        </button>
                        <div id="qa_after_click" style="display: none; margin-top: 20px;">
                            <h4 style="margin: 0 0 10px 0; font-size: 14px; color: #0c5460;">Integrated tables</h4>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 15px;">
                                <div><strong style="font-size: 13px; color: #0c5460;">Table Search Integration</strong></div>
                                <div><strong style="font-size: 13px; color: #0c5460;">Model Search Integration</strong></div>
                            </div>
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
            document.getElementById('resultsSection').classList.add('active');
        }
        
        // Fixed viewport: table never expands the page; user scrolls inside this window only
        const INTEGRATION_TABLE_VIEWPORT_STYLE = 'height: 320px; width: 100%; max-width: 100%; overflow-x: auto; overflow-y: auto; border: 1px solid #dee2e6; border-radius: 6px; background: #fff; resize: none; position: relative;';
        window.__integrationTables = window.__integrationTables || {};
        window.__tableViewports = window.__tableViewports || {};
        
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
        
        function renderIntegrationTable(table, stats, options) {
            const { title = 'Integration', successColor = '#28a745', extraHtml = '', savedPath = '', downloadId = '' } = options || {};
            if (!table || (stats && stats.output_rows === 0)) {
                return `<div style="padding: 15px; background: #fff; border-radius: 4px; border: 1px solid #dee2e6;">
                    <h4 style="margin-top: 0; color: ${successColor};">✅ ${title}</h4>
                    <div style="margin-bottom: 15px;">${(stats && stats.output_columns === 0) ? '⚠️ No common columns. Intersection result is empty.' : '⚠️ No common rows. Intersection result is empty.'}</div></div>`;
            }
            if (downloadId) window.__integrationTables[downloadId] = table;
            const footer = [];
            if (savedPath) footer.push(`<span style="font-size: 12px; color: #666;">Saved to: <code style="background: #f1f3f5; padding: 2px 6px; border-radius: 4px;">${savedPath}</code></span>`);
            footer.push(`<button type="button" onclick="downloadIntegrationTableAsCsv('${downloadId}')" style="margin-left: 10px; padding: 6px 12px; font-size: 13px; background: #28a745; color: white; border: none; border-radius: 6px; cursor: pointer;">📥 Download CSV</button>`);
            let html = `<div style="padding: 15px; background: #fff; border-radius: 4px; border: 1px solid #dee2e6;">
                <h4 style="margin-top: 0; color: ${successColor};">✅ ${title}</h4>
                <div style="margin-bottom: 10px; font-size: 13px;">Input: ${stats.input_tables} tables, ${stats.input_rows} rows → Output: ${stats.output_rows} rows, ${stats.output_columns} cols</div>
                ${extraHtml}
                <div style="position: relative;">
                    <div style="${INTEGRATION_TABLE_VIEWPORT_STYLE}" id="table-viewport-${downloadId}" title="Scroll inside this window to see the full table">
                        <table style="width: max-content; min-width: 100%; border-collapse: collapse; font-size: 12px;">
                            <thead><tr style="background: #f8f9fa; position: sticky; top: 0; z-index: 10;">
                                ${table.columns.map(col => `<th style="border: 1px solid #dee2e6; padding: 6px; text-align: left; background: #f8f9fa;">${col}</th>`).join('')}
                            </tr></thead>
                            <tbody>
                                ${table.data.map(row => `<tr>${row.map(cell => `<td style="border: 1px solid #dee2e6; padding: 6px;">${cell != null && cell !== '' ? cell : ''}</td>`).join('')}</tr>`).join('')}
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
            btn.textContent = '⏳ Running both...';
            container.style.display = 'block';
            leftDiv.innerHTML = '<div style="padding: 12px; background: #f8f9fa; border-radius: 4px;">⏳ Model Search integration...</div>';
            rightDiv.innerHTML = '<div style="padding: 12px; background: #f8f9fa; border-radius: 4px;">⏳ Table Search integration...</div>';
            
            const fetchModelSearch = fetch('{{BACKEND_URL}}/api/integrate-model-search', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ job_id: jobId, integration_type: integrationType, k, max_models: maxModels, card2card_retrieval_mode: modelSearchMode })
            }).then(r => r.json());
            const fetchTableSearch = fetch('{{BACKEND_URL}}/api/integrate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ job_id: jobId, search_type: searchType, integration_type: integrationType, k, max_models: maxModels })
            }).then(r => r.json());
            
            try {
                const [modelRes, tableRes] = await Promise.all([fetchModelSearch, fetchTableSearch]);
                
                if (modelRes.status === 'success') {
                    const stats = modelRes.stats || {};
                    const table = modelRes.integrated_table;
                    let extra = '';
                    if (modelRes.models_with_tables && modelRes.models_with_tables.length > 0) {
                        extra = `<div style="margin-bottom: 10px; padding: 8px; background: #e7f3ff; border-radius: 4px; font-size: 12px;">Models with tables: ${modelRes.models_with_tables.slice(0, 5).join(', ')}${modelRes.models_with_tables.length > 5 ? ' ...' : ''}</div>`;
                    }
                    leftDiv.innerHTML = renderIntegrationTable(table, stats, { title: 'Model Search integration', successColor: '#007bff', extraHtml: extra, savedPath: modelRes.saved_path || '', downloadId: 'model-search' });
                } else {
                    leftDiv.innerHTML = `<div style="padding: 15px; border-radius: 4px; border: 1px solid #dc3545; color: #dc3545;">❌ ${modelRes.message || 'Integration failed'}</div>`;
                }
                
                if (tableRes.status === 'success') {
                    const stats = tableRes.stats || {};
                    const table = tableRes.integrated_table;
                    rightDiv.innerHTML = renderIntegrationTable(table, stats, { title: 'Table Search integration', successColor: '#28a745', savedPath: tableRes.saved_path || '', downloadId: 'table-search' });
                } else {
                    rightDiv.innerHTML = `<div style="padding: 15px; border-radius: 4px; border: 1px solid #dc3545; color: #dc3545;">❌ ${tableRes.message || 'Integration failed'}</div>`;
                }
            } catch (err) {
                leftDiv.innerHTML = `<div style="padding: 15px; border: 1px solid #dc3545; color: #dc3545;">❌ ${err.message}</div>`;
                rightDiv.innerHTML = `<div style="padding: 15px; border: 1px solid #dc3545; color: #dc3545;">❌ ${err.message}</div>`;
            } finally {
                btn.disabled = false;
                btn.textContent = '🔗 Run Both Integrations (Model Search + Table Search)';
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
                    const stats = data.stats;
                    const table = data.integrated_table;
                    resultsDiv.innerHTML = renderIntegrationTable(table, stats, { title: 'Integration Successful', successColor: '#28a745', savedPath: data.saved_path || '', downloadId: 'table-search-single' });
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
                    let extra = '';
                    if (data.models_with_tables && data.models_with_tables.length > 0) {
                        extra = `<div style="margin-bottom: 10px; padding: 8px; background: #e7f3ff; border-radius: 4px; font-size: 12px;">Models with tables: ${data.models_with_tables.slice(0, 5).map(m => `<a href="https://huggingface.co/${m}" target="_blank">${m}</a>`).join(', ')}${data.models_with_tables.length > 5 ? ' ...' : ''}</div>`;
                    }
                    resultsDiv.innerHTML = renderIntegrationTable(table, stats, { title: 'Model Search integration', successColor: '#007bff', extraHtml: extra, savedPath: data.saved_path || '', downloadId: 'model-search-single' });
                } else {
                    resultsDiv.innerHTML = `<div style="padding: 15px; border: 1px solid #dc3545; color: #dc3545;">❌ ${data.message || 'Unknown error'}</div>`;
                }
            } catch (error) {
                resultsDiv.innerHTML = `<div style="padding: 15px; border: 1px solid #dc3545; color: #dc3545;">❌ ${error.message}</div>`;
            }
        }
        
        let fakeResponseFile = null;
        let qaFakeResponseFile = null;
        
        function toggleQAMode() {
            const fakeMode = document.getElementById('qa_mode_fake')?.checked || false;
            const fakeOptions = document.getElementById('qa_use_fake_options');
            if (fakeOptions) fakeOptions.style.display = fakeMode ? 'block' : 'none';
        }
        
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
        
        function toggleEvaluationMode() {
            const generateMode = document.getElementById('eval_mode_generate').checked;
            const fakeMode = document.getElementById('eval_mode_fake').checked;
            const generateOptions = document.getElementById('evaluation_generate_options');
            const fakeOptions = document.getElementById('evaluation_use_fake_options');
            
            if (generateMode) {
                if (generateOptions) generateOptions.style.display = 'block';
                if (fakeOptions) fakeOptions.style.display = 'none';
            } else if (fakeMode) {
                if (generateOptions) generateOptions.style.display = 'none';
                if (fakeOptions) fakeOptions.style.display = 'block';
            }
        }
        
        function handleFakeFileSelect() {
            const fileInput = document.getElementById('evaluation_fake_file');
            const fileNameSpan = document.getElementById('fake_file_name');
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
            
            // Get comparison scores - Note: model_search is LEFT, table_search is RIGHT
            const comparisonScore = eval_result.comparison_score || {};
            const modelSearchScore = comparisonScore.model_search_quality || eval_result.model_search_quality || 'N/A';
            const tableSearchScore = comparisonScore.table_search_quality || eval_result.table_search_quality || 'N/A';
            const winner = comparisonScore.winner || eval_result.winner || 'N/A';
            const difference = comparisonScore.overall_difference || eval_result.overall_difference || 'N/A';
            
            const qualityAnalysis = eval_result.quality_analysis || {};
            const modelSearchAnalysis = qualityAnalysis.model_search || {};
            const tableSearchAnalysis = qualityAnalysis.table_search || {};
            
            let html = `
                <div style="padding: 15px; background: #fff; border-radius: 4px; border: 1px solid #dee2e6;">
                    <h4 style="margin-top: 0; color: #ffc107; margin-bottom: 15px;">📊 Quality Comparison - Table Search vs Model Search</h4>
                    
                    <!-- Quality Score Comparison - Left: Model Search, Right: Table Search -->
                    <div style="margin-bottom: 20px; padding: 15px; background: #f8f9fa; border-radius: 4px;">
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 15px;">
                            <div style="padding: 15px; background: ${winner === 'model_search' ? '#d4edda' : '#e7f3ff'}; border-radius: 4px; border: 2px solid ${winner === 'model_search' ? '#28a745' : '#007bff'};">
                                <div style="font-size: 14px; color: #666; margin-bottom: 5px;">Model Search Quality</div>
                                <div style="font-size: 32px; font-weight: bold; color: ${winner === 'model_search' ? '#28a745' : '#004085'};">
                                    ${modelSearchScore}/100
                                </div>
                                ${winner === 'model_search' ? '<div style="font-size: 11px; color: #28a745; margin-top: 5px;">🏆 Winner</div>' : ''}
                            </div>
                            <div style="padding: 15px; background: ${winner === 'table_search' ? '#d4edda' : '#fff3cd'}; border-radius: 4px; border: 2px solid ${winner === 'table_search' ? '#28a745' : '#ffc107'};">
                                <div style="font-size: 14px; color: #666; margin-bottom: 5px;">Table Search Quality</div>
                                <div style="font-size: 32px; font-weight: bold; color: ${winner === 'table_search' ? '#28a745' : '#856404'};">
                                    ${tableSearchScore}/100
                                </div>
                                ${winner === 'table_search' ? '<div style="font-size: 11px; color: #28a745; margin-top: 5px;">🏆 Winner</div>' : ''}
                            </div>
                        </div>
                        <div style="text-align: center; padding: 10px; background: white; border-radius: 4px;">
                            <div style="font-size: 14px; color: #666;">Quality Difference</div>
                            <div style="font-size: 20px; font-weight: bold; color: #dc3545;">
                                ${difference > 0 ? '+' : ''}${difference} points
                            </div>
                        </div>
                    </div>
                    
                    <!-- Quality Analysis - Left: Model Search, Right: Table Search -->
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 20px;">
                        <div style="padding: 15px; background: #e7f3ff; border-radius: 4px; border-left: 4px solid #007bff;">
                            <h5 style="margin-top: 0; color: #004085;">Model Search Analysis</h5>
                            ${modelSearchAnalysis.summary ? `<p style="font-size: 12px; margin-bottom: 10px;">${modelSearchAnalysis.summary}</p>` : ''}
                            ${modelSearchAnalysis.strengths && modelSearchAnalysis.strengths.length > 0 ? `
                                <div style="margin-top: 10px;">
                                    <strong style="font-size: 12px; color: #28a745;">Strengths:</strong>
                                    <ul style="font-size: 11px; margin: 5px 0 0 0; padding-left: 20px;">
                                        ${modelSearchAnalysis.strengths.map(s => `<li>${s}</li>`).join('')}
                                    </ul>
                                </div>
                            ` : ''}
                            ${modelSearchAnalysis.weaknesses && modelSearchAnalysis.weaknesses.length > 0 ? `
                                <div style="margin-top: 10px;">
                                    <strong style="font-size: 12px; color: #dc3545;">Weaknesses:</strong>
                                    <ul style="font-size: 11px; margin: 5px 0 0 0; padding-left: 20px;">
                                        ${modelSearchAnalysis.weaknesses.map(w => `<li>${w}</li>`).join('')}
                                    </ul>
                                </div>
                            ` : ''}
                        </div>
                        <div style="padding: 15px; background: #fff3cd; border-radius: 4px; border-left: 4px solid #ffc107;">
                            <h5 style="margin-top: 0; color: #856404;">Table Search Analysis</h5>
                            ${tableSearchAnalysis.summary ? `<p style="font-size: 12px; margin-bottom: 10px;">${tableSearchAnalysis.summary}</p>` : ''}
                            ${tableSearchAnalysis.strengths && tableSearchAnalysis.strengths.length > 0 ? `
                                <div style="margin-top: 10px;">
                                    <strong style="font-size: 12px; color: #28a745;">Strengths:</strong>
                                    <ul style="font-size: 11px; margin: 5px 0 0 0; padding-left: 20px;">
                                        ${tableSearchAnalysis.strengths.map(s => `<li>${s}</li>`).join('')}
                                    </ul>
                                </div>
                            ` : ''}
                            ${tableSearchAnalysis.weaknesses && tableSearchAnalysis.weaknesses.length > 0 ? `
                                <div style="margin-top: 10px;">
                                    <strong style="font-size: 12px; color: #dc3545;">Weaknesses:</strong>
                                    <ul style="font-size: 11px; margin: 5px 0 0 0; padding-left: 20px;">
                                        ${tableSearchAnalysis.weaknesses.map(w => `<li>${w}</li>`).join('')}
                                    </ul>
                                </div>
                            ` : ''}
                        </div>
                    </div>
                    
                    <!-- Comparison Summary -->
                    ${eval_result.comparison_summary ? `
                        <div style="margin-top: 15px; padding: 12px; background: #e7f3ff; border-radius: 4px; border-left: 4px solid #007bff;">
                            <strong>Comparison Summary:</strong>
                            <p style="margin: 8px 0 0 0; font-size: 13px; line-height: 1.6;">${eval_result.comparison_summary}</p>
                        </div>
                    ` : ''}
                    ${eval_result.key_differences && eval_result.key_differences.length > 0 ? `
                        <div style="margin-top: 15px; padding: 12px; background: #e7f3ff; border-radius: 4px;">
                            <strong>Key Differences:</strong>
                            <ul style="margin: 8px 0 0 0; padding-left: 20px; font-size: 13px;">
                                ${eval_result.key_differences.map(diff => `<li>${diff}</li>`).join('')}
                            </ul>
                        </div>
                    ` : ''}
                    ${eval_result.recommendation ? `
                        <div style="margin-top: 15px; padding: 12px; background: #d4edda; border-radius: 4px; border-left: 4px solid #28a745;">
                            <strong>Recommendation:</strong>
                            <p style="margin: 8px 0 0 0; font-size: 13px; line-height: 1.6;">${eval_result.recommendation}</p>
                        </div>
                    ` : ''}
                    ${eval_result.source ? `
                        <div style="margin-top: 10px; font-size: 11px; color: #999; font-style: italic;">
                            Source: ${eval_result.source}
                        </div>
                    ` : ''}
                </div>
            `;
            resultsDiv.innerHTML = html;
        }
        
        async function runEvaluation(jobId) {
            // Check which mode is selected
            const generateMode = document.getElementById('eval_mode_generate')?.checked || false;
            const fakeMode = document.getElementById('eval_mode_fake')?.checked || false;
            const useFake = fakeMode;  // Use fake only if fake mode is selected
            
            // Get the appropriate button based on mode
            const evaluationBtn = useFake ? 
                (document.getElementById('evaluationBtnFake') || document.getElementById('evaluationBtn')) :
                document.getElementById('evaluationBtn');
            const resultsDiv = document.getElementById('evaluationResults');
            
            if (!evaluationBtn || !resultsDiv) {
                console.error('Evaluation elements not found');
                return;
            }
            
            // Debug: log mode state
            console.log('🔍 Evaluation mode state:', {
                generateMode: generateMode,
                fakeMode: fakeMode,
                useFake: useFake
            });
            
            // Disable button and show loading
            evaluationBtn.disabled = true;
            evaluationBtn.textContent = '⏳ Evaluating...';
            resultsDiv.style.display = 'block';
            resultsDiv.innerHTML = '<div style="padding: 15px; background: #fff; border-radius: 4px;">⏳ Running evaluation...</div>';
            
            try {
                // Don't hardcode integration types - let backend auto-discover from saved integration files
                const requestBody = {
                    job_id: jobId,
                    // integration1_type and integration2_type are optional - backend will auto-discover
                    use_fake: useFake  // Explicitly set based on radio button selection
                };
                
                console.log('📤 Sending evaluation request:', { use_fake: useFake, job_id: jobId });
                
                // If fake file is selected, read it
                if (useFake && fakeResponseFile) {
                    const fileReader = new FileReader();
                    fileReader.onload = async function(e) {
                        try {
                            const fakeContent = e.target.result;
                            const fakeData = JSON.parse(fakeContent);
                            
                            requestBody.fake_response_content = fakeData;
                            
                            await sendEvaluationRequest(requestBody, resultsDiv, evaluationBtn);
                        } catch (error) {
                            resultsDiv.innerHTML = `
                                <div style="padding: 15px; background: #fff; border-radius: 4px; border: 1px solid #dc3545; color: #dc3545;">
                                    <strong>❌ Error:</strong> Failed to parse fake response file: ${error.message}
                                </div>
                            `;
                            evaluationBtn.disabled = false;
                            evaluationBtn.textContent = '📊 Generate Evaluation';
                        }
                    };
                    fileReader.readAsText(fakeResponseFile);
                    return;
                }
                
                await sendEvaluationRequest(requestBody, resultsDiv, evaluationBtn);
            } catch (error) {
                resultsDiv.innerHTML = `
                    <div style="padding: 15px; background: #fff; border-radius: 4px; border: 1px solid #dc3545; color: #dc3545;">
                        <strong>❌ Error:</strong> ${error.message}
                    </div>
                `;
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
                    evaluationBtn.disabled = false;
                    evaluationBtn.textContent = '📊 Generate Evaluation';
                    return;
                }
                
                const data = await response.json();
                
                if (data.status === 'success') {
                    const eval_result = data.evaluation;
                    const table1Data = data.table1 || null;
                    const table2Data = data.table2 || null;
                    displayEvaluationResults(eval_result, resultsDiv, table1Data, table2Data);
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
                evaluationBtn.disabled = false;
                evaluationBtn.textContent = '📊 Generate Evaluation';
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
            if (!qaBtn || !resultsDivTable || !resultsDivModel) { console.error('QA elements not found'); return; }
            if (afterClickDiv) afterClickDiv.style.display = 'block';
            const useFake = document.getElementById('qa_mode_fake')?.checked || false;
            qaBtn.disabled = true;
            qaBtn.textContent = '⏳ Generating...';
            resultsDivTable.innerHTML = '<div style="padding: 12px;">⏳ Running QA...</div>';
            resultsDivModel.innerHTML = '<div style="padding: 12px;">⏳ Running QA...</div>';
            const restoreBtn = function() { qaBtn.disabled = false; qaBtn.textContent = '💬 Generate Answer'; };
            try {
                let fakeContent = null;
                if (useFake && qaFakeResponseFile) {
                    fakeContent = await new Promise((resolve, reject) => {
                        const fr = new FileReader();
                        fr.onload = () => { try { resolve(JSON.parse(fr.result)); } catch (e) { reject(e); } };
                        fr.onerror = () => reject(new Error('Failed to read file'));
                        fr.readAsText(qaFakeResponseFile);
                    });
                }
                const bodyTable = { job_id: jobId, use_table_search: true, use_fake: useFake };
                const bodyModel = { job_id: jobId, use_table_search: false, use_fake: useFake };
                if (fakeContent) { bodyTable.fake_response_content = fakeContent; bodyModel.fake_response_content = fakeContent; }
                await Promise.all([
                    sendQARequest(bodyTable, resultsDivTable, null),
                    sendQARequest(bodyModel, resultsDivModel, null)
                ]);
            } catch (error) {
                resultsDivTable.innerHTML = resultsDivTable.innerHTML.indexOf('❌') >= 0 ? resultsDivTable.innerHTML : '<div style="padding: 12px; color: #dc3545;">❌ ' + error.message + '</div>';
                resultsDivModel.innerHTML = resultsDivModel.innerHTML.indexOf('❌') >= 0 ? resultsDivModel.innerHTML : '<div style="padding: 12px; color: #dc3545;">❌ ' + error.message + '</div>';
            }
            restoreBtn();
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
                    return;
                }
                
                const data = await response.json();
                
                if (data.status === 'success') {
                    displayQAResults(data.qa, data.query, resultsDiv);
                } else {
                    resultsDiv.innerHTML = `
                        <div style="padding: 15px; background: #fff; border-radius: 4px; border: 1px solid #dc3545; color: #dc3545;">
                            <strong>❌ QA Failed:</strong> ${data.message || 'Unknown error'}
                        </div>
                    `;
                }
                if (qaBtn) { qaBtn.disabled = false; qaBtn.textContent = '💬 Generate Answer' + (qaBtn.id === 'qaBtn' ? ' (both)' : ''); }
            } catch (error) {
                resultsDiv.innerHTML = `
                    <div style="padding: 15px; background: #fff; border-radius: 4px; border: 1px solid #dc3545; color: #dc3545;">
                        <strong>❌ Error:</strong> ${error.message}
                    </div>
                `;
                if (qaBtn) { qaBtn.disabled = false; qaBtn.textContent = '💬 Generate Answer' + (qaBtn.id === 'qaBtn' ? ' (both)' : ''); }
            }
        }
        
        function displayQAResults(qaResult, query, resultsDiv) {
            if (!resultsDiv) return;
            
            resultsDiv.style.display = 'block';
            
            const answer = qaResult.answer || {};
            const answerText = answer.answer || 'No answer provided';
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
    </script>
</body>
</html>
""".replace('{{BACKEND_URL}}', BACKEND_URL)


@app.route('/')
def index():
    """Serve frontend HTML"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/static/fig/<path:filename>')
def serve_fig(filename):
    """Serve files from fig directory (PDF, PNG, etc.)"""
    # Try multiple possible paths for fig directory
    # Worktree path: /Users/doradong/.cursor/worktrees/ModelSearchDemo/gl4Cp
    # Main repo path: /Users/doradong/Repo/ModelSearchDemo
    possible_paths = [
        os.path.join(PROJECT_ROOT, 'fig'),  # Current worktree
        os.path.join(os.path.dirname(PROJECT_ROOT), 'fig'),  # Parent of worktree
        '/Users/doradong/Repo/ModelSearchDemo/fig',  # Main repo (absolute path)
        os.path.join(os.path.expanduser('~'), 'Repo', 'ModelSearchDemo', 'fig'),  # Main repo (home-relative)
    ]
    
    # Also try to find by going up from worktree to find main repo
    current = PROJECT_ROOT
    for _ in range(5):  # Go up max 5 levels
        parent = os.path.dirname(current)
        possible_paths.append(os.path.join(parent, 'ModelSearchDemo', 'fig'))
        possible_paths.append(os.path.join(parent, 'fig'))
        if 'worktrees' in current:
            # If we're in a worktree, try to find the main repo
            main_repo = os.path.join(os.path.dirname(os.path.dirname(current)), 'Repo', 'ModelSearchDemo', 'fig')
            possible_paths.append(main_repo)
        current = parent
    
    for fig_dir in possible_paths:
        if fig_dir and os.path.exists(fig_dir):
            file_path = os.path.join(fig_dir, filename)
            if os.path.exists(file_path):
                print(f"✅ Serving file from: {file_path}")
                return send_from_directory(fig_dir, filename)
    
    # If not found, return 404 with debug info
    print(f"❌ File not found. Tried paths: {possible_paths[:5]}")
    return jsonify({"error": "File not found", "filename": filename}), 404


if __name__ == '__main__':
    print("Starting ModelSearch Frontend...")
    print("Open http://localhost:5001 in your browser")
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
