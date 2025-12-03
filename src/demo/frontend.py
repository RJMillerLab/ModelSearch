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
            margin: 6px 0;
            padding: 6px;
            background: #f8f9fa;
            border-radius: 4px;
        }
        .search-type-header {
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 5px 8px;
            background: white;
            border-radius: 4px;
            margin-bottom: 6px;
        }
        .search-type-header:hover {
            background: #e9ecef;
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
            </div>
            
            <div id="new-search-inputs">
            <div id="diagram-section" class="pdf-section" style="margin-bottom: 20px;">
                <img id="search-diagram" src="/static/fig/modelsearch.png" alt="ModelSearch Overview" />
            </div>
            
            <div class="mode-selector" style="margin-top: 15px;">
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
            
            <div style="margin-top: 15px; padding: 12px; background: #f8f9fa; border-radius: 4px; border: 1px solid #ddd;">
                <label style="margin-bottom: 10px; display: block; font-weight: bold;">Tab2Tab Step Mode (Card2Tab2Card Intermediate Step):</label>
                <div class="mode-option" style="margin-bottom: 10px;">
                    <input type="radio" id="tab2tab_mode_search" name="tab2tab_mode" value="search" checked onchange="toggleTab2TabMode()">
                    <label for="tab2tab_mode_search">Temporary Dataset Search (default)</label>
                </div>
                <div class="mode-option" style="margin-bottom: 10px;">
                    <input type="radio" id="tab2tab_mode_load" name="tab2tab_mode" value="load" onchange="toggleTab2TabMode()">
                    <label for="tab2tab_mode_load">Load from Saved JSON Results</label>
                </div>
                <div id="tab2tab_json_input" style="display: none; margin-top: 10px;">
                    <label for="tab2tab_json_file" style="font-size: 12px; color: #666;">Select saved tab2tab JSON file:</label>
                    <input type="file" id="tab2tab_json_file" accept=".json" style="margin-top: 5px; padding: 5px; width: 100%; font-size: 12px;">
                    <p style="font-size: 10px; color: #999; margin-top: 3px;">
                        JSON file should contain tab2tab search results (from tab2tab.py --output)
                    </p>
                </div>
            </div>
            
            <label for="table_search_k" style="margin-top: 15px;">Table Search Top K (only used when mode is "Temporary Dataset Search"):</label>
            <div style="display: flex; align-items: center; gap: 10px;">
                <input type="range" id="table_search_k_slider" min="1" max="20" value="20" step="1" 
                       style="flex: 1;" oninput="updateTableSearchKValue(this.value)">
                <input type="number" id="table_search_k" value="20" min="1" max="20" 
                       style="width: 80px;" oninput="updateTableSearchKSlider(this.value)">
            </div>
            <p style="font-size: 11px; color: #666; margin-top: 3px;">
                Controls how many tables to retrieve in Card2Tab2Card search (intermediate step). 
                Final modelcard count is still controlled by "Top K Results" above.
                <span id="table_search_k_note" style="display: none; color: #999;"> (Disabled when loading from JSON)</span>
            </p>
            
            <button id="searchBtn" onclick="startSearch()">Start Search</button>
            </div>
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
            
            // Initialize tab2tab mode
            toggleTab2TabMode();
        });
        
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
            // When top_k changes, suggest a default table_search_k (1.5x top_k, but at least 20, max 20)
            const topK = parseInt(document.getElementById('top_k').value) || 20;
            const suggestedTableSearchK = Math.min(Math.max(Math.round(topK * 1.5), 20), 20);
            // Only update if current value is close to old default (within 5)
            const currentTableSearchK = parseInt(document.getElementById('table_search_k').value) || 20;
            const oldTopK = Math.floor(currentTableSearchK / 1.5);
            if (Math.abs(currentTableSearchK - Math.round(oldTopK * 1.5)) <= 5) {
                // User hasn't manually adjusted much, update to new default
                const newValue = Math.min(Math.max(suggestedTableSearchK, 1), 20);
                document.getElementById('table_search_k').value = newValue;
                document.getElementById('table_search_k_slider').value = newValue;
            }
        }
        
        function toggleTab2TabMode() {
            const mode = document.querySelector('input[name="tab2tab_mode"]:checked').value;
            const jsonInput = document.getElementById('tab2tab_json_input');
            const tableSearchKInputs = document.getElementById('table_search_k');
            const tableSearchKSlider = document.getElementById('table_search_k_slider');
            const note = document.getElementById('table_search_k_note');
            
            if (mode === 'load') {
                jsonInput.style.display = 'block';
                tableSearchKInputs.disabled = true;
                tableSearchKSlider.disabled = true;
                note.style.display = 'inline';
            } else {
                jsonInput.style.display = 'none';
                tableSearchKInputs.disabled = false;
                tableSearchKSlider.disabled = false;
                note.style.display = 'none';
            }
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
                    
                    if (data.searches.length === 0 && !data.template_available) {
                        html = '<div style="text-align: center; color: #666; padding: 20px;">No saved searches found. Run a new search to create one.</div>';
                    } else {
                        // Display each saved search as a clickable card
                        data.searches.forEach(search => {
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
            
            const mode = document.querySelector('input[name="search_mode"]:checked').value;
            const query = document.getElementById('query').value.trim();
            const modelId = document.getElementById('model_id').value.trim();
            const topK = parseInt(document.getElementById('top_k').value);
            const tableSearchK = parseInt(document.getElementById('table_search_k').value);
            const tab2tabMode = document.querySelector('input[name="tab2tab_mode"]:checked').value;
            
            // Validate input based on mode
            if (mode === 'query' && !query) {
                showError('Please enter a query');
                return;
            }
            if (mode === 'modelid' && !modelId) {
                showError('Please enter a model ID');
                return;
            }
            
            // Validate tab2tab mode
            if (tab2tabMode === 'load') {
                const jsonFile = document.getElementById('tab2tab_json_file').files[0];
                if (!jsonFile) {
                    showError('Please select a JSON file when using "Load from Saved JSON Results" mode');
                    return;
                }
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
                    search_mode: 'new',  // Explicitly set to new search
                    mode: mode,
                    top_k: topK,
                    tab2tab_mode: tab2tabMode
                };
                
                // Add table_search_k only if mode is search
                if (tab2tabMode === 'search') {
                    requestBody.table_search_k = tableSearchK;
                }
                
                // Add tab2tab_json if mode is load
                if (tab2tabMode === 'load') {
                    const jsonFile = document.getElementById('tab2tab_json_file').files[0];
                    // Read file as text
                    const fileReader = new FileReader();
                    fileReader.onload = async function(e) {
                        try {
                            const jsonContent = e.target.result;
                            // Parse to validate JSON
                            const jsonData = JSON.parse(jsonContent);
                            requestBody.tab2tab_json = jsonContent;
                            
                            // Continue with the request
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
                            showError('Error reading JSON file: ' + error.message);
                            document.getElementById('searchBtn').disabled = false;
                        }
                    };
                    fileReader.onerror = function() {
                        showError('Error reading JSON file');
                        document.getElementById('searchBtn').disabled = false;
                    };
                    fileReader.readAsText(jsonFile);
                    return; // Exit early, will continue in fileReader.onload
                }
                
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
            const diagramImg = document.getElementById('search-diagram');
            
            if (mode === 'query') {
                queryInput.classList.add('active');
                modelIdInput.classList.remove('active');
                // Show query diagram
                if (diagramImg) {
                    diagramImg.src = '/static/fig/modelsearch_wquery.png';
                }
            } else {
                queryInput.classList.remove('active');
                modelIdInput.classList.add('active');
                // Show modelId diagram
                if (diagramImg) {
                    diagramImg.src = '/static/fig/modelsearch.png';
                }
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
                    <div class="result-card" style="min-width: 0;">
                        <h3 style="margin-top: 0; margin-bottom: 10px; font-size: 16px;"><span class="number-badge">1</span> Card2Card Results (${results.card2card_results.length})</h3>
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
                    <div class="result-card" style="min-width: 0;">
                        <h3 style="margin-top: 0; margin-bottom: 10px; font-size: 16px;"><span class="number-badge">2</span> Card2Tab2Card Results</h3>
                        ${Object.entries(results.card2tab2card_results).map(([type, data]) => {
                            const sectionId = card2tab2cardIds[type];
                            // Handle both old format (array) and new format (object with model_ids and intermediate)
                            const models = Array.isArray(data) ? data : (data.model_ids || []);
                            const intermediate = data.intermediate || {};
                            const tableToModels = intermediate.table_to_models || {};
                            
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
                                        <h4>${type} (${models.length} models)</h4>
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
            
            // Add Integration Section
            html += `
                <div class="integration-section" style="margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 8px;">
                    <h3>Table Integration</h3>
                    <p style="font-size: 14px; color: #666; margin-bottom: 15px;">
                        Integrate tables from <span class="number-badge">2</span> Card2Tab2Card search results using Union or Intersection.
                    </p>
                    <div style="display: flex; gap: 15px; align-items: center; flex-wrap: wrap;">
                        <label>
                            Search Type:
                            <select id="integration_search_type" style="margin-left: 5px; padding: 5px;">
                                <option value="single_column">Single Column</option>
                                <option value="keyword">Keyword</option>
                                <option value="multi_column">Multi Column</option>
                                <option value="unionable">Unionable</option>
                                <option value="complex">Complex (Union+Join+Correlation)</option>
                                <option value="correlation">Correlation</option>
                                <option value="imputation">Imputation</option>
                                <option value="augmentation">Augmentation</option>
                                <option value="dependent_data">Dependent Data</option>
                                <option value="feature_for_ml">Feature for ML</option>
                                <option value="multi_column_collinearity">Multi-Column Collinearity</option>
                                <option value="negative_example">Negative Example</option>
                            </select>
                        </label>
                        <label>
                            Integration Type:
                            <select id="integration_type" style="margin-left: 5px; padding: 5px;">
                                <option value="union">Union</option>
                                <option value="intersection">Intersection</option>
                                <option value="alite">ALITE (FD-based)</option>
                                <option value="outer_join">Outer Join</option>
                            </select>
                        </label>
                        <label>
                            Top K Tables:
                            <input type="number" id="integration_k" value="10" min="1" max="50" style="margin-left: 5px; padding: 5px; width: 60px;">
                        </label>
                        <button id="integrationBtn" onclick="runIntegration('${results.job_id || currentJobId}')" 
                                style="padding: 8px 16px; background: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: 500;">
                            🔗 Integrate Tables
                        </button>
                    </div>
                    <div id="integrationResults" style="margin-top: 20px; display: none;"></div>
                </div>
            `;
            
            container.innerHTML = html;
            document.getElementById('resultsSection').classList.add('active');
        }
        
        async function runIntegration(jobId) {
            const searchType = document.getElementById('integration_search_type').value;
            const integrationType = document.getElementById('integration_type').value;
            const k = parseInt(document.getElementById('integration_k').value);
            
            const integrationBtn = document.getElementById('integrationBtn');
            const resultsDiv = document.getElementById('integrationResults');
            
            // Disable button and show loading
            integrationBtn.disabled = true;
            integrationBtn.textContent = '⏳ Integrating...';
            resultsDiv.style.display = 'block';
            resultsDiv.innerHTML = '<div style="padding: 15px; background: #fff; border-radius: 4px;">⏳ Running integration...</div>';
            
            try {
                const response = await fetch('{{BACKEND_URL}}/api/integrate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        job_id: jobId,
                        search_type: searchType,
                        integration_type: integrationType,
                        k: k
                    })
                });
                
                const data = await response.json();
                
                if (data.status === 'success') {
                    // Display integration results
                    const stats = data.stats;
                    const table = data.integrated_table;
                    
                    let html = `
                        <div style="padding: 15px; background: #fff; border-radius: 4px; border: 1px solid #dee2e6;">
                            <h4 style="margin-top: 0; color: #28a745;">✅ Integration Successful</h4>
                            <div style="margin-bottom: 15px;">
                                <strong>Statistics:</strong><br>
                                Input: ${stats.input_tables} tables, ${stats.input_rows} total rows<br>
                                Output: ${stats.output_rows} rows, ${stats.output_columns} columns<br>
                                Type: ${stats.integration_type}
                            </div>
                    `;
                    
                    if (stats.output_rows === 0) {
                        // Empty result (e.g., intersection with no common rows/columns)
                        html += `
                            <div style="padding: 20px; text-align: center; color: #666; background: #f8f9fa; border-radius: 4px;">
                                <p style="margin: 0; font-size: 14px;">
                                    ${stats.output_columns === 0 
                                        ? '⚠️ No common columns found between tables. Intersection result is empty.' 
                                        : '⚠️ No common rows found between tables. Intersection result is empty.'}
                                </p>
                            </div>
                        `;
                    } else {
                        // Show table
                        html += `
                            <div style="max-height: 400px; overflow: auto;">
                                <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
                                    <thead>
                                        <tr style="background: #f8f9fa; position: sticky; top: 0;">
                                            ${table.columns.map(col => `<th style="border: 1px solid #dee2e6; padding: 6px; text-align: left;">${col}</th>`).join('')}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        ${table.data.slice(0, 100).map(row => `
                                            <tr>
                                                ${row.map(cell => `<td style="border: 1px solid #dee2e6; padding: 6px;">${cell || ''}</td>`).join('')}
                                            </tr>
                                        `).join('')}
                                    </tbody>
                                </table>
                                ${table.data.length > 100 ? `<p style="font-size: 11px; color: #666; margin-top: 10px;">Showing first 100 of ${table.data.length} rows</p>` : ''}
                            </div>
                        `;
                    }
                    
                    html += `</div>`;
                    resultsDiv.innerHTML = html;
                } else {
                    resultsDiv.innerHTML = `
                        <div style="padding: 15px; background: #fff; border-radius: 4px; border: 1px solid #dc3545; color: #dc3545;">
                            <strong>❌ Integration Failed:</strong> ${data.message || 'Unknown error'}
                        </div>
                    `;
                }
            } catch (error) {
                resultsDiv.innerHTML = `
                    <div style="padding: 15px; background: #fff; border-radius: 4px; border: 1px solid #dc3545; color: #dc3545;">
                        <strong>❌ Error:</strong> ${error.message}
                    </div>
                `;
            } finally {
                integrationBtn.disabled = false;
                integrationBtn.textContent = '🔗 Integrate Tables';
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
                    const count = currentText.match(/Show (\d+)/)[1];
                    toggleElement.textContent = `Hide ${count} more`;
                } else if (currentText.includes('Hide')) {
                    const count = currentText.match(/Hide (\d+)/)[1];
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
