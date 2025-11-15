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
            <label for="query">Query Text:</label>
            <input type="text" id="query" placeholder="e.g., transformer model for code generation" value="transformer model for code generation">
            
            <label for="top_k" style="margin-top: 15px;">Top K Results:</label>
            <input type="number" id="top_k" value="20" min="1" max="100">
            
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
        
        async function startSearch() {
            const query = document.getElementById('query').value.trim();
            const topK = parseInt(document.getElementById('top_k').value);
            
            if (!query) {
                showError('Please enter a query');
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
                const response = await fetch('{{BACKEND_URL}}/api/search', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({query: query, top_k: topK})
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
                
                addLog(data.message || data.timestamp + ': ' + data.message);
            };
            
            eventSource.onerror = function(error) {
                console.error('SSE error:', error);
                eventSource.close();
            };
        }
        
        function addLog(message) {
            const container = document.getElementById('logContainer');
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.textContent = message;
            container.appendChild(entry);
            container.scrollTop = container.scrollHeight;
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
            
            let html = `
                <div class="results-grid">
                    <div class="result-card">
                        <h3>Card2Card Results (${results.card2card_results.length})</h3>
                        <ul class="result-list">
                            ${results.card2card_results.slice(0, 10).map(m => `<li class="result-item">${m}</li>`).join('')}
                            ${results.card2card_results.length > 10 ? `<li>... and ${results.card2card_results.length - 10} more</li>` : ''}
                        </ul>
                    </div>
                    <div class="result-card">
                        <h3>Card2Tab2Card Results</h3>
                        ${Object.entries(results.card2tab2card_results).map(([type, models]) => `
                            <h4>${type} (${Array.isArray(models) ? models.length : 0})</h4>
                            <ul class="result-list">
                                ${Array.isArray(models) ? models.slice(0, 5).map(m => `<li class="result-item">${m}</li>`).join('') : '<li>Error</li>'}
                            </ul>
                        `).join('')}
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
