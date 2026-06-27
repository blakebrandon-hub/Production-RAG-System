let isModelReady = false;

        // Initialize model on page load
        window.addEventListener('load', async () => {
            await initializeModel();
            await loadMemories();
        });

        async function initializeModel() {
            const statusEl = document.getElementById('modelStatus');
            statusEl.textContent = '⏳';
            
            try {
                const response = await fetch('/api/initialize', {
                    method: 'POST'
                });
                
                const data = await response.json();
                
                if (data.status === 'success') {
                    isModelReady = true;
                    statusEl.textContent = '✅';
                    showStatus('storeStatus', 'Model initialized successfully', 'success');
                } else {
                    statusEl.textContent = '❌';
                    showStatus('storeStatus', 'Model initialization failed', 'error');
                }
            } catch (error) {
                statusEl.textContent = '❌';
                showStatus('storeStatus', 'Error: ' + error.message, 'error');
            }
        }

        async function storeMemory() {
            const text = document.getElementById('storeText').value;
            const metadataStr = document.getElementById('storeMetadata').value;
            
            if (!text) {
                showStatus('storeStatus', 'Please enter text to store', 'error');
                return;
            }
            
            let metadata = {};
            if (metadataStr) {
                try {
                    metadata = JSON.parse(metadataStr);
                } catch (e) {
                    showStatus('storeStatus', 'Invalid JSON metadata', 'error');
                    return;
                }
            }
            
            try {
                const response = await fetch('/api/store', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text, metadata })
                });
                
                const data = await response.json();
                
                if (data.status === 'success') {
                    showStatus('storeStatus', data.message, 'success');
                    document.getElementById('storeText').value = '';
                    document.getElementById('storeMetadata').value = '';
                    await loadMemories();
                } else {
                    showStatus('storeStatus', data.message, 'error');
                }
            } catch (error) {
                showStatus('storeStatus', 'Error: ' + error.message, 'error');
            }
        }

        async function queryStore() {
            const query = document.getElementById('queryText').value;
            
            if (!query) {
                showStatus('queryStatus', 'Please enter a search query', 'error');
                return;
            }
            
            try {
                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query })
                });
                
                const data = await response.json();
                
                if (data.status === 'success' || data.results) {
                    displayResults(data.results);
                    showStatus('queryStatus', 'Search completed', 'success');
                } else {
                    showStatus('queryStatus', data.message, 'error');
                }
            } catch (error) {
                showStatus('queryStatus', 'Error: ' + error.message, 'error');
            }
        }

        function displayResults(results) {
            const resultsEl = document.getElementById('queryResults');
            
            if (!results || results.length === 0) {
                resultsEl.innerHTML = '<p style="color: #666; margin-top: 15px;">No results found</p>';
                return;
            }
            
            resultsEl.innerHTML = results.map(r => `
                <div class="result-item">
                    <span class="score">Score: ${(r.score * 100).toFixed(1)}%</span>
                    <div class="text">${escapeHtml(r.text)}</div>
                </div>
            `).join('');
        }

        async function loadMemories() {
            try {
                const response = await fetch('/api/list');
                const data = await response.json();
                
                if (data.status === 'success') {
                    document.getElementById('memoryCount').textContent = data.count;
                    displayMemories(data.memories);
                }
            } catch (error) {
                showStatus('listStatus', 'Error loading memories: ' + error.message, 'error');
            }
        }

        function displayMemories(memories) {
            const listEl = document.getElementById('memoryList');
            
            if (!memories || memories.length === 0) {
                listEl.innerHTML = '<p style="color: #666; margin-top: 15px; text-align: center;">No memories stored</p>';
                return;
            }
            
            listEl.innerHTML = memories.map(m => {
                const date = new Date(m.timestamp * 1000).toLocaleString();
                return `
                    <div class="memory-item">
                        <div class="content">
                            <div class="text">${escapeHtml(m.text)}</div>
                            <div class="meta">${date}</div>
                        </div>
                        <button onclick="deleteMemory('${m.id}')">Delete</button>
                    </div>
                `;
            }).join('');
        }

        async function deleteMemory(id) {
            try {
                const response = await fetch(`/api/delete/${id}`, {
                    method: 'DELETE'
                });
                
                const data = await response.json();
                
                if (data.status === 'success') {
                    showStatus('listStatus', data.message, 'success');
                    await loadMemories();
                } else {
                    showStatus('listStatus', data.message, 'error');
                }
            } catch (error) {
                showStatus('listStatus', 'Error: ' + error.message, 'error');
            }
        }

        async function clearStore() {
            if (!confirm('Are you sure you want to clear all memories?')) return;
            
            try {
                const response = await fetch('/api/clear', {
                    method: 'POST'
                });
                
                const data = await response.json();
                
                if (data.status === 'success') {
                    showStatus('listStatus', data.message, 'success');
                    await loadMemories();
                } else {
                    showStatus('listStatus', data.message, 'error');
                }
            } catch (error) {
                showStatus('listStatus', 'Error: ' + error.message, 'error');
            }
        }

        async function pruneBySize() {
            const maxSize = prompt('Enter maximum number of memories to keep:', '200');
            if (!maxSize) return;
            
            try {
                const response = await fetch('/api/prune/size', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ max_memories: parseInt(maxSize) })
                });
                
                const data = await response.json();
                showStatus('listStatus', data.message, 'success');
                await loadMemories();
            } catch (error) {
                showStatus('listStatus', 'Error: ' + error.message, 'error');
            }
        }

        async function pruneByAge() {
            const hours = prompt('Remove memories older than (hours):', '24');
            if (!hours) return;
            
            try {
                const response = await fetch('/api/prune/age', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ max_age_ms: parseInt(hours) * 3600000 })
                });
                
                const data = await response.json();
                showStatus('listStatus', data.message, 'success');
                await loadMemories();
            } catch (error) {
                showStatus('listStatus', 'Error: ' + error.message, 'error');
            }
        }

        async function pruneDuplicates() {
            const threshold = prompt('Enter similarity threshold (0.0-1.0):', '0.95');
            if (!threshold) return;
            
            try {
                const response = await fetch('/api/prune/duplicates', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ threshold: parseFloat(threshold) })
                });
                
                const data = await response.json();
                showStatus('listStatus', data.message, 'success');
                await loadMemories();
            } catch (error) {
                showStatus('listStatus', 'Error: ' + error.message, 'error');
            }
        }

        function showStatus(elementId, message, type) {
            const el = document.getElementById(elementId);
            el.innerHTML = `<div class="status ${type}">${escapeHtml(message)}</div>`;
            setTimeout(() => {
                el.innerHTML = '';
            }, 5000);
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }