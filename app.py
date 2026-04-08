from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime
import uuid
import json

app = Flask(__name__)
CORS(app)

# Global state
embedding_model = None
memory_store = []
is_model_ready = False

def initialize_model():
    global embedding_model, is_model_ready
    if embedding_model is None:
        print("Loading embedding model...")
        embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        is_model_ready = True
        print("Model loaded successfully")

def cosine_similarity(vec_a, vec_b):
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot / (norm_a * norm_b)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/initialize', methods=['POST'])
def api_initialize():
    try:
        initialize_model()
        return jsonify({
            'status': 'success',
            'message': 'Model initialized',
            'is_ready': is_model_ready
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/store', methods=['POST'])
def api_store():
    global memory_store
    
    if not is_model_ready:
        return jsonify({'status': 'error', 'message': 'Model not initialized'}), 400
    
    data = request.json
    text = data.get('text', '')
    metadata = data.get('metadata', {})
    
    if not text:
        return jsonify({'status': 'error', 'message': 'Text is required'}), 400
    
    try:
        # Generate embedding
        vector = embedding_model.encode(text, normalize_embeddings=True)
        
        memory = {
            'id': str(uuid.uuid4()),
            'text': text,
            'vector': vector.tolist(),
            'metadata': metadata,
            'timestamp': datetime.now().timestamp()
        }
        
        memory_store.append(memory)
        
        return jsonify({
            'status': 'success',
            'id': memory['id'],
            'message': f'Stored: {text[:40]}...'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/query', methods=['POST'])
def api_query():
    if not is_model_ready:
        return jsonify({
            'status': 'error',
            'results': [{
                'text': 'Embedding model is not initialized.',
                'metadata': {'type': 'system_message'},
                'score': 0
            }]
        })
    
    if len(memory_store) == 0:
        return jsonify({
            'status': 'success',
            'results': [{
                'text': 'Memory store is empty.',
                'metadata': {'type': 'system_message'},
                'score': 0
            }]
        })
    
    data = request.json
    query_text = data.get('query', '')
    
    if not query_text:
        return jsonify({'status': 'error', 'message': 'Query is required'}), 400
    
    try:
        # Generate query embedding
        query_vector = embedding_model.encode(query_text, normalize_embeddings=True)
        
        # Calculate similarities
        results = []
        for mem in memory_store:
            mem_vector = np.array(mem['vector'])
            score = float(cosine_similarity(query_vector, mem_vector))
            results.append({
                'id': mem['id'],
                'text': mem['text'],
                'metadata': mem['metadata'],
                'score': score,
                'timestamp': mem['timestamp']
            })
        
        # Sort by score descending
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Filter and limit
        relevant_results = [r for r in results if r['score'] > 0.40][:3]
        
        if len(relevant_results) == 0:
            return jsonify({
                'status': 'success',
                'results': [{
                    'text': 'No relevant results found.',
                    'metadata': {'type': 'system_message'},
                    'score': 0
                }]
            })
        
        return jsonify({
            'status': 'success',
            'results': relevant_results
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'results': [{
                'text': 'An error occurred during query.',
                'metadata': {'type': 'system_message'},
                'score': 0
            }]
        }), 500

@app.route('/api/delete/<memory_id>', methods=['DELETE'])
def api_delete(memory_id):
    global memory_store
    
    for i, mem in enumerate(memory_store):
        if mem['id'] == memory_id:
            removed = memory_store.pop(i)
            return jsonify({
                'status': 'success',
                'message': f'Removed: {removed["text"][:40]}...'
            })
    
    return jsonify({'status': 'error', 'message': 'Memory not found'}), 404

@app.route('/api/clear', methods=['POST'])
def api_clear():
    global memory_store
    memory_store = []
    return jsonify({'status': 'success', 'message': 'Memory store cleared'})

@app.route('/api/list', methods=['GET'])
def api_list():
    return jsonify({
        'status': 'success',
        'count': len(memory_store),
        'memories': [{
            'id': mem['id'],
            'text': mem['text'],
            'metadata': mem['metadata'],
            'timestamp': mem['timestamp']
        } for mem in memory_store]
    })

@app.route('/api/prune/age', methods=['POST'])
def api_prune_age():
    global memory_store
    
    data = request.json
    max_age_ms = data.get('max_age_ms', 3600000)  # 1 hour default
    
    now = datetime.now().timestamp()
    initial_count = len(memory_store)
    
    memory_store = [mem for mem in memory_store 
                   if (now - mem['timestamp']) * 1000 < max_age_ms]
    
    removed_count = initial_count - len(memory_store)
    
    return jsonify({
        'status': 'success',
        'message': f'Removed {removed_count} old entries',
        'remaining': len(memory_store)
    })

@app.route('/api/prune/size', methods=['POST'])
def api_prune_size():
    global memory_store
    
    data = request.json
    max_memories = data.get('max_memories', 200)
    
    if len(memory_store) <= max_memories:
        return jsonify({
            'status': 'success',
            'message': 'No pruning needed',
            'count': len(memory_store)
        })
    
    # Sort by timestamp, keep most recent
    memory_store.sort(key=lambda x: x['timestamp'], reverse=True)
    removed_count = len(memory_store) - max_memories
    memory_store = memory_store[:max_memories]
    
    return jsonify({
        'status': 'success',
        'message': f'Removed {removed_count} entries',
        'remaining': len(memory_store)
    })

@app.route('/api/prune/duplicates', methods=['POST'])
def api_prune_duplicates():
    global memory_store
    
    data = request.json
    threshold = data.get('threshold', 0.95)
    
    pruned = []
    for mem in memory_store:
        is_duplicate = False
        mem_vector = np.array(mem['vector'])
        
        for existing in pruned:
            existing_vector = np.array(existing['vector'])
            if cosine_similarity(mem_vector, existing_vector) > threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            pruned.append(mem)
    
    removed_count = len(memory_store) - len(pruned)
    memory_store = pruned
    
    return jsonify({
        'status': 'success',
        'message': f'Removed {removed_count} duplicates',
        'remaining': len(memory_store)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
