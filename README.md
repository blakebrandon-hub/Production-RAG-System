This project is a **RAG Vector Store Demo**, a lightweight application designed to demonstrate semantic search and memory management for Retrieval-Augmented Generation (RAG) systems. It uses sentence embeddings to transform text into high-dimensional vectors, allowing for search based on meaning rather than just keywords.

---

## 🚀 Features

*   **Semantic Memory Storage**: Encode and store text snippets as vectors using the `all-MiniLM-L6-v2` model.
*   **Vector Search**: Query the store using natural language to find the most relevant entries based on **cosine similarity**.
*   **Advanced Pruning**:
    *   **Age-based**: Remove entries older than a specified duration.
    *   **Size-based**: Limit the store to a maximum number of recent entries.
    *   **Duplicate Removal**: Identify and remove semantically similar entries based on a similarity threshold (e.g., $> 0.95$).
*   **Interactive Dashboard**: A modern web interface to manage memories, monitor model status, and view search scores in real-time.

---

## 🛠️ Tech Stack

*   **Backend**: Python with **Flask** and **Flask-CORS**.
*   **Embeddings**: **Sentence-Transformers** (SBERT).
*   **Mathematics**: **NumPy** for vector operations and similarity calculations.
*   **Frontend**: HTML5, CSS3 (Modern Flex/Grid), and Vanilla JavaScript.

---

## 📐 Technical Details

The core of the search functionality relies on calculating the **cosine similarity** between a query vector ($q$) and a stored memory vector ($m$):

$$score = \frac{q \cdot m}{\|q\| \|m\|}$$

The system filters results to return only those with a similarity score greater than **0.40**, limited to the top 3 most relevant matches.[cite: 3]

---

## 📂 API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/initialize` | `POST` | Loads the `SentenceTransformer` model into memory.[cite: 3] |
| `/api/store` | `POST` | Encodes text and metadata into the vector store.[cite: 3] |
| `/api/query` | `POST` | Performs a semantic search against stored vectors.[cite: 3] |
| `/api/list` | `GET` | Retrieves all stored memories and their timestamps.[cite: 3] |
| `/api/delete/<id>`| `DELETE` | Removes a specific memory by its unique UUID.[cite: 3] |
| `/api/prune/duplicates` | `POST` | Removes entries that are semantically redundant.[cite: 3] |

---

## ⚙️ Installation & Usage

### 1. Install Dependencies
Ensure you have Python installed, then run:
```bash
pip install flask flask-cors sentence-transformers numpy
```

### 2. Run the Server
```bash
python app.py
```
The server will start at `[http://0.0.0.0:5000](http://0.0.0.0:5000)`.[cite: 3]

### 3. Access the Dashboard
Open your browser to `http://localhost:5000` to interact with the UI.[cite: 4] **Note**: You must click "Initialize Model" (or refresh the page) to load the embedding model before storing or querying data.[cite: 3, 4]

---

> **Note**: This demo uses an in-memory list (`memory_store`) to keep data.[cite: 3] If you restart the Flask server, all stored memories will be cleared.
