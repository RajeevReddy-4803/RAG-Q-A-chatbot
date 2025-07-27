## 🧠 RAG Q\&A Chatbot with FAISS & Open-Source LLMs

This project is a **Retrieval-Augmented Generation (RAG)** chatbot that uses **FAISS** for document retrieval and **open-source transformer models** (e.g., HuggingFace LLMs) for answer generation. It allows users to query their custom documents and receive accurate, context-aware answers.

---

### 🚀 Features

* 🔍 **Semantic Search** using FAISS vector index
* 🤖 **Open-Source LLMs** for answer generation (no OpenAI required)
* 📁 **Dynamic document uploading** and preprocessing
* 🧹 **Auto fills missing/null values** in uploaded CSVs
* ⚡ **Fast FAISS index loading** with `.pkl` caching
* 🎛️ **User-selectable Top-K document retrieval**
* 🌐 **Streamlit Web UI** for interaction

---

### 🏗️ Tech Stack

* `Python 3.10+`
* `HuggingFace Transformers`
* `Sentence-Transformers`
* `FAISS`
* `Scikit-learn`
* `Streamlit`
* `Pandas`, `NumPy`
* `dotenv` for API key & environment config

---

### 📦 Installation

```bash
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot

# Setup virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

### 🔐 Environment Setup

Create a `.env` file and add:

```
HUGGINGFACE_API_KEY=your_huggingface_key
MODEL_NAME=your_preferred_model_id
```

---

### 📁 How to Use

1. **Run the app**:

   ```bash
   streamlit run app.py
   ```

2. **Upload** a `.csv` or `.txt` file.

3. **Ask questions** based on the uploaded document.

4. **Select Top-K documents** to refine context and answer relevance.

---

### 💾 Index Optimization

To speed up performance, the FAISS index is **saved as `.pkl`** after first creation and **loaded instantly** on next runs.

### 🧪 Example Query

> **Input**: "What is the capital of France?"
> **Output**: "The capital of France is Paris."

