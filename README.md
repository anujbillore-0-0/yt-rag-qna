# YouTube RAG Q\&A App

This **Streamlit** application lets you ask questions about any YouTube video using a **Retrieval-Augmented Generation (RAG)** pipeline.
It leverages **FAISS** for vector storage, **LangChain** for orchestration, and **Groq's LLaMA 3** model for natural language understanding and generation.

With this tool, you can:

* Extract transcripts from a YouTube video.
* Split and embed the text into a FAISS vector database.
* Retrieve the most relevant segments for your query.
* Get context-aware answers directly from the video content.

---

## 🚀 Features

* **YouTube Transcript Extraction** – Automatically fetches captions/transcripts from videos.
* **RAG Pipeline** – Combines retrieval and generation for more accurate answers.
* **FAISS Vector Database** – Stores and indexes embeddings for fast similarity search.
* **LangChain Integration** – Manages prompt creation, chaining, and query handling.
* **Groq LLaMA 3 Model** – Provides state-of-the-art language generation.
* **Streamlit UI** – Interactive, easy-to-use web interface.

---

## 🛠️ Tech Stack

* **[Python](https://www.python.org/)**
* **[Streamlit](https://streamlit.io/)**
* **[LangChain](https://www.langchain.com/)**
* **[FAISS](https://faiss.ai/)**
* **[Groq API](https://groq.com/)**
* **[LLaMA 3](https://ai.meta.com/llama/)**

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/anujbillore-0-0/yt-rag-qna
cd yt-rag-qna
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables

Create a `.env` file in the project root and add:

```env
GROQ_API_KEY=your_groq_api_key_here
```

You can get your API key from [Groq's developer portal](https://console.groq.com/).

---

## ▶️ Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

### Steps in the App:

1. Enter the **YouTube video URL**.
2. Click **Process Video** – the app will:

   * Fetch the transcript.
   * Split the transcript into chunks.
   * Embed them using a language model.
   * Store them in a FAISS vector store.
3. Type your **question** in the input box.
4. Get an **AI-generated answer** based on the retrieved video segments.

---


## ⚙️ How It Works

1. **Transcript Loading** – Extracts the video's transcript.
2. **Text Chunking** – Splits into manageable chunks for embedding.
3. **Embedding Creation** – Converts text chunks into embeddings using LangChain.
4. **FAISS Storage** – Stores embeddings for fast similarity search.
5. **Query Handling** – Retrieves the most relevant chunks for a user’s query.
6. **Answer Generation** – Sends retrieved context to Groq's LLaMA 3 for final response.

---
