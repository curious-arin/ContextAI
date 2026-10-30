# ContextAI 📑

## 🙂 Overview

ContextAI is an AI-powered **PDF-based RAG chatbot** that allows users to upload PDF documents and ask questions about their content using natural language. Built with Streamlit, LangChain, Groq, HuggingFace Embeddings, and ChromaDB, the application uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from uploaded PDFs and generate context-aware answers.

Users can provide their own Groq API key, enter their preferred Groq model name, upload PDF documents, and interact with them through an intuitive chatbot interface.

*The goal is to transform static PDF documents into interactive knowledge sources, making document understanding and information retrieval faster, easier, and more accessible.*

---

## 💡 Problem Statement & Business Impact

- Large PDF documents such as research papers, business reports, contracts, and technical manuals contain valuable information that can be difficult and time-consuming to locate manually.
- Finding specific information often requires searching through multiple pages and reading lengthy sections.
- Traditional document-reading methods make it difficult to extract relevant insights quickly.
- **Goal:** Build an AI-powered PDF-based chatbot using RAG that processes PDF documents, generates vector embeddings, retrieves relevant content through semantic search, and answers user questions based on the uploaded PDFs.
- **Potential value:** Saves time, simplifies document understanding, improves information accessibility, and allows users to interact with lengthy PDF documents through natural language conversations.

---

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain   
- Groq API
- HuggingFace Embeddings
- ChromaDB
- PyPDF

---

## 🚀 Installation

**1. Clone the repository**

```bash
git clone https://github.com/curious-arin/ContextAI.git
```

**2. Navigate to the project directory**

```bash
cd ContextAI
```

**3. Create a virtual environment (optional but recommended)**

```bash
python -m venv .venv
```

**4. Activate the virtual environment**

Windows:

```bash
.venv\Scripts\activate
```

**5. Install dependencies**

```bash
pip install -r requirements.txt
```

**6. Run the application**

```bash
streamlit run app.py
```

---

## ▶️ Usage

1. Launch the ContextAI application.
2. Enter your Groq API key.
3. Enter the Groq model ID you want to use.
4. Authenticate with Groq.
5. Upload your PDF document.
6. Process the PDF to create its vector database.
7. Start asking questions about your document.
8. Receive AI-generated answers based on the PDF content.

---

## 📁 Project Structure

```text
ContextAI/
│
├── raw/                 # Complete workflow in command line
│   ├── database.py     
│   └── main.py          
│
├── .gitignore
├── app.py             
├── LICENSE   
├── README.md          
└── requirements.txt                
```

---

## 🎯 Use Cases

- Academic research papers
- Business reports and documents
- Legal documents and contracts
- Technical manuals and documentation
- Educational and study materials
- Books and other PDF resources

---

## 🔮 Future Improvements

- Build a modern, scalable frontend using React and backend architecture using FastAPI.
- Integrate Groq, OpenAI, Google Gemini, and Mistral AI as multiple AI provider support.
- Add Conversational Memory & Chat Management to remember previous interactions and save, access, and manage past conversations.
- Add Context-Based Image Generation functionality to generate images using concepts and information extracted from uploaded PDFs.
- Add Advanced RAG Capabilities supporting multiple PDFs & improve retrieval accuracy.