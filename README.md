
# AI-Powered Document Search and Question Answering using RAG

## 📌 Project Overview

This project is an AI-powered document question-answering system based on **Retrieval-Augmented Generation (RAG)**.

It allows users to ask questions about a document and receive answers based on the information available in that document.

The system retrieves the most relevant sections from the document and provides them to an AI model to generate a grounded answer.

## 🚀 Features

- PDF document processing
- Text extraction from PDF
- Text chunking
- Semantic embeddings
- FAISS-based similarity search
- AI-generated answers using Gemini
- Retrieved source display
- Simple web interface using Gradio

## 🏗️ Architecture

PDF Document  
↓  
Text Extraction  
↓  
Text Chunking  
↓  
Sentence Transformer Embeddings  
↓  
FAISS Vector Search  
↓  
Relevant Document Chunks  
↓  
Gemini AI  
↓  
Answer + Sources

## 🛠️ Technologies Used

- Python
- PyPDF
- Sentence Transformers
- FAISS
- Google Gemini API
- Gradio
- NumPy

## 🧠 How RAG Works

1. The document is converted into text.
2. The text is divided into smaller chunks.
3. Each chunk is converted into a numerical embedding.
4. FAISS stores and searches these embeddings.
5. When a user asks a question, the question is also converted into an embedding.
6. The most relevant document chunks are retrieved.
7. Gemini generates an answer using only the retrieved information.

## ▶️ How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt
