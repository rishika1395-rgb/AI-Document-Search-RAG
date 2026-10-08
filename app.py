import os
import numpy as np
import faiss
import gradio as gr

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from google import genai


# ============================================================
# Configuration
# ============================================================

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 3


# ============================================================
# Gemini API
# ============================================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

client = genai.Client(api_key=api_key)


# ============================================================
# Embedding Model
# ============================================================

model = SentenceTransformer("all-MiniLM-L6-v2")


# ============================================================
# Build Document Index
# ============================================================

def build_index(pdf_path):

    if not pdf_path:
        return None, [], "Please upload a PDF document."

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    if not text.strip():
        return None, [], "Could not extract text from the PDF."

    # Create chunks
    chunks = []

    start = 0

    while start < len(text):

        end = start + CHUNK_SIZE

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start += CHUNK_SIZE - CHUNK_OVERLAP

    # Create embeddings
    embeddings = model.encode(chunks)

    embedding_matrix = np.array(
        embeddings
    ).astype("float32")

    # Create FAISS index
    dimension = embedding_matrix.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embedding_matrix)

    return (
        index,
        chunks,
        f"✅ Document processed successfully! {len(chunks)} chunks created."
    )


# ============================================================
# RAG Question Answering
# ============================================================

def rag_interface(question, index, chunks):

    if not question.strip():
        return "Please enter a question.", ""

    if index is None or not chunks:
        return "Please upload and process a PDF first.", ""

    # Convert question into embedding
    query_embedding = model.encode([question])

    # Retrieve relevant chunks
    distances, indices = index.search(
        np.array(query_embedding).astype("float32"),
        k=min(TOP_K, len(chunks))
    )

    # Create context
    context = "\n\n".join(
        [chunks[idx] for idx in indices[0]]
    )

    # Prompt
    prompt = f"""
You are a document question-answering assistant.

Answer the question using ONLY the information provided in the context.

Rules:
- Do not use outside knowledge.
- Do not invent or guess information.
- If the answer is not present in the context, say:
  "The answer is not available in the provided document."
- Give a clear and simple answer.

Context:
{context}

Question:
{question}

Answer:
"""

    # Generate answer
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt
    )

    # Retrieved sources
    sources = ""

    for i, idx in enumerate(indices[0]):

        sources += f"--- Source {i + 1} ---\n"
        sources += chunks[idx][:500]
        sources += "\n\n"

    return response.text, sources


# ============================================================
# Gradio Interface
# ============================================================

with gr.Blocks() as demo:

    gr.Markdown(
        "# 🤖 AI-Powered Document Search using RAG"
    )

    gr.Markdown(
        "Upload a PDF, ask a question, and get an AI-generated "
        "answer based on the document."
    )

    pdf_file = gr.File(
        label="📄 Upload PDF Document",
        file_types=[".pdf"],
        type="filepath"
    )

    process_button = gr.Button(
        "⚙️ Process Document"
    )

    status = gr.Textbox(
        label="Document Status"
    )

    index_state = gr.State()
    chunks_state = gr.State()

    process_button.click(
        fn=build_index,
        inputs=pdf_file,
        outputs=[
            index_state,
            chunks_state,
            status
        ]
    )

    question = gr.Textbox(
        label="Ask a question about your document",
        placeholder="Example: What is blockchain?"
    )

    ask_button = gr.Button(
        "🔍 Ask Question"
    )

    answer = gr.Textbox(
        label="🤖 AI Answer",
        lines=5
    )

    sources = gr.Textbox(
        label="📚 Retrieved Sources",
        lines=12
    )

    ask_button.click(
        fn=rag_interface,
        inputs=[
            question,
            index_state,
            chunks_state
        ],
        outputs=[
            answer,
            sources
        ]
    )


# ============================================================
# Launch
# ============================================================

demo.launch()
