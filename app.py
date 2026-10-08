
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

PDF_PATH = "unit 2.pdf"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 3


# ============================================================
# Gemini
# ============================================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

client = genai.Client(api_key=api_key)


# ============================================================
# Load PDF
# ============================================================

reader = PdfReader(PDF_PATH)

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"


# ============================================================
# Create chunks
# ============================================================

chunks = []

start = 0

while start < len(text):

    end = start + CHUNK_SIZE

    chunk = text[start:end]

    if chunk.strip():
        chunks.append(chunk.strip())

    start += CHUNK_SIZE - CHUNK_OVERLAP


# ============================================================
# Create embeddings
# ============================================================

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)


# ============================================================
# FAISS index
# ============================================================

embedding_matrix = np.array(embeddings).astype("float32")

dimension = embedding_matrix.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embedding_matrix)


# ============================================================
# RAG function
# ============================================================

def rag_interface(question):

    if not question.strip():
        return "Please enter a question.", ""


    # Convert question into embedding

    query_embedding = model.encode([question])


    # Retrieve relevant chunks

    distances, indices = index.search(
        np.array(query_embedding).astype("float32"),
        k=TOP_K
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
# Gradio UI
# ============================================================

demo = gr.Interface(

    fn=rag_interface,

    inputs=gr.Textbox(
        label="Ask a question about your document",
        placeholder="Example: What is blockchain?"
    ),

    outputs=[
        gr.Textbox(label="🤖 AI Answer"),
        gr.Textbox(label="📚 Retrieved Sources")
    ],

    title="AI-Powered Document Search using RAG",

    description=(
        "Ask questions about the uploaded PDF "
        "and view the document sources used to generate the answer."
    )
)


# ============================================================
# Launch
# ============================================================

demo.launch()
