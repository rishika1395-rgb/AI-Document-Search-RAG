import streamlit as st
import numpy as np
import faiss
from io import BytesIO

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
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="AI Document Search using RAG",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# Title
# ============================================================

st.title("🤖 AI-Powered Document Search using RAG")

st.write(
    "Upload a PDF, ask a question, and get an AI-generated "
    "answer based only on the document."
)


# ============================================================
# Gemini API
# ============================================================

try:
    api_key = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)

except Exception:
    st.error("Gemini API key is not configured.")
    st.stop()


# ============================================================
# Embedding Model
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_embedding_model()


# ============================================================
# Build Document Index
# ============================================================

def build_index(pdf_bytes):

    reader = PdfReader(BytesIO(pdf_bytes))

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    if not text.strip():

        return None, []

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

    return index, chunks


# ============================================================
# PDF Upload
# ============================================================

uploaded_file = st.file_uploader(
    "📄 Upload PDF Document",
    type=["pdf"]
)


# ============================================================
# Process PDF
# ============================================================

if uploaded_file:

    if st.button("⚙️ Process Document"):

        with st.spinner("Processing document..."):

            index, chunks = build_index(
                uploaded_file.getvalue()
            )

            st.session_state["index"] = index
            st.session_state["chunks"] = chunks

        st.success(
            f"✅ Document processed successfully! "
            f"{len(chunks)} chunks created."
        )


# ============================================================
# Question Answering
# ============================================================

question = st.text_input(
    "🔍 Ask a question about your document",
    placeholder="Example: What is blockchain?"
)


if st.button("Ask Question"):

    if "index" not in st.session_state:

        st.warning(
            "Please upload and process a PDF first."
        )

    elif not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        index = st.session_state["index"]
        chunks = st.session_state["chunks"]

        with st.spinner("Searching document and generating answer..."):

            # Convert question into embedding

            query_embedding = model.encode(
                [question]
            )

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

        # Display answer

        st.subheader("🤖 AI Answer")

        st.write(response.text)

        # Display sources

        st.subheader("📚 Retrieved Sources")

        for i, idx in enumerate(indices[0]):

            with st.expander(
                f"Source {i + 1}"
            ):

                st.write(
                    chunks[idx][:500]
)
