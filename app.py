import streamlit as st
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq
import os

# ---------- CONFIG ----------
st.set_page_config(page_title="AI Study Assistant", layout="wide")
MODEL_NAME = "llama-3.1-8b-instant"

# ---------- API KEY ----------
api_key = st.secrets.get("GROQ_API_KEY") or st.text_input("Enter Groq API Key", type="password")

if not api_key:
    st.warning("Please enter your Groq API key")
    st.stop()

client = Groq(api_key=api_key)

# ---------- CACHE MODEL ----------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------- FUNCTIONS ----------
def chunk_text(text, size=400, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = words[i:i + size]
        if len(chunk) > 20:
            chunks.append(" ".join(chunk))
    return chunks

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve(query, chunks, embeddings):
    q_emb = model.encode([query])[0]
    sims = [cosine(q_emb, e) for e in embeddings]
    idx = np.argsort(sims)[-4:][::-1]
    return [chunks[i] for i in idx]

# ---------- SESSION ----------
if "processed" not in st.session_state:
    st.session_state.processed = False

# ---------- UI ----------
st.title("📚 AI Study Assistant")

pdf = st.file_uploader("Upload PDF", type="pdf")

if pdf and not st.session_state.processed:
    with st.spinner("Processing PDF..."):
        try:
            reader = PdfReader(pdf)
            text = ""

            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"

            if len(text.strip()) == 0:
                st.error("No readable text found in PDF")
                st.stop()

            chunks = chunk_text(text)
            embeddings = model.encode(chunks)

            st.session_state.chunks = chunks
            st.session_state.embeddings = embeddings
            st.session_state.processed = True

            st.success("✅ PDF processed successfully!")

        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

# ---------- FEATURES ----------
if st.session_state.processed:
    option = st.selectbox("Choose Feature", ["Q&A", "Summary"])

    # Q&A
    if option == "Q&A":
        question = st.text_input("Ask a question")

        if question:
            rel = retrieve(
                question,
                st.session_state.chunks,
                st.session_state.embeddings
            )

            context = "\n".join(rel)

            prompt = f"""
Answer ONLY using the context below.
If not found, say "Not in document".

Context:
{context}

Question: {question}
"""

            try:
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=MODEL_NAME
                )

                st.write(response.choices[0].message.content)

            except Exception as e:
                st.error(f"Groq Error: {e}")

    # Summary
    if option == "Summary":
        if st.button("Generate Summary"):
            try:
                text_sample = " ".join(st.session_state.chunks[:10])

                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": f"Summarize:\n{text_sample}"}],
                    model=MODEL_NAME
                )

                st.write(response.choices[0].message.content)

            except Exception as e:
                st.error(f"Groq Error: {e}")
