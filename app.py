# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
import time
import tiktoken
from sentence_transformers import SentenceTransformer
# Groq client
import groq
# Optional: for OpenAI embeddings if preferred
# import openai

st.set_page_config(layout="wide", page_title="Taller: Desmontando los LLMs")

# ---------- Helpers ----------
@st.cache_resource
def load_tokenizer(encoding_name="cl100k_base"):
    try:
        enc = tiktoken.get_encoding(encoding_name)
    except Exception:
        enc = tiktoken.encoding_for_model("gpt-4o")  # fallback
    return enc

@st.cache_resource
def load_embedding_model(name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(name)
    return model

def tokenize_text(enc, text):
    tokens = enc.encode(text, disallowed_special=[])
    # Get token strings only if tokenizer supports decoding of single ids
    token_strs = [enc.decode([t]) for t in tokens]
    return token_strs, tokens

def compute_embeddings(model, words):
    # sentence-transformers accepts list of strings
    vecs = model.encode(words, normalize_embeddings=False)
    return np.array(vecs)

def reduce_2d(vectors):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(vectors)
    return reduced

def connect_groq(api_key):
    client = groq.Groq(api_key=api_key)
    return client

def groq_generate(client, model, prompt, max_tokens=64, temperature=0.2, top_p=1.0, system_prompt=None):
    # Example: using groq SDK generate call structure; adapt if SDK differs
    payload = {
        "model": model,
        "prompt": (system_prompt + "\n" + prompt) if system_prompt else prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    start = time.time()
    resp = client.generate(**payload)
    end = time.time()
    # Parse resp depending on SDK response format
    text = resp.get("text", "") if isinstance(resp, dict) else getattr(resp, "text", str(resp))
    # Token accounting: attempt to read returned token counts
    resp_info = {
        "text": text,
        "time_s": end - start,
        "tokens_out": resp.get("usage", {}).get("completion_tokens") if isinstance(resp, dict) else None,
        "tokens_in": resp.get("usage", {}).get("prompt_tokens") if isinstance(resp, dict) else None,
    }
    return resp_info

# ---------- UI ----------
st.title("Taller Técnico: Desmontando los LLMs — App interactiva")

st.sidebar.header("Configuración general")
api_key = st.sidebar.text_input("Groq API Key (obligatorio para módulo 3)", type="password")
use_groq = st.sidebar.checkbox("Activar módulo Groq (API)", value=bool(api_key))
embedding_backend = st.sidebar.selectbox("Embeddings backend", ["sentence-transformers", "openai"], index=0)
st.sidebar.markdown("Recomendado: sentence-transformers (local, rápido).")

# Tabs
tabs = st.tabs(["Tokenizador", "Embeddings (2D)", "Groq / Inferencia", "Métricas y Entrega"])

# ---------- Tab 1: Tokenizador ----------
with tabs[0]:
    st.header("Módulo 1: El Laboratorio del Tokenizador")
    enc = load_tokenizer()
    text = st.text_area("Ingrese texto para tokenizar", value="Rey, hombre, mujer, reina")
    if st.button("Tokenizar"):
        token_strs, token_ids = tokenize_text(enc, text)
        # Display tokens with alternate colors using dataframe and HTML styling
        df = pd.DataFrame({
            "token_str": token_strs,
            "token_id": token_ids,
            "chars": [len(s) for s in token_strs]
        })
        # Metrics
        n_chars = len(text)
        n_tokens = len(token_ids)
        col1, col2 = st.columns(2)
        col1.metric("Número de caracteres", n_chars)
        col2.metric("Número de tokens", n_tokens)
        st.markdown("**Tokens (colores alternos)**")
        # Create colored tokens line
        token_spans = []
        for i, t in enumerate(token_strs):
            color = "#D6EAF8" if i % 2 == 0 else "#FADBD8"
            token_spans.append(f"<span style='background:{color};padding:4px;margin:2px;border-radius:4px;'>{t}</span>")
        st.write(" ".join(token_spans), unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)

# ---------- Tab 2: Embeddings ----------
with tabs[1]:
    st.header("Módulo 2: Geometría de las Palabras (Embeddings 2D)")
    embed_model = load_embedding_model()
    words_input = st.text_input("Lista de palabras separadas por comas", value="rey,hombre,mujer,reina")
    words = [w.strip() for w in words_input.split(",") if w.strip()]
    if st.button("Generar embeddings y graficar"):
        with st.spinner("Obteniendo embeddings..."):
            vecs = compute_embeddings(embed_model, words)
            reduced = reduce_2d(vecs)
        df = pd.DataFrame({
            "word": words,
            "x": reduced[:, 0],
            "y": reduced[:, 1]
        })
        fig = px.scatter(df, x="x", y="y", text="word", width=800, height=600)
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)
        # Vector algebra check if relevant words present
        if set(["rey","hombre","mujer","reina"]).issubset(set([w.lower() for w in words])):
            # mapping preserve original order: find indices
            idx = {w.lower(): i for i, w in enumerate(words)}
            v_king = vecs[idx["rey"]]
            v_man = vecs[idx["hombre"]]
            v_woman = vecs[idx["mujer"]]
            v_queen = vecs[idx["reina"]]
            approx = v_king - v_man + v_woman
            # cosine similarity
            def cos(a,b): return np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b))
            sim = cos(approx, v_queen)
            st.write(f"Cosine similarity entre v(king)-v(man)+v(woman) y v(queen): **{sim:.4f}**")
        st.dataframe(df, use_container_width=True)

# ---------- Tab 3: Groq / Inferencia ----------
with tabs[2]:
    st.header("Módulo 3: Interferencia y Razonamiento (Groq API)")
    st.markdown("Ajusta parámetros y genera texto usando Groq.")
    col_a, col_b = st.columns(2)
    with col_a:
        model = st.text_input("Modelo Groq (ej. groq-1-small)", value="groq-1-small")
        system_prompt = st.text_area("System prompt (Instruction tuning)", value="Eres un asistente técnico conciso.")
    with col_b:
        temp = st.slider("Temperatura", 0.0, 1.0, 0.2, 0.01)
        top_p = st.slider("Top-P (nucleus)", 0.0, 1.0, 1.0, 0.01)
        max_tokens = st.number_input("Max tokens (salida)", value=64, min_value=1, max_value=1024)
    user_prompt = st.text_area("User prompt / mensaje", value="Explica brevemente qué hace self-attention.")
    if st.button("Generar con Groq"):
        if not api_key:
            st.error("Se requiere Groq API Key en la barra lateral.")
        else:
            client = connect_groq(api_key)
            with st.spinner("Llamando a la API de Groq..."):
                info = groq_generate(client, model, user_prompt, max_tokens=int(max_tokens), temperature=float(temp), top_p=float(top_p), system_prompt=system_prompt)
            st.write("**Respuesta**")
            st.code(info["text"])
            # Show basic timing info
            time_s = info["time_s"]
            st.metric("Tiempo total (s)", f"{time_s:.3f}")
            # Attempt to compute time per token if token counts available
            if info.get("tokens_out"):
                t_out = info["tokens_out"]
                t_per_token_ms = (time_s / max(1, t_out)) * 1000
                throughput = t_out / time_s
                st.metric("Time per token (ms)", f"{t_per_token_ms:.2f}")
                st.metric("Throughput (tokens/s)", f"{throughput:.1f}")
            else:
                st.info("No se devolvieron conteos de tokens; revise formato de respuesta del SDK.")

# ---------- Tab 4: Métricas y Entrega ----------
with tabs[3]:
    st.header("Módulo 4: Métricas de Desempeño y Entregables")
    st.markdown("""
    - Capture Time per Token (ms), Throughput (tokens/s) y Total Tokens (input vs output).
    - Prepare README.md explicando observaciones sobre self-attention al cambiar el contexto.
    - Incluya requirements.txt y despliegue en Streamlit Community Cloud si lo desea.
    """)
    st.subheader("Checklist de entrega")
    checklist = {
        "Repositorio en GitHub con app.py": True,
        "requirements.txt incluido": True,
        "README.md con ensayo": True,
        "Despliegue (opcional)": False,
        "Verificar modelos low cost en Groq": False
    }
    for k, v in checklist.items():
        st.write(f"- {'✅' if v else '⬜'} {k}")
    st.markdown("Rúbrica de evaluación:")
    st.write("""
    - Tokenización: 20%
    - Embeddings 2D: 30%
    - Integración Groq: 20%
    - Análisis de parámetros: 15%
    - Métricas y razonamiento: 15%
    """)

st.footer = st.write("Fecha: 2026-04-15")
