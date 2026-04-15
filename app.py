import streamlit as st
import pandas as pd
import plotly.express as px
import tiktoken
from groq import Groq
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import time

# Configuración de la página
st.set_page_config(page_title="Desmontando los LLMs", layout="wide")
st.title("Taller Técnico: Desmontando los LLMs 🤖")
st.markdown("Universidad EAFIT - Prof. Jorge Ivan Padilla Buritica")

# --- Autenticación Groq ---
with st.sidebar:
    st.header("Configuración de API")
    groq_api_key = st.text_input("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", type="password")
    if not groq_api_key:
        st.warning("⚠️ Debes ingresar tu API Key de Groq para usar el Módulo 3 y 4.")

# Crear pestañas para los módulos
tab1, tab2, tab3 = st.tabs(["Módulo 1: Tokenizador", "Módulo 2: Embeddings", "Módulo 3 y 4: Inferencia y Métricas"])

# ==========================================
# Módulo 1: El Laboratorio del Tokenizador
# ==========================================
with tab1:
    st.header("El Laboratorio del Tokenizador")
    texto_input = st.text_area("Ingresa un texto para tokenizar:", "El procesamiento de lenguaje natural es fascinante.")
    
    if st.button("Tokenizar Texto"):
        # Usamos cl100k_base que es el estándar actual para modelos como GPT-4
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(texto_input)
        tokens_text = [enc.decode([t]) for t in tokens]
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Mapeo de Tokens a IDs")
            # Creamos el DataFrame asegurando tipos de datos simples
            df_tokens = pd.DataFrame({
                "Token": tokens_text,
                "Token ID": tokens
            })
            st.dataframe(df_tokens, use_container_width=True)
            
        with col2:
            st.subheader("Métricas Comparativas")
            num_chars = len(texto_input)
            num_tokens = len(tokens)
            st.metric("Número de Caracteres", num_chars)
            st.metric("Número de Tokens", num_tokens)
            if num_tokens > 0:
                st.metric("Ratio (Chars/Token)", f"{(num_chars/num_tokens):.2f}")
        
        st.subheader("Visualización con colores alternos")
        html_string = "<div style='line-height: 1.5; font-size: 18px;'>"
        colors = ["#FFB6C1", "#ADD8E6", "#90EE90", "#FFDAB9", "#E6E6FA"]
        for i, token in enumerate(tokens_text):
            color = colors[i % len(colors)]
            html_string += f"<span style='background-color: {color}; padding: 2px 4px; border-radius: 4px; margin: 2px;'>{token}</span>"
        html_string += "</div>"
        st.markdown(html_string, unsafe_allow_html=True)

# ==========================================
# Módulo 2: Geometría de las Palabras
# ==========================================
with tab2:
    st.header("Geometría de las Palabras (Embeddings)")
    st.markdown("Basado en el espacio de embeddings, reducimos la dimensionalidad para verificar relaciones algebraicas.")
    
    # Cargar modelo local de embeddings
    @st.cache_resource
    def load_embedding_model():
        return SentenceTransformer('all-MiniLM-L6-v2')
    
    model = load_embedding_model()
    
    palabras_input = st.text_input("Ingresa palabras separadas por comas:", "king, man, woman, queen, apple, orange")
    
    if st.button("Proyectar Embeddings a 2D"):
        palabras = [p.strip() for p in palabras_input.split(",") if p.strip()]
        
        if len(palabras) < 3:
            st.error("Por favor, ingresa al menos 3 palabras para aplicar PCA.")
        else:
            with st.spinner('Calculando embeddings...'):
                # Obtener vectores
                embeddings = model.encode(palabras)
                
                # Aplicar PCA para reducir a 2D
                pca = PCA(n_components=2)
                embeddings_2d = pca.fit_transform(embeddings)
                
                df_pca = pd.DataFrame({
                    'Palabra': palabras,
                    'X': embeddings_2d[:, 0],
                    'Y': embeddings_2d[:, 1]
                })
                
                # Graficar con Plotly
                fig = px.scatter(df_pca, x='X', y='Y', text='Palabra', size_max=10, 
                                 title="Plano Cartesiano de Embeddings (PCA a 2D)")
                fig.update_traces(textposition='top center', marker=dict(size=12, color='rgba(152, 0, 0, .8)'))
                st.plotly_chart(fig, use_container_width=True)

# ==========================================
# Módulo 3 y 4: Inferencia y Métricas
# ==========================================
with tab3:
    st.header("Inferencia, Razonamiento y Métricas de Desempeño")
    
    col_params, col_chat = st.columns([1, 2])
    
    with col_params:
        st.subheader("Hiperparámetros")
        modelo_seleccionado = st.selectbox("Modelo (Low cost / rápidos)", ["llama-3.1-8b-instant", "gemma2-9b-it", "mixtral-8x7b-32768"])
        temperatura = st.slider("Temperatura (Determinismo vs Creatividad)", 0.0, 2.0, 0.7, 0.1)
        top_p = st.slider("Top-P (Nucleus Sampling)", 0.0, 1.0, 1.0, 0.1)
        system_prompt = st.text_area("System Prompt (Instruction Tuning):", "Eres un asistente útil y conciso.")
        
    with col_chat:
        st.subheader("Generación")
        user_prompt = st.text_area("User Prompt:", "¿Por qué el cielo es azul? Explícalo en un párrafo.")
        
        if st.button("Generar Respuesta con Groq"):
            if not groq_api_key:
                st.error("Falta la API Key de Groq en la barra lateral.")
            else:
                client = Groq(api_key=groq_api_key)
                
                start_time = time.time()
                try:
                    response = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        model=modelo_seleccionado,
                        temperature=temperatura,
                        top_p=top_p,
                    )
                    end_time = time.time()
                    
                    # Respuesta del modelo
                    st.markdown("**Respuesta:**")
                    st.info(response.choices[0].message.content)
                    
                    # --- Módulo 4: Métricas de Desempeño ---
                    st.divider()
                    st.subheader("Módulo 4: Métricas de Desempeño (Groq Cloud)")
                    
                    # Groq API provee estadísticas de uso en la respuesta
                    usage = response.usage
                    total_time_s = end_time - start_time
                    
                    # Cálculos seguros de métricas
                    completion_tokens = usage.completion_tokens
                    prompt_tokens = usage.prompt_tokens
                    total_tokens = usage.total_tokens
                    
                    time_per_token_ms = (usage.completion_time * 1000) / completion_tokens if completion_tokens > 0 else 0
                    throughput = completion_tokens / usage.completion_time if usage.completion_time > 0 else 0
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Time per Token (ms)", f"{time_per_token_ms:.2f}")
                    m2.metric("Throughput (tokens/s)", f"{throughput:.2f}")
                    m3.metric("Tokens (In / Out / Total)", f"{prompt_tokens} / {completion_tokens} / {total_tokens}")
                    
                except Exception as e:
                    st.error(f"Error en la API: {e}")
