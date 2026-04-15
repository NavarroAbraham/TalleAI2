# Taller Técnico: Desmontando los LLMs — App interactiva

Repositorio para el taller de Deep Learning y Arquitecturas Transformer (Semestre 2026-1).

Contenido
- app.py: aplicación Streamlit con 4 módulos:
  1. Tokenizador (visualización y métricas).
  2. Embeddings 2D (PCA + visualización + prueba algebra vectorial).
  3. Integración con Groq Cloud (generación con control de temperatura y top-p).
  4. Métricas de rendimiento (time per token, throughput, tokens entrada/salida).

- requirements.txt: dependencias necesarias.

Instrucciones rápidas
1. Clona el repositorio.
2. Crea un virtualenv e instala dependencias:
   pip install -r requirements.txt
3. Ejecuta localmente:
   streamlit run app.py
4. Para el módulo Groq: obtén una API Key gratuita en Groq Cloud y pégala en la barra lateral de la app.

Observaciones para la entrega (ensayo requerido)
- Escriba un breve ensayo (1-2 párrafos) en README.md describiendo qué observó sobre Self-Attention al cambiar el contexto del system prompt y del user prompt. Incluya ejemplos concretos de prompts usados y cómo cambió la salida del modelo (determinista vs. creativo) al ajustar temperatura y top-p.

Recomendaciones
- Use el modelo de sentence-transformers `all-MiniLM-L6-v2` para embeddings locales rápidos.
- Verifique modelos de bajo costo / pocos parámetros disponibles en Groq para reducir latencia y costo (ej.: modelos "small" o "mini").
- Despliegue sugerido: Streamlit Community Cloud (opcional).

Fecha: 2026-04-15
