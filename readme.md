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
