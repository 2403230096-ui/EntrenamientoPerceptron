# Perceptrón Monocapa — Compuerta AND

App web interactiva (Streamlit) que entrena un perceptrón para la compuerta AND
y muestra el procedimiento matemático paso a paso.

## Estructura
- `perceptron.py` — lógica pura de aprendizaje e historial
- `pdf_generator.py` — informe PDF (tabla de épocas + desglose)
- `app.py` — interfaz Streamlit

## Ejecución
```bash
streamlit run app.py
```
Valores sugeridos: w1=0.1, w2=0.3, b=0.2, η=0.1.
