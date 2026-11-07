"""
app.py
Capa de presentación (Streamlit) del Perceptrón Monocapa - Compuerta AND.

Solo se encarga de la interfaz: pide los hiperparámetros, muestra los
resultados y delega el cálculo en perceptron.py y el reporte en
pdf_generator.py.

Ejecutar con:  python -m streamlit run app.py
"""
import pandas as pd
import streamlit as st

from pdf_generator import generar_pdf
from perceptron import Perceptron, fmt

st.set_page_config(page_title="Perceptrón AND", page_icon="", layout="wide")


# ======================= Funciones auxiliares de vista =======================
def p(v):
    """
    Formatea un número para usarlo dentro de una fórmula LaTeX.

    Los valores negativos se encierran entre paréntesis para que
    expresiones como 1 · (-0.1) se lean correctamente.

    Parámetros:
        v (float): Número a formatear.

    Retorna:
        str: Número como texto; "(-0.1)" si es negativo, "0.1" si no.
    """
    s = fmt(v)
    return f"({s})" if s.startswith("-") else s


def esquema(w1, w2, b, x1=None, x2=None, z=None, y=None):
    """
    Genera el esquema del perceptrón como código Graphviz (DOT).

    Dibuja entradas, pesos, bias, suma ponderada, escalón y salida. Los
    valores opcionales (x1, x2, z, y) se muestran solo si se proporcionan,
    lo que permite reutilizar el mismo esquema antes de entrenar, después
    de entrenar y durante la inferencia.

    Parámetros:
        w1, w2 (float): Pesos que se rotulan en las conexiones.
        b (float): Bias que se muestra en su nodo.
        x1, x2 (float, opcional): Entradas a mostrar en los nodos.
        z (float, opcional): Suma ponderada a mostrar en el nodo Σ.
        y (int, opcional): Salida a mostrar en el nodo final.

    Retorna:
        str: Código DOT para st.graphviz_chart.
    """
    # Etiquetas de los nodos: solo llevan valor si se recibió uno
    lx1 = "x₁" if x1 is None else f"x₁ = {fmt(x1)}"
    lx2 = "x₂" if x2 is None else f"x₂ = {fmt(x2)}"
    lz = "Σ" if z is None else f"Σ\\nz = {fmt(z)}"
    ly = "y" if y is None else f"y = {y}"
    return f"""
    digraph G {{
        rankdir=LR; nodesep=0.6; bgcolor="transparent";
        node [fontname="Helvetica", fontsize=13];
        edge [fontname="Helvetica", fontsize=12];
        x1 [label="{lx1}" shape=circle];
        x2 [label="{lx2}" shape=circle];
        b  [label="b = {fmt(b)}" shape=box style=rounded];
        s  [label="{lz}" shape=circle];
        f  [label="Escalón\\nf(z)" shape=box];
        y  [label="{ly}" shape=doublecircle];
        x1 -> s [label="w₁ = {fmt(w1)}"];
        x2 -> s [label="w₂ = {fmt(w2)}"];
        b  -> s [style=dashed];
        s -> f -> y;
    }}"""


# ================= REQ 2: hiperparámetros (barra lateral) ====================
st.sidebar.header("Hiperparámetros")

# Valores iniciales de pesos y bias (por defecto, los de la práctica)
w1_0 = st.sidebar.number_input("Peso inicial w₁", value=0.1, step=0.1,
                               format="%.4f")
w2_0 = st.sidebar.number_input("Peso inicial w₂", value=0.3, step=0.1,
                               format="%.4f")
b_0 = st.sidebar.number_input("Bias inicial b", value=0.2, step=0.1,
                              format="%.4f")

# Tasa de aprendizaje (mayor que 0) y límite de épocas
eta = st.sidebar.number_input("Tasa de aprendizaje η", min_value=0.0001,
                              value=0.1, step=0.01, format="%.4f")
max_ep = st.sidebar.number_input("Límite máximo de épocas", min_value=1,
                                 max_value=1000, value=20, step=1)

# Botón que dispara el entrenamiento. El modelo se guarda en session_state
# para que sobreviva a los reruns que Streamlit hace en cada interacción.
if st.sidebar.button("Entrenar perceptrón", type="primary"):
    modelo = Perceptron(w1_0, w2_0, b_0, eta, int(max_ep))
    modelo.entrenar()
    st.session_state["modelo"] = modelo

st.title("Perceptrón Monocapa — Compuerta AND")
st.caption("Regla: f(z) = 1 si z ≥ 0, 0 en otro caso · "
           "Datos: (0,0)→0, (0,1)→0, (1,0)→0, (1,1)→1")

# Modelo entrenado (None si todavía no se ha pulsado el botón)
modelo = st.session_state.get("modelo")

# ====================== REQ 1: esquema de la red =============================
st.subheader("1. Esquema de la red")
if modelo is None:
    # Antes de entrenar se muestran los valores iniciales y se detiene la app
    st.graphviz_chart(esquema(w1_0, w2_0, b_0))
    st.info("Configura los hiperparámetros y pulsa **Entrenar perceptrón**.")
    st.stop()

# Después de entrenar el esquema muestra los pesos finales
st.graphviz_chart(esquema(modelo.w1, modelo.w2, modelo.b))
st.caption("Valores finales tras el entrenamiento.")

# Mensaje de resultado según si el perceptrón convergió
if modelo.convergio:
    st.success(f"Convergió en la época {len(modelo.historial_epocas)} "
               f"(0 errores).")
else:
    st.warning("No convergió dentro del límite de épocas.")

# ================== REQ 4: tabla resumen por épocas ==========================
st.subheader("2. Tabla resumen por épocas")
df = pd.DataFrame([{"Época": r.epoca, "w₁": fmt(r.w1), "w₂": fmt(r.w2),
                    "b": fmt(r.b), "Errores": r.errores}
                   for r in modelo.historial_epocas])
# st.table muestra todas las filas completas (st.dataframe podría truncar)
st.table(df.set_index("Época"))

# ============= REQ 3: procedimiento matemático paso a paso ===================
st.subheader("3. Procedimiento matemático paso a paso")

# Valores iniciales con los que empezó el entrenamiento
st.latex(rf"\eta = {fmt(modelo.eta)},\quad "
         rf"w_1^{{(0)}}={fmt(modelo.w1_inicial)},\ "
         rf"w_2^{{(0)}}={fmt(modelo.w2_inicial)},\ "
         rf"b^{{(0)}}={fmt(modelo.b_inicial)}")

# Un bloque desplegable por época, abierto para no ocultar ningún dato
for r in modelo.historial_epocas:
    with st.expander(f"Época {r.epoca} — {r.errores} error(es)",
                     expanded=True):
        # Solo las muestras que pertenecen a esta época
        for s in [s for s in modelo.historial_pasos if s.epoca == r.epoca]:
            st.markdown(f"**Muestra {s.muestra}:** "
                        f"$(x_1,x_2)=({fmt(s.x1)},{fmt(s.x2)})$, $t={s.t}$")
            # Suma ponderada con valores sustituidos
            st.latex(rf"z=(x_1\cdot w_1)+(x_2\cdot w_2)+b="
                     rf"({p(s.x1)}\cdot {p(s.w1_ant)})+"
                     rf"({p(s.x2)}\cdot {p(s.w2_ant)})+{p(s.b_ant)}"
                     rf"={fmt(s.z)}")
            # Salida del escalón y cálculo del error
            st.latex(rf"y=f(z)={s.y}\qquad e=t-y={s.t}-{s.y}={s.error}")
            # Regla delta aplicada a w1, w2 y b
            st.latex(rf"w_1^{{(n)}}={p(s.w1_ant)}+{p(modelo.eta)}\cdot "
                     rf"{p(s.error)}\cdot {p(s.x1)}={fmt(s.w1_nuevo)}")
            st.latex(rf"w_2^{{(n)}}={p(s.w2_ant)}+{p(modelo.eta)}\cdot "
                     rf"{p(s.error)}\cdot {p(s.x2)}={fmt(s.w2_nuevo)}")
            st.latex(rf"b^{{(n)}}={p(s.b_ant)}+{p(modelo.eta)}\cdot "
                     rf"{p(s.error)}\cdot 1={fmt(s.b_nuevo)}")
            st.divider()
        # Resumen al cierre de la época
        st.markdown(f"**Fin de la época {r.epoca}:** "
                    f"$w_1={fmt(r.w1)},\\ w_2={fmt(r.w2)},\\ b={fmt(r.b)}$, "
                    f"errores = {r.errores}")

# ====================== REQ 5: exportación a PDF =============================
st.subheader("4. Exportar informe")
# El PDF se genera con pdf_generator.py a partir del modelo entrenado
st.download_button("Descargar informe PDF", data=generar_pdf(modelo),
                   file_name="informe_perceptron_AND.pdf",
                   mime="application/pdf")

# ================= REQ 6: módulo de inferencia / prueba ======================
st.subheader("5. Módulo de inferencia")
# Entradas libres: el usuario puede probar cualquier combinación (x1, x2)
c1, c2 = st.columns(2)
x1 = c1.number_input("x₁", value=1.0, step=1.0)
x2 = c2.number_input("x₂", value=1.0, step=1.0)

# Se evalúa con los pesos finales y se actualiza el esquema en tiempo real
z, y = modelo.predecir(x1, x2)
st.graphviz_chart(esquema(modelo.w1, modelo.w2, modelo.b, x1, x2, z, y))
st.latex(rf"z=({p(x1)}\cdot {p(modelo.w1)})+({p(x2)}\cdot {p(modelo.w2)})"
         rf"+{p(modelo.b)}={fmt(z)}\ \Rightarrow\ y=f(z)={y}")
