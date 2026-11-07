"""
pdf_generator.py
Generador dinámico del informe PDF del perceptrón.

Recibe un perceptrón ya entrenado y construye un documento con la tabla
resumen por épocas y el procedimiento matemático de cada muestra. No
depende de Streamlit: solo usa fpdf2 y los datos del modelo.
"""
from fpdf import FPDF

from perceptron import Perceptron, fmt


class _Informe(FPDF):
    """
    Documento PDF con pie de página personalizado.

    Extiende FPDF únicamente para agregar el número de página al final
    de cada hoja.
    """

    def footer(self):
        """
        Dibuja el número de página al pie. FPDF lo llama automáticamente
        al terminar cada página.

        Retorna:
            None
        """
        self.set_y(-12)  # 12 mm antes del borde inferior
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 8, f"Pagina {self.page_no()}", align="C")


def _linea(pdf, texto, fuente="Courier", tam=9, estilo=""):
    """
    Escribe un renglón de texto (con salto de línea automático).

    Función auxiliar que evita repetir la configuración de fuente y el
    salto de línea cada vez que se agrega texto al documento.

    Parámetros:
        pdf (FPDF): Documento donde se escribe.
        texto (str): Texto a escribir.
        fuente (str): Familia tipográfica ("Courier" para fórmulas).
        tam (int): Tamaño de la letra en puntos.
        estilo (str): "B" negrita, "I" cursiva o "" normal.

    Retorna:
        None
    """
    pdf.set_font(fuente, estilo, tam)
    # new_x/new_y regresan el cursor al margen izquierdo, en la línea siguiente
    pdf.multi_cell(0, 5, texto, new_x="LMARGIN", new_y="NEXT")


def generar_pdf(p):
    """
    Genera el informe PDF completo de un perceptrón entrenado.

    Estructura del documento:
        1. Título y parámetros iniciales.
        2. Tabla resumen por épocas y resultado de convergencia.
        3. Procedimiento paso a paso (z, y, e y regla delta por muestra).

    Los textos usan solo caracteres latinos y notación simple (eta, *)
    porque las fuentes básicas de PDF no incluyen símbolos griegos.

    Parámetros:
        p (Perceptron): Modelo ya entrenado (con historial).

    Retorna:
        bytes: Contenido del PDF, listo para st.download_button.
    """
    pdf = _Informe(format="Letter")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ---- Título ----
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Perceptron Monocapa - Compuerta AND", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # ---- Parámetros iniciales ----
    _linea(pdf, "Parametros iniciales", "Helvetica", 12, "B")
    _linea(pdf, f"w1 = {fmt(p.w1_inicial)}   w2 = {fmt(p.w2_inicial)}   "
                f"b = {fmt(p.b_inicial)}   eta = {fmt(p.eta)}   "
                f"epocas max = {p.max_epocas}")
    _linea(pdf, "Activacion: f(z) = 1 si z >= 0; 0 si z < 0")
    pdf.ln(3)

    # ---- Tabla resumen por épocas ----
    _linea(pdf, "Tabla resumen por epocas", "Helvetica", 12, "B")
    pdf.set_font("Helvetica", "", 10)
    with pdf.table(col_widths=(15, 25, 25, 25, 20),
                   text_align="CENTER") as tabla:
        # Fila de encabezados
        fila = tabla.row()
        for h in ("Epoca", "w1", "w2", "b", "Errores"):
            fila.cell(h)
        # Una fila por época registrada en el historial
        for r in p.historial_epocas:
            fila = tabla.row()
            for v in (r.epoca, fmt(r.w1), fmt(r.w2), fmt(r.b), r.errores):
                fila.cell(str(v))
    pdf.ln(3)

    # ---- Resultado del entrenamiento ----
    if p.convergio:
        _linea(pdf, f"Convergio en la epoca {len(p.historial_epocas)} "
                    f"(0 errores). Pesos finales: w1 = {fmt(p.w1)}, "
                    f"w2 = {fmt(p.w2)}, b = {fmt(p.b)}", "Helvetica", 10)
    else:
        _linea(pdf, "No convergio dentro del limite de epocas.",
               "Helvetica", 10)

    # ---- Procedimiento paso a paso (en página nueva) ----
    pdf.add_page()
    _linea(pdf, "Procedimiento paso a paso", "Helvetica", 14, "B")

    epoca_actual = 0
    for s in p.historial_pasos:
        # Al cambiar de época se imprime un encabezado nuevo
        if s.epoca != epoca_actual:
            epoca_actual = s.epoca
            pdf.ln(2)
            _linea(pdf, f"EPOCA {epoca_actual}", "Helvetica", 12, "B")

        # Datos de la muestra evaluada
        _linea(pdf, f"Muestra {s.muestra}: (x1, x2) = ({fmt(s.x1)}, "
                    f"{fmt(s.x2)}),  t = {s.t}", "Helvetica", 10, "B")
        # Suma ponderada con los valores sustituidos
        _linea(pdf, f"z = (x1*w1) + (x2*w2) + b = ({fmt(s.x1)}*{fmt(s.w1_ant)})"
                    f" + ({fmt(s.x2)}*{fmt(s.w2_ant)}) + {fmt(s.b_ant)}"
                    f" = {fmt(s.z)}")
        # Salida del escalón y error
        _linea(pdf, f"y = f(z) = {s.y}   ->   e = t - y = {s.t} - {s.y}"
                    f" = {s.error}")
        # Regla delta para w1, w2 y b
        _linea(pdf, f"w1 = {fmt(s.w1_ant)} + {fmt(p.eta)}*({s.error})*"
                    f"{fmt(s.x1)} = {fmt(s.w1_nuevo)}")
        _linea(pdf, f"w2 = {fmt(s.w2_ant)} + {fmt(p.eta)}*({s.error})*"
                    f"{fmt(s.x2)} = {fmt(s.w2_nuevo)}")
        _linea(pdf, f"b  = {fmt(s.b_ant)} + {fmt(p.eta)}*({s.error})*1"
                    f" = {fmt(s.b_nuevo)}")
        pdf.ln(2)

        # Tras la última muestra de la época, se imprime su resumen
        if s.muestra == len(p.datos):
            r = p.historial_epocas[s.epoca - 1]
            _linea(pdf, f"Fin de epoca {s.epoca}: errores = {r.errores}  "
                        f"w1 = {fmt(r.w1)}  w2 = {fmt(r.w2)}  b = {fmt(r.b)}",
                   "Helvetica", 10, "I")

    # fpdf2 devuelve un bytearray; se convierte a bytes para Streamlit
    return bytes(pdf.output())
