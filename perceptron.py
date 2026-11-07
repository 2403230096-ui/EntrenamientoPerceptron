"""
perceptron.py
Lógica pura del Perceptrón Monocapa (SLP) para la compuerta AND.

Este módulo contiene únicamente la parte matemática del aprendizaje: no
depende de Streamlit (interfaz) ni de fpdf (reportes), de modo que puede
probarse o reutilizarse por separado.
"""
from dataclasses import dataclass
from typing import List, Tuple

# Tabla de verdad de la compuerta AND, con el formato ((x1, x2), t),
# donde t es la salida deseada (target) para cada combinación de entradas.
DATOS_AND: List[Tuple[Tuple[int, int], int]] = [
    ((0, 0), 0),
    ((0, 1), 0),
    ((1, 0), 0),
    ((1, 1), 1),
]

# Cantidad de decimales usada al redondear los cálculos. Evita errores de
# punto flotante como 0.1 + 0.1 - 0.2 = 2.7e-17 en lugar de 0.
PRECISION = 10


def redondear(v):
    """
    Redondea un valor a PRECISION decimales.

    Se aplica a z y a cada actualización de pesos para que los errores
    de punto flotante no alteren el resultado del escalón (por ejemplo,
    cuando z debería ser exactamente 0).

    Parámetros:
        v (float): Valor a redondear.

    Retorna:
        float: Valor redondeado a PRECISION decimales.
    """
    return round(v, PRECISION)


def fmt(v, dec=4):
    """
    Convierte un número a texto sin ceros sobrantes.

    Se usa para mostrar valores limpios en la interfaz y en el PDF
    (0.1000 se muestra como 0.1 y -0.0 se muestra como 0).

    Parámetros:
        v (float): Número a formatear.
        dec (int): Máximo de decimales a conservar.

    Retorna:
        str: Número como texto, sin ceros finales ni signo "-0".
    """
    # Se formatea con 'dec' decimales y se eliminan ceros y punto finales
    s = f"{round(v, dec):.{dec}f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def escalon(z):
    """
    Función de activación tipo escalón.

    Parámetros:
        z (float): Suma ponderada de las entradas más el bias.

    Retorna:
        int: 1 si z >= 0, 0 en cualquier otro caso.
    """
    return 1 if z >= 0 else 0


@dataclass
class Paso:
    """
    Registro de una muestra evaluada dentro de una época.

    Guarda todo lo necesario para mostrar el procedimiento matemático
    paso a paso: entradas, valores antes y después de la actualización,
    la suma ponderada, la salida y el error.

    Atributos:
        epoca, muestra: Época actual y número de muestra (1 a 4).
        x1, x2, t: Entradas y salida deseada de la muestra.
        w1_ant, w2_ant, b_ant: Pesos y bias antes de actualizar.
        z, y, error: Suma ponderada, salida del escalón y error (t - y).
        w1_nuevo, w2_nuevo, b_nuevo: Pesos y bias después de actualizar.
    """
    epoca: int
    muestra: int
    x1: float
    x2: float
    t: int
    w1_ant: float
    w2_ant: float
    b_ant: float
    z: float
    y: int
    error: int
    w1_nuevo: float
    w2_nuevo: float
    b_nuevo: float


@dataclass
class ResumenEpoca:
    """
    Estado del perceptrón al terminar una época.

    Alimenta la tabla resumen de la interfaz y del PDF.

    Atributos:
        epoca: Número de la época.
        w1, w2, b: Pesos y bias al terminar la época.
        errores: Cantidad de muestras mal clasificadas en la época.
    """
    epoca: int
    w1: float
    w2: float
    b: float
    errores: int


class Perceptron:
    """
    Perceptrón monocapa de dos entradas entrenado con la regla delta.

    Conserva los valores iniciales (para poder reiniciar y para el
    reporte), los pesos actuales y el historial completo del entrenamiento.
    """

    def __init__(self, w1, w2, b, eta, max_epocas, datos=None):
        """
        Crea el perceptrón con sus hiperparámetros.

        Parámetros:
            w1, w2 (float): Pesos iniciales.
            b (float): Bias inicial.
            eta (float): Tasa de aprendizaje.
            max_epocas (int): Límite de épocas de entrenamiento.
            datos (list): Muestras ((x1, x2), t). Por defecto, la AND.

        Retorna:
            None
        """
        # Se guardan los valores iniciales para mostrarlos en el informe
        self.w1_inicial, self.w2_inicial, self.b_inicial = w1, w2, b
        self.eta = eta
        self.max_epocas = max_epocas
        self.datos = datos if datos is not None else DATOS_AND
        self.reiniciar()

    def reiniciar(self):
        """
        Restaura los pesos iniciales y borra el historial.

        Se llama al crear el perceptrón y al inicio de cada entrenamiento,
        para que entrenar dos veces dé siempre el mismo resultado.

        Retorna:
            None
        """
        self.w1, self.w2, self.b = (self.w1_inicial, self.w2_inicial,
                                    self.b_inicial)
        self.historial_pasos: List[Paso] = []         # una entrada por muestra
        self.historial_epocas: List[ResumenEpoca] = []  # una entrada por época
        self.convergio = False

    def predecir(self, x1, x2):
        """
        Calcula la salida del perceptrón con los pesos actuales.

        Se usa tanto durante el entrenamiento como en el módulo de
        inferencia de la interfaz.

        Parámetros:
            x1, x2 (float): Valores de entrada.

        Retorna:
            tuple: (z, y), donde z es la suma ponderada y y la salida
            (0 o 1) tras aplicar el escalón.
        """
        # z = (x1·w1) + (x2·w2) + b
        z = redondear(x1 * self.w1 + x2 * self.w2 + self.b)
        return z, escalon(z)

    def entrenar(self):
        """
        Entrena el perceptrón con la regla delta.

        Recorre las muestras en orden en cada época. Por cada muestra
        calcula z, y y el error e = t - y, y actualiza:
            w = w + eta * e * x      b = b + eta * e
        Termina cuando una época completa no tiene errores o cuando se
        alcanza max_epocas. Registra todo en el historial.

        Retorna:
            bool: True si convergió (época con 0 errores), False si se
            agotó el límite de épocas.
        """
        self.reiniciar()

        for epoca in range(1, self.max_epocas + 1):
            errores = 0  # contador de errores de esta época

            for i, ((x1, x2), t) in enumerate(self.datos, start=1):
                # 1) Suma ponderada y activación
                z, y = self.predecir(x1, x2)

                # 2) Error: salida deseada menos salida obtenida
                e = t - y

                # 3) Regla delta: nuevos pesos y bias
                #    (la entrada asociada al bias vale siempre 1)
                w1n = redondear(self.w1 + self.eta * e * x1)
                w2n = redondear(self.w2 + self.eta * e * x2)
                bn = redondear(self.b + self.eta * e * 1)

                # 4) Se guarda el paso con los valores antes y después
                self.historial_pasos.append(Paso(
                    epoca, i, x1, x2, t,
                    self.w1, self.w2, self.b, z, y, e, w1n, w2n, bn))

                # 5) Se aplican los nuevos valores (el entrenamiento es
                #    por muestra: cada actualización afecta a la siguiente)
                self.w1, self.w2, self.b = w1n, w2n, bn
                if e != 0:
                    errores += 1

            # Al terminar la época se guarda su resumen para la tabla
            self.historial_epocas.append(
                ResumenEpoca(epoca, self.w1, self.w2, self.b, errores))

            # Criterio de paro: ninguna muestra falló en toda la época
            if errores == 0:
                self.convergio = True
                break

        return self.convergio
