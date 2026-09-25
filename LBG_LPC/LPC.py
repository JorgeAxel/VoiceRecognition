import numpy as np

EPSILON = 1e-10
ORDEN_LPC = 12

def levinson_durbin(autocorr, orden):
    """
    Algoritmo de Levinson-Durbin para obtener coeficientes LPC.
    """

    a      = np.zeros(orden)      # Coeficientes LPC
    error  = autocorr[0]          # Error inicial = R(0) = energía total

    if error < EPSILON:
        return a, error           # Trama de silencio -> retornar ceros

    for i in range(orden):
        # Coeficiente de reflexión (parámetro PARCOR)
        if error < EPSILON:
            break
        k = -np.dot(a[:i], autocorr[i:0:-1]) - autocorr[i + 1]
        k /= error

        # Actualizar coeficientes
        a_nuevo    = a.copy()
        a_nuevo[i] = k
        a_nuevo[:i] += k * a[:i][::-1]   # [::-1] invierte el arreglo
        a = a_nuevo

        # Actualizar error de predicción
        error *= (1 - k ** 2)

    return a, max(error, EPSILON)

def calcular_lpc(trama, orden=ORDEN_LPC):
    """
    Cálculo de los coeficientes LPC de una trama.
    """

    # np.correlate computa correlación cruzada
    autocorr_completa = np.correlate(trama, trama, mode='full')
    # La autocorrelación tiene centro en índice N-1
    centro = len(trama) - 1
    autocorr = autocorr_completa[centro : centro + orden + 1]

    # Normalización por longitud
    autocorr /= len(trama)

    coefs, error = levinson_durbin(autocorr, orden)
    return coefs, error
