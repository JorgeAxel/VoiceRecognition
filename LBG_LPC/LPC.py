import numpy as np

EPSILON = 1e-10
ORDEN_LPC = 12

def levinson_durbin(autocorr, orden):
    """
    Algoritmo de Levinson-Durbin para obtener coeficientes LPC.

    Implementación del algoritmo directamente para no depender de toolboxes.

    Entrada:
        autocorr: arreglo de coeficientes de autocorrelación [R(0), R(1), ..., R(p)]
        orden:    orden del modelo LPC (p)
    Salida:
        a:    coeficientes LPC [a1, a2, ..., ap] (sin el 1 inicial)
        error: energía del error de predicción

    El algoritmo resuelve el sistema de Yule-Walker:
        [R(0)   R(1)  ... R(p-1)] [a1]   [R(1)]
        [R(1)   R(0)  ... R(p-2)] [a2] = [R(2)]
        ...
        [R(p-1) R(p-2) ... R(0) ] [ap]   [R(p)]
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

    Pasos:
    1. Cálculo de autocorrelación de la trama
    2. Resolución de Levinson-Durbin
    3. Retorno de coeficientes
    """

    # np.correlate computa correlación cruzada; mode='full' da 2N-1 puntos
    # Se usan solo los primeros (orden+1) valores de la autocorrelación
    autocorr_completa = np.correlate(trama, trama, mode='full')
    # La autocorrelación tiene centro en índice N-1. Se toma desde ahí
    centro = len(trama) - 1
    autocorr = autocorr_completa[centro : centro + orden + 1]

    # Normalización por longitud
    autocorr /= len(trama)

    coefs, error = levinson_durbin(autocorr, orden)
    return coefs, error