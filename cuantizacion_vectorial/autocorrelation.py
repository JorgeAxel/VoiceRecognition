import numpy as np

def autocorrelation(signal, order):
    """
    Calcula el vector de autocorrelación de una señal.
    
    Parámetros:
        signal: np.array
            Señal de entrada (discreta).
        order: int
            Orden máximo para calcular la autocorrelación.
    
    Retorno:
        r: np.array
            Vector de autocorrelación.
    """
    signal = np.asarray(signal)
    r = np.correlate(signal, signal, mode='full')
    mid = len(signal) - 1
    return r[mid:mid + order + 1]

