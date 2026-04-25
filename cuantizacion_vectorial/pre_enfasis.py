import numpy as np

def pre_enfasis(signal, alpha=0.95):
    """
    Aplica el filtro de pre-enfasis a una señal de voz.
    
    Parámetros:
    - signal: Lista o arreglo que representa la señal de voz.
    - alpha: Coeficiente de pre-enfasis (típicamente 0.97).

    Retorna:
    - emphasized: Señal pre-enfatizada.
    """
    signal = np.asarray(signal, dtype=float)
    emphasized = np.empty_like(signal)
    emphasized[0] = signal[0]
    emphasized[1:] = signal[1:] - alpha * signal[:-1]
    return emphasized
