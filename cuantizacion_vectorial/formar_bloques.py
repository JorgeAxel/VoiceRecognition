import numpy as np

def formar_bloques(signal, tamaño_ventana, desplazamiento):
    """
    Divide una señal en ventanas con solapamiento de forma vectorizada.

    Parámetros:
    - signal: Arreglo 1D de la señal de voz.
    - tamaño_ventana: Tamaño de cada ventana en muestras.
    - desplazamiento: Desplazamiento entre ventanas en muestras.

    Retorna:
    - Arreglo 2D de forma (num_ventanas, tamaño_ventana), donde cada fila es una ventana.
    """
    signal = np.asarray(signal)

    if signal.ndim != 1:
        raise ValueError("La señal debe ser 1D")

    # Número de ventanas
    if len(signal) < tamaño_ventana:
        return np.empty((0, tamaño_ventana))
    num_ventanas = 1 + (len(signal) - tamaño_ventana) // desplazamiento
    if num_ventanas <= 0:
        return np.empty((0, tamaño_ventana))

    # Crear índices de forma vectorizada
    indices = (
        np.arange(tamaño_ventana)[None, :] +
        np.arange(num_ventanas)[:, None] * desplazamiento
    )

    return signal[indices]
