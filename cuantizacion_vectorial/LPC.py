import numpy as np
from pre_enfasis import pre_enfasis
from formar_bloques import formar_bloques
from ventana_hamming import ventana_hamming
from autocorrelation import autocorrelation
from levinson_durbin import levinson_durbin

def LPC(signal, orden, tamaño_ventana, desplazamiento):

    # Paso 1: Filtro de pre-énfasis
    señal_preenfatizada = pre_enfasis(signal, alpha=0.95)

    # Paso 2: Formar bloques
    ventanas = formar_bloques(señal_preenfatizada, tamaño_ventana, desplazamiento)

    # Paso 3: Aplicar la ventana de Hamming
    hamming = ventana_hamming(ventanas)
    if hamming.size == 0:
        print("No hay ventanas para procesar.")
        return np.empty((0, orden + 1))  # Retornar arrays vacíos si no hay ventanas
    
    # Compute LPC per frame
    lpc_coefficients_list = []
    for frame in hamming:
        # Paso 4: Calcular la autocorrelación
        R = autocorrelation(frame, orden)

        # Paso 5: Aplicar el método de Levinson-Durbin
        coeficientes_lpc, error_prediccion = levinson_durbin(R, orden)

        coeficientes_lpc = np.array(coeficientes_lpc)

        lpc_coefficients_list.append(coeficientes_lpc)

    return np.array(lpc_coefficients_list)  # Retornar los coeficientes LPC
