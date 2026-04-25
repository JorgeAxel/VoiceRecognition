import numpy as np

def levinson_durbin(r, order):
    """
    Aplica el algoritmo de Levinson-Durbin para calcular los coeficientes LPC.
    
    Parámetros:
        r: np.array
            Vector de autocorrelación.
        order: int
            Orden del modelo LPC.
    
    Retorno:
        a: np.array
            Coeficientes LPC.
        error: float
            Error de predicción final.
    """
    # Inicialización
    a = np.zeros(order + 1)
    e = r[0]
    
    # a[0] siempre es 1 por definición en la ecuación de predicción
    a[0] = 1.0
    
    for i in range(1, order + 1):
        # 1. Calcular Coeficiente de Reflexión (k)
        # Nota: a[0:i] son los coeficientes del paso anterior
        # r[1:i+1][::-1] son las autocorrelaciones r[i], r[i-1]...r[1]
        num = r[i] - np.dot(a[1:i], r[1:i][::-1])
        if e <= 1e-12:
            break
        k = num / e
        
        # 2. Actualizar coeficientes
        # Guardamos temporalmente para evitar el error de actualización in-place
        a_old = a[:i].copy()
        
        a[i] = k
        # La actualización clásica es: a_nuevo = a_viejo - k * a_viejo_reverso
        # Pero en LPC a menudo se define con signo opuesto según la convención
        for j in range(1, i):
            a[j] = a_old[j] - k * a_old[i - j]
        
        # 3. Actualizar el error
        e *= (1 - k**2)
        
    # Retornamos los coeficientes (típicamente a[0] es 1)
    return a, e
