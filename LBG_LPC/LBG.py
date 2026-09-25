import numpy as np
from LPC import calcular_lpc

# Coeficiente del filtro de preénfasis.
# Rango típico: 0.90 – 0.97
# - 0.95 es el valor estándar de la práctica
# - Valores más altos -> realzan más las altas frecuencias
# - Valores más bajos -> son más suaves
COEF_PREENFASIS = 0.95

# Longitud de la ventana de análisis en muestras
# Con FS=16000 Hz: 320 muestras = 20 ms (valor estándar de la práctica)
# - Aumento -> mayor resolución espectral, menor resolución temporal
# - Disminución -> menor resolución espectral, mayor resolución temporal
LONGITUD_VENTANA = 320

# Salto (hop) entre tramas consecutivas en muestras
# 128 muestras = 8 ms con FS=16000 Hz
# - Salto menor -> más solapamiento -> más tramas -> más preciso pero más lento
# - Salto mayor -> menos solapamiento -> menos tramas -> más rápido
SALTO = 128

# Orden del modelo LPC
# La práctica especifica 12. Rango típico en voz: 10–16
# - Orden mayor -> más detalle espectral, pero más sensible a ruido
# - Orden menor -> espectro más suavizado, más robusto
ORDEN_LPC = 12

# Parámetros de ventana (se reutilizan las variables LONGITUD_VENTANA y SALTO del Paso 2)

# Coeficiente de preénfasis (se reutiliza la variable COEF_PREENFASIS del Paso 2)

# Valor de regularización (floor de energía de error)
# Evita divisiones por cero en el cálculo de LPC para tramas de silencio
EPSILON = 1e-8

# Número máximo de iteraciones de k-means en cada nivel de splitting
MAX_ITER_KMEANS = 50

# Umbral de convergencia del k-means
# Cuando la mejora relativa en distorsión sea menor que esto, se detiene
EPSILON_CONVERGENCIA = 1e-4

# Factor de perturbación para el splitting
# Cada centroide se divide en: centroide ± EPSILON_SPLIT
EPSILON_SPLIT = 0.001

# Regularización para la distancia de Itakura-Saito
# Evita logaritmos de cero o divisiones por cero
REG_IS = 1e-8

def lpc_a_lsf(lpc):
    """
    Convierte coeficientes LPC a LSF (Line Spectral Frequencies).
    LPC: [1, a1, a2, ..., ap]
    Retorna LSF en radianes (rango 0 a pi).
    """
    p = len(lpc)  # Orden del filtro
    a = np.concatenate(([1.0], lpc))
    a_inv = a[::-1]

    # 2. Definir los polinomios P(z) y Q(z)
    p_poly = a + a_inv
    q_poly = a - a_inv

    # Eliminar las raíces triviales en z=1 y z=-1
    p_reduced = np.polydiv(p_poly, [1, 1])[0]  # Eliminar la raíz en z=1
    q_reduced = np.polydiv(q_poly, [1, -1])[0]  # Eliminar la raíz en z=-1
    # 3. Encontrar las raíces de ambos polinomios
    roots_p = np.roots(p_reduced)
    roots_q = np.roots(q_reduced)

    # 4. Filtrar solo las raíces que están en la mitad superior del círculo unitario
    # y extraer su fase (ángulo)
    lsf_p = np.angle(roots_p[np.imag(roots_p) >= 0])
    lsf_q = np.angle(roots_q[np.imag(roots_q) >= 0])

    # 5. Combinar, ordenar y limpiar resultados
    lsf = np.sort(np.concatenate((lsf_p, lsf_q)))
    
    # Asegurar que el número de LSF coincida con el orden del filtro
    if len(lsf) != p:
        lsf = np.interp(np.linspace(0, 1, p), np.linspace(0, 1, len(lsf)), lsf)

    return lsf

def distancia_itakura_saito_lsf(lsf_a, lsf_b):
    """
    Distancia de Itakura-Saito aproximada en el espacio LSF.

    Dado que la conversión exacta a espectro LPC puede ser costosa trama a trama,
    se usa una aproximación basada en diferencias de LSF con pesos que simulan
    la sensibilidad perceptual:

        d_IS_approx = sum( w_i * (lsf_a_i - lsf_b_i)^2 )

    donde w_i son pesos que dan más importancia a los coeficientes bajos
    (las frecuencias bajas del espectro, más relevantes perceptualmente).

    Esta aproximación ponderada es estándar en literatura de VQ para voz.
    """
    # Pesos decrecientes con el índice: más peso a coefs de baja frecuencia
    orden = len(lsf_a)
    pesos = 1.0 / (np.arange(1, orden + 1))    # w_i = 1/i
    pesos /= pesos.sum()                         # Normalización

    diff = lsf_a - lsf_b
    return np.dot(pesos, diff ** 2)

def distancia_matriz(vectores, centroides):
    """
    Cálculo de la matriz de distancias IS entre todos los vectores y centroides.

    Retorna:
        D: arreglo de shape (num_vectores, num_centroides)
           D[i, j] = distancia_IS entre vector i y centroide j
    """
    
    n_vec = len(vectores)
    n_cen = len(centroides)
    D = np.zeros((n_vec, n_cen))

    for j, centroide in enumerate(centroides):
        for i, vec in enumerate(vectores):
            D[i, j] = distancia_itakura_saito_lsf(vec, centroide)

    return D


def extraer_lsf_señal(señal, fs):
    """
    Aplicación de preénfasis + enmarcado + LPC -> LSF para toda la señal.
    Retorna:
        arreglo 2D de shape (num_tramas, orden_lpc).
    """

    # Preénfasis 
    señal_pre = np.append(señal[0], señal[1:] - COEF_PREENFASIS * señal[:-1])

    # Ventana de Hamming
    ventana   = np.hamming(LONGITUD_VENTANA)
    num_tramas = (len(señal_pre) - LONGITUD_VENTANA) // SALTO + 1

    # Inicialización de matriz de LSF
    matriz_lsf = np.zeros((num_tramas, ORDEN_LPC))

    for i in range(num_tramas):
        inicio = i * SALTO
        trama  = señal_pre[inicio : inicio + LONGITUD_VENTANA] * ventana

        # Saltar tramas de silencio (muy poca energía)
        if np.sum(trama ** 2) < EPSILON:
            continue

        coefs, _ = calcular_lpc(trama, ORDEN_LPC)

        try:
            lsf = lpc_a_lsf(coefs)
            matriz_lsf[i] = lsf
        except Exception:
            # Si la conversión LPC->LSF falla (señal inestable), dejar ceros
            pass

    return matriz_lsf

def kmeans_is(vectores, centroides_iniciales, max_iter=MAX_ITER_KMEANS):
    """
    K-means con distancia de Itakura-Saito.

    Retorna:
        centroides: codevectores finales, shape (k, dimension)
        etiquetas:  índice del centroide más cercano para cada vector
        distorsion: distancia IS promedio final
    """
    centroides = centroides_iniciales.copy()
    distorsion_anterior = np.inf

    for iteracion in range(max_iter):
        # Paso E: asignación de cada vector al centroide más cercano
        D = distancia_matriz(vectores, centroides)
        # argmin a lo largo de los centroides (axis=1)
        etiquetas = np.argmin(D, axis=1)

        # Paso M: actualización de los centroides como media de su cluster
        nuevos_centroides = np.zeros_like(centroides)
        for k in range(len(centroides)):
            miembros = vectores[etiquetas == k]
            if len(miembros) == 0:
                # Cluster vacío: mantener centroide anterior (estrategia estándar)
                nuevos_centroides[k] = centroides[k]
            else:
                nuevos_centroides[k] = miembros.mean(axis=0)

        # Verificación de convergencia
        distorsion_actual = np.mean(np.min(D, axis=1))
        mejora = abs(distorsion_anterior - distorsion_actual)

        centroides = nuevos_centroides
        distorsion_anterior = distorsion_actual

        if mejora < EPSILON_CONVERGENCIA:
            break

    # Recálculo de etiquetas con centroides finales
    D_final = distancia_matriz(vectores, centroides)
    etiquetas = np.argmin(D_final, axis=1)

    return centroides, etiquetas, distorsion_anterior

def lbg_algorithm(vectores, tamaño_codebook):
    """
    Algoritmo LBG (Linde-Buzo-Gray) para entrenamiento de VQ.

    Pasos:
    1. Iniciar con 1 centroide (media global)
    2. Splitting: duplicar cada centroide con perturbación ± ε
    3. K-means hasta convergencia
    4. Repetir desde paso 2 hasta tener tamaño_codebook centroides

    La potencia de 2 del tamaño se logra con log2(tamaño) rondas de splitting.
    Si tamaño_codebook no es potencia de 2, se usa el mayor más cercano.

    Parámetros:
        vectores:        arreglo (N, dim) con todos los vectores de entrenamiento
        tamaño_codebook: número deseado de codevectores (16, 32 o 64)

    Retorna:
        codebook: arreglo (tamaño_codebook, dim) con los codevectores finales
    """
    
    # Número de niveles de splitting (potencia de 2)
    import math
    niveles = int(math.log2(tamaño_codebook))

    # Inicialización con 1 centroide = media de todos los vectores de entrenamiento
    centroides = vectores.mean(axis=0, keepdims=True)   # shape: (1, dim)
    print(f"    Iniciando LBG con 1 centroide, meta: {tamaño_codebook}")

    for nivel in range(niveles):
        n_actuales = len(centroides)
        n_nuevos   = n_actuales * 2

        # Splitting: duplicar cada centroide con perturbación
        # centroide + ε (perturbación positiva)
        # centroide - ε (perturbación negativa)
        perturbacion = EPSILON_SPLIT * np.ones(centroides.shape[1])
        superiores   = centroides + perturbacion
        inferiores   = centroides - perturbacion
        # np.vstack apila verticalmente
        centroides   = np.vstack([superiores, inferiores])

        print(f"    Nivel {nivel+1}/{niveles}: {n_actuales} -> {n_nuevos} centroides", end="")

        # K-means con distancia IS
        centroides, etiquetas, distorsion = kmeans_is(vectores, centroides)

        print(f"  |  distorsión: {distorsion:.6f}")

    return centroides
