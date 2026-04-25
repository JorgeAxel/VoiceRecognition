import numpy as np
from scipy.signal import freqz
#import matplotlib.pyplot as plt

# ==================================================
# 1. LPC -> LSF
# ==================================================

def lpc_to_lsf(lpc):
    """
    Convierte coeficientes LPC a LSF (Line Spectral Frequencies).
    LPC: [1, a1, a2, ..., ap]
    Retorna LSF en radianes (rango 0 a pi).
    """
    # 1. Asegurar que el primer coeficiente sea 1 y sea un array de numpy
    a = np.array(lpc) / lpc[0]
    p = len(a) - 1  # Orden del filtro

    if np.any(np.abs(np.roots(a)) >= 1):
        return None

    # 2. Definir los polinomios P(z) y Q(z)
    # P(z) = A(z) + z^{-(p+1)} A(z^{-1})
    # Q(z) = A(z) - z^{-(p+1)} A(z^{-1})
    p_poly = np.poly1d(np.concatenate((a, [0]))) + np.poly1d(np.concatenate(([0], a[::-1])))
    q_poly = np.poly1d(np.concatenate((a, [0]))) - np.poly1d(np.concatenate(([0], a[::-1])))

    # 3. Encontrar las raíces de ambos polinomios
    roots_p = p_poly.roots
    roots_q = q_poly.roots

    # 4. Filtrar solo las raíces que están en la mitad superior del círculo unitario
    # y extraer su fase (ángulo)
    #lsf_p = np.angle(roots_p[np.imag(roots_p) > 0])
    #lsf_q = np.angle(roots_q[np.imag(roots_q) > 0])
    eps = 1e-6
    lsf_p = np.angle(roots_p[np.imag(roots_p) > eps])
    lsf_q = np.angle(roots_q[np.imag(roots_q) > eps])

    # 5. Combinar, ordenar y limpiar resultados
    lsf = np.sort(np.concatenate((lsf_p, lsf_q)))
    
    if len(lsf) != p:
        return None

    return lsf

def lpc_spectrum_batch(lpc_array, n_freqs=256):
    """
    lpc_array: (N x p)
    devuelve: (N x n_freqs)
    """
    spectra = []
    for a in lpc_array:
        _, h = freqz([1], a, worN=n_freqs)
        spectra.append(np.abs(h)**2)
    return np.array(spectra)

# ==================================================
# 2. LSF -> LPC
# ==================================================

def lsf_to_lpc(lsf):
    """
    Convierte coeficientes LSF a LPC.
    lsf: array de frecuencias en radianes (0 a pi).
    Retorna lpc: [1, a1, a2, ..., ap]
    """
    p = len(lsf)
    
    # 1. Reconstruir las raíces en el círculo unitario (conjugados complejos)
    # Las raíces de P y Q están en e^(j*omega)
    roots_p = np.exp(1j * lsf[::2])
    roots_q = np.exp(1j * lsf[1::2])
    
    # Añadir los pares conjugados
    roots_p = np.concatenate((roots_p, np.conj(roots_p)))
    roots_q = np.concatenate((roots_q, np.conj(roots_q)))
    
    # 2. Volver a formar los polinomios P(z) y Q(z)
    poly_p = np.poly(roots_p)
    poly_q = np.poly(roots_q)
    
    # 3. Reintroducir las raíces triviales eliminadas en la conversión original
    # P tiene una raíz en z = -1 (z + 1)
    # Q tiene una raíz en z = 1 (z - 1)
    P = np.convolve(poly_p, [1.0, 1.0])
    Q = np.convolve(poly_q, [1.0, -1.0])
    
    # 4. El polinomio original es A(z) = (P(z) + Q(z)) / 2
    lpc = 0.5 * (P + Q)
    
    # Devolvemos solo la parte real (la imaginaria es despreciable por redondeo)
    # y recortamos al orden original p + 1
    return np.real(lpc[:p+1])


# ==================================================
# 3. Itakura-Saito distance
# ==================================================
def itakura_saito_matrix(P_data, P_code):
    """
    P_data: (N x F)
    P_code: (K x F)

    devuelve: (N x K) distancias
    """
    # Expand dims para broadcasting
    P_d = P_data[:, None, :]     # (N x 1 x F)
    P_c = P_code[None, :, :]     # (1 x K x F)

    ratio = np.maximum(P_d, 1e-12) / np.maximum(P_c, 1e-12)

    d = np.mean(ratio - np.log(ratio + 1e-12) - 1, axis=2)
    return d


# ==================================================
# 4. Inicialización LBG (splitting)
# ==================================================
def split_codebook(codebook, epsilon=0.01):
    new_codebook = []
    for c in codebook:
        new_codebook.append(c * (1 + epsilon))
        new_codebook.append(c * (1 - epsilon))
    return np.array(new_codebook)


# ==================================================
# 5. Clustering en LSF
# ==================================================
def lbg_algorithm(data_lpc, k=16, max_iter=30, n_freqs=128):
    
    # LPC -> LSF
    #data_lsf = np.array([lpc_to_lsf(a) for a in data_lpc])
    lpc_filtered = []
    lsf_list = []
    for a in data_lpc:
        lsf = lpc_to_lsf(a)
        if lsf is None:
            continue
        lpc_filtered.append(a)
        lsf_list.append(lsf)
    data_lpc_filtered = np.array(lpc_filtered)
    data_lsf = np.array(lsf_list)
    
    # inicialización
    centroid = np.mean(data_lsf, axis=0)
    codebook_lsf = np.array([centroid])
    
    P_data = lpc_spectrum_batch(data_lpc_filtered, n_freqs)

    while len(codebook_lsf) < k:
        codebook_lsf = split_codebook(codebook_lsf)

        for _ in range(max_iter):

            # LSF -> LPC
            codebook_lpc = np.array([lsf_to_lpc(c) for c in codebook_lsf])

            # PRECOMPUTAR espectros
            P_code = lpc_spectrum_batch(codebook_lpc, n_freqs)

            # DISTANCIAS VECTORIAL
            D = itakura_saito_matrix(P_data, P_code)

            # asignación
            labels = np.argmin(D, axis=1)

            # actualización en LSF (única parte no vectorizable fácil)
            new_codebook = []
            for i in range(len(codebook_lsf)):
                cluster = data_lsf[labels == i]

                if len(cluster) == 0:
                    new_codebook.append(
                        data_lsf[np.random.randint(len(data_lsf))]
                    )
                else:
                    mean_lsf = np.mean(cluster, axis=0)
                    mean_lsf = np.sort(mean_lsf)
                    eps = 1e-3
                    for j in range(1, len(mean_lsf)):
                        if mean_lsf[j] <= mean_lsf[j-1]:
                            mean_lsf[j] = mean_lsf[j-1] + eps
                    
                    new_codebook.append(mean_lsf)

            new_codebook = np.array(new_codebook)

            if np.allclose(codebook_lsf, new_codebook, atol=1e-4):
                break

            codebook_lsf = new_codebook

    codebook_lpc = np.array([lsf_to_lpc(c) for c in codebook_lsf])
    return codebook_lpc, codebook_lsf

"""
# data_lpc: (N_frames x order+1)
# Coeficientes LPC proporcionados (ejemplo)
data_lpc = np.array([[1.0, -0.75, 0.5, -0.25, 0.1]]) # Ejemplo de coeficientes LPC
codebook_lpc, codebook_lsf = lbg_algorithm(data_lpc, k=8)
lsf = lpc_to_lsf(data_lpc[0])
reconstructed_lpc = lsf_to_lpc(lsf)
print("LPC:\n", data_lpc)
print("LSF:\n", lsf)
print("Reconstructed LPC:\n", reconstructed_lpc)
#print("Codebook LPC:\n", codebook_lpc)
#print("Codebook LSF:\n", codebook_lsf)


P_data = lpc_spectrum_batch(data_lpc)
P_code = lpc_spectrum_batch(codebook_lpc)
plt.figure(figsize=(10, 6))
plt.plot(10 * np.log10(P_data[0]), label="Dato Original", linewidth=3, color='black')
for i, p in enumerate(P_code):
    plt.plot(10 * np.log10(p), alpha=0.5, linestyle='--', label=f"Centroide {i}")

plt.title("Comparación de Espectros: Datos vs Codebook")
plt.ylabel("Amplitud (dB)")
plt.legend()
plt.show()
"""