import numpy as np


# =========================
# LPC → LSF (ROBUSTO)
# =========================
def lpc_to_lsf(lpc):
    """
    Convierte coeficientes LPC a LSF (Line Spectral Frequencies).
    LPC: [1, a1, a2, ..., ap]
    Retorna LSF en radianes (rango 0 a pi).
    """
    # 1. Asegurar que el primer coeficiente sea 1 y sea un array de numpy
    a = np.array(lpc) / lpc[0]
    p = len(a) - 1  # Orden del filtro

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
    lsf_p = np.angle(roots_p[np.imag(roots_p) > 0])
    lsf_q = np.angle(roots_q[np.imag(roots_q) > 0])

    # 5. Combinar, ordenar y limpiar resultados
    lsf = np.sort(np.concatenate((lsf_p, lsf_q)))
    
    return lsf


# =========================
# LSF → LPC
# =========================
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


# =========================
# TEST
# =========================
if __name__ == "__main__":

    lpc = np.array([1.0, -0.75, 0.5, -0.25, 0.1])

    print("Coeficientes LPC originales:")
    print(lpc)

    lsf = lpc_to_lsf(lpc)
    print("\nLSF:")
    print(lsf)

    lpc_rec = lsf_to_lpc(lsf)
    print("\nLPC reconstruido:")
    print(lpc_rec)