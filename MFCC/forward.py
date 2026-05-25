import numpy as np

def forward_algorithm(pi, A, B, O):
    """
    Implementa el algoritmo Forward para un HMM.
    
    Parámetros:
    - pi: Vector de probabilidades iniciales (numpy array de tamaño N).
    - A: Matriz de transición (numpy array de tamaño NxN).
    - B: Matriz de emisión (numpy array de tamaño NxM).
    - O: Secuencia observada (lista de índices de observaciones).
    
    Retorna:
    - P(O|lambda): Probabilidad de la secuencia observada.
    - alpha: Matriz alpha (tamaño TxN), donde alpha[t][i] es la probabilidad parcial en el tiempo t para el estado i.
    """
    N = len(pi)        # Número de estados
    T = len(O)         # Longitud de la secuencia observada
    
    # Inicializar la matriz alpha (T x N)
    alpha = np.zeros((T, N))
    
    # Paso 1: Inicialización
    for i in range(N):
        alpha[0, i] = pi[i] * B[i, O[0]]
    
    # Paso 2: Recursión
    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = sum(alpha[t-1, i] * A[i, j] for i in range(N)) * B[j, O[t]]
    
    # Paso 3: Terminación
    P_O_given_lambda = sum(alpha[T-1, i] for i in range(N))
    
    return P_O_given_lambda, alpha

# Parámetros del modelo HMM
# Estados: S1, S2, S3
# Observaciones: S = 0, A = 1
pi = np.array([0.33, 0.33, 0.33])  # Probabilidades iniciales
A = np.array([[0.33, 0.33, 0.33],  # Matriz de transición
              [0.33, 0.33, 0.33],
              [0.33, 0.33, 0.33]])
B = np.array([[0.5, 0.5],  # Matriz de emisión
              [0.75, 0.25],
              [0.25, 0.75]])
O = [0, 0, 0, 0, 1, 0, 1, 1, 1, 1]  # Secuencia observada (SSSSASAAAA)

# Calcular la probabilidad de la secuencia observada
P_O_given_lambda, alpha = forward_algorithm(pi, A, B, O)

# Resultados
print("Probabilidad de la secuencia observada P(O|λ):", P_O_given_lambda)
print("\nMatriz Alpha (probabilidades parciales):")
print(alpha)