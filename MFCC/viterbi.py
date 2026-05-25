import numpy as np

# Definimos los parámetros del HMM
states = ['S1', 'S2', 'S3']  # Estados ocultos
observations = ['o1', 'o2']  # Observaciones
start_prob = {'S1': 0.33, 'S2': 0.33, 'S3': 0.33}  # Probabilidades iniciales
transition_prob = {
    'S1': {'S1': 0.9, 'S2': 0.05, 'S3': 0.05},
    'S2': {'S1': 0.45, 'S2': 0.1, 'S3': 0.45},
    'S3': {'S1': 0.45, 'S2': 0.45, 'S3': 0.1}
}  # Probabilidades de transición
emission_prob = {
    'S1': {'o1': 0.5, 'o2': 0.5},
    'S2': {'o1': 0.75, 'o2': 0.25},
    'S3': {'o1': 0.25, 'o2': 0.75}
}  # Probabilidades de emisión

# Secuencia de observaciones
obs_sequence = ['o1', 'o2', 'o2', 'o1']

# Inicialización
V = [{}]  # Matriz de probabilidades acumuladas
backpointer = [{}]  # Matriz de backtracking

# Paso 1: Inicialización
for state in states:
    V[0][state] = start_prob[state] * emission_prob[state][obs_sequence[0]]
    backpointer[0][state] = None

# Paso 2: Recursión
for t in range(1, len(obs_sequence)):
    V.append({})
    backpointer.append({})
    for curr_state in states:
        max_prob, prev_state = max(
            (V[t-1][prev_state] * transition_prob[prev_state][curr_state], prev_state)
            for prev_state in states
        )
        V[t][curr_state] = max_prob * emission_prob[curr_state][obs_sequence[t]]
        backpointer[t][curr_state] = prev_state

# Paso 3: Terminación
final_prob, last_state = max((V[len(obs_sequence) - 1][state], state) for state in states)

# Paso 4: Backtracing
best_path = [last_state]
for t in range(len(obs_sequence) - 1, 0, -1):
    best_path.insert(0, backpointer[t][best_path[0]])

# Resultados
print("Probabilidad de la mejor secuencia:", final_prob)
print("Mejor secuencia de estados:", best_path)