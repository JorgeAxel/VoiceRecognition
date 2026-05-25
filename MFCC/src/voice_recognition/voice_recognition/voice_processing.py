import numpy as np
from glob import glob
import librosa
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pickle

COMMANDS = ["start", "pause", "next", "stop"]
EXAMPLE_COMMAND = "next"

USERS = ["axel"]

DATASET_PATH = "dataset"

FRAME_SIZE = 320
HOP_SIZE = 160
ORDER = 12
N_STATES = 5
FREQ = 16000

CODEBOOK_SIZE = 256

config = {

    "FRAME_SIZE": FRAME_SIZE,
    "HOP_SIZE": HOP_SIZE,
    "ORDER": ORDER,
    "N_STATES": N_STATES,
    "FREQ": FREQ,
    "CODEBOOK_SIZE": CODEBOOK_SIZE
}

np.random.seed(0)


"""
=======================
Processing functions
=======================
"""

def preprocess_audio(audio):

    audio = librosa.effects.preemphasis(audio)

    audio = librosa.util.normalize(audio)

    audio, _ = librosa.effects.trim(
        audio,
        top_db=20
    )

    return audio

def extract_mfcc(audio, sr):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=ORDER,
        n_fft=FRAME_SIZE,
        hop_length=HOP_SIZE,
        n_mels=26,
        fmax=8000
    )

    return mfcc.T

def create_bakis_matrix(n_states):

    """
    Crea una matriz de transición Bakis:
    
    Solo permite:
        aii       (mismo estado)
        ai,i+1    (siguiente estado)
    """

    A = np.zeros((n_states, n_states))

    for i in range(n_states):

        # Último estado
        if i == n_states - 1:

            A[i, i] = 1.0

        else:

            # Permanecer en el estado
            A[i, i] = 0.5

            # Avanzar al siguiente
            A[i, i + 1] = 0.5

    return A

def create_emission_matrix(n_states, codebook_size):

    """
    Inicialización uniforme de emisiones
    
    B.shape = (n_states, 256)
    """

    B = np.ones((n_states, codebook_size))

    B = B / codebook_size

    return B

def create_initial_probabilities(n_states):

    """
    El modelo siempre comienza en el estado 0
    """

    pi = np.zeros(n_states)

    pi[0] = 1.0

    return pi

def create_hmm_model(n_states, codebook_size):

    model = {

        "A": create_bakis_matrix(n_states),

        "B": create_emission_matrix(
            n_states,
            codebook_size
        ),

        "pi": create_initial_probabilities(n_states)
    }

    return model

def linear_segmentation(observations, n_states):

    """
    Divide una secuencia de observaciones
    en segmentos iguales.
    """

    T = len(observations)

    segments = []

    for state in range(n_states):

        start = int(state * T / n_states)

        end = int((state + 1) * T / n_states)

        segment = observations[start:end]

        segments.append(segment)

    return segments

def estimate_emission_matrix(
    state_observations,
    n_states,
    codebook_size,
    epsilon=1e-6
):

    """
    Estima probabilidades de emisión
    usando smoothing.
    """

    B = np.ones((n_states, codebook_size)) * epsilon

    for state in range(n_states):

        observations = state_observations[state]

        # Contar ocurrencias
        for symbol in observations:

            B[state, symbol] += 1

        # Normalizar
        total = np.sum(B[state])

        if total > 0:

            B[state] /= total

        else:

            # Evitar división por cero
            B[state] = 1.0 / codebook_size

    return B

def estimate_transition_matrix(
    quantized_sequences,
    n_states
):

    """
    Estima matriz A usando duración promedio
    de cada estado.
    """

    A = np.zeros((n_states, n_states))

    durations = [[] for _ in range(n_states)]

    # Recorrer secuencias
    for observations in quantized_sequences:

        T = len(observations)

        # Segmentación lineal
        for state in range(n_states):

            start = int(state * T / n_states)

            end = int((state + 1) * T / n_states)

            duration = end - start

            durations[state].append(duration)

    # Calcular probabilidades
    for state in range(n_states):

        # Último estado
        if state == n_states - 1:

            A[state, state] = 1.0

        else:

            if len(durations[state]) == 0:

                continue

            avg_duration = np.mean(durations[state])

            avg_duration = max(avg_duration, 2.0)

            aii = (avg_duration - 1) / avg_duration

            aii1 = 1 / avg_duration

            A[state, state] = aii

            A[state, state + 1] = aii1

    return A

def load_training_data():

    all_features = []

    training_sequences = {}

    for command in COMMANDS:

        training_sequences[command] = []

    for user in USERS:

        for command in COMMANDS:

            pattern = f"{DATASET_PATH}/data_{user}/train/{command}/*.wav"

            files = glob(pattern)

            print(f"\n{user} - {command}: {len(files)} archivos")

            for file in files:

                try:

                    # CARGAR AUDIO
                    audio, sr = librosa.load(file, sr=FREQ)

                    # PREPROCESAMIENTO
                    audio = preprocess_audio(audio)

                    if len(audio) == 0:

                        print(f"Empty file: {file}")

                        continue

                    # MFCC
                    mfcc = extract_mfcc(audio, sr)

                    if len(mfcc) < N_STATES:

                        print(f"Too few frames in {file}")

                        continue

                    all_features.append(mfcc)

                    training_sequences[command].append(mfcc)

                except Exception as e:

                    print(f"Error in {file}")

                    print(e)

    if len(all_features) == 0:

        raise ValueError(
            "No valid training data found."
        )

    all_features = np.vstack(all_features)

    return all_features, training_sequences

def train_codebook(all_features):

    print("\nTraining KMeans...")

    kmeans = KMeans(
        n_clusters=CODEBOOK_SIZE,
        random_state=0,
        n_init=20,
        max_iter=300
    )

    kmeans.fit(all_features)

    print("\nCodebook trained")

    print("Codebook shape:")

    print(kmeans.cluster_centers_.shape)

    return kmeans

def quantize_sequences(training_sequences, kmeans):

    quantized_sequences = {}

    for command in COMMANDS:

        quantized_sequences[command] = []

        for mfcc_sequence in training_sequences[command]:

            observations = kmeans.predict(mfcc_sequence)

            quantized_sequences[command].append(
                observations
            )

    return quantized_sequences

def initialize_models(quantized_sequences):

    models = {}

    state_observations = {}

    for command in COMMANDS:

        print(f"\nInitializing model: {command}")

        model = create_hmm_model(
            n_states=N_STATES,
            codebook_size=CODEBOOK_SIZE
        )

        state_observations[command] = {}

        for state in range(N_STATES):

            state_observations[command][state] = []

        # Segmentación lineal
        for observations in quantized_sequences[command]:

            segments = linear_segmentation(
                observations,
                N_STATES
            )

            for state in range(N_STATES):

                state_observations[command][state].extend(
                    segments[state]
                )

        # B
        B = estimate_emission_matrix(
            state_observations[command],
            N_STATES,
            CODEBOOK_SIZE
        )

        # A
        A = estimate_transition_matrix(
            quantized_sequences[command],
            N_STATES
        )

        model["A"] = A

        model["B"] = B

        model["observations"] = quantized_sequences[command]

        models[command] = model

        # Verificaciones
        print("\nA row sums:")

        print(np.sum(A, axis=1))

        print("\nB row sums:")

        print(np.sum(B, axis=1))

    return models

def save_models(
    kmeans,
    models,
    quantized_sequences
):

    with open("Models/codebook.pkl", "wb") as f:

        pickle.dump(kmeans, f)

    with open("Models/models.pkl", "wb") as f:

        pickle.dump(models, f)

    with open("Models/quantized_sequences.pkl", "wb") as f:

        pickle.dump(quantized_sequences, f)
    
    with open("Models/config.pkl", "wb") as f:

        pickle.dump(config, f)

    print("\nModels saved")


"""
=======================
Recognition functions
=======================
"""

def log_sum_exp(log_probs):

    """
    Computa log(sum(exp(log_probs)))
    de forma numéricamente estable.
    """

    m = np.max(log_probs)

    return m + np.log(
        np.sum(np.exp(log_probs - m))
    )

def forward_algorithm(observations, model):

    """
    Forward algorithm en espacio logarítmico.

    observations:
        secuencia cuantizada

    model:
        HMM = {A, B, pi}
    """

    A = model["A"]

    B = model["B"]

    pi = model["pi"]

    N = A.shape[0]

    T = len(observations)

    # Convertir a log
    log_A = np.log(A + 1e-12)

    log_B = np.log(B + 1e-12)

    log_pi = np.log(pi + 1e-12)

    # Forward matrix
    log_alpha = np.zeros((T, N))

    # ====================================
    # Inicialización
    # ====================================

    first_obs = observations[0]

    for j in range(N):

        log_alpha[0, j] = (
            log_pi[j]
            + log_B[j, first_obs]
        )

    # ====================================
    # Recursión
    # ====================================

    for t in range(1, T):

        obs = observations[t]

        for j in range(N):

            transition_probs = []

            for i in range(N):

                transition_probs.append(
                    log_alpha[t - 1, i]
                    + log_A[i, j]
                )

            log_alpha[t, j] = (
                log_sum_exp(
                    np.array(transition_probs)
                )
                + log_B[j, obs]
            )

    # ====================================
    # Terminación
    # ====================================

    log_likelihood = log_sum_exp(
        log_alpha[T - 1]
    )

    return log_likelihood

def recognize_command(observations, models):

    """
    Clasifica una secuencia observada
    usando todos los HMMs.
    """

    best_command = None

    best_score = -np.inf

    scores = {}

    for command, model in models.items():

        score = forward_algorithm(
            observations,
            model
        )

        scores[command] = score

        print(f"{command}: {score}")

        if score > best_score:

            best_score = score

            best_command = command

    return best_command, best_score, scores


"""
=======================
Analysis functions
=======================
"""

def load_test_data():

    test_data = []

    for user in USERS:

        for command in COMMANDS:

            pattern = (
                f"{DATASET_PATH}/data_{user}/test/{command}/*.wav"
            )

            files = glob(pattern)

            print(f"\nTEST {user} - {command}: {len(files)}")

            for file in files:

                try:

                    # Load
                    audio, sr = librosa.load(
                        file,
                        sr=FREQ
                    )

                    # Preprocess
                    audio = preprocess_audio(audio)

                    if len(audio) == 0:

                        continue

                    # MFCC
                    mfcc = extract_mfcc(audio, sr)

                    if len(mfcc) < N_STATES:

                        continue

                    test_data.append({

                        "command": command,

                        "mfcc": mfcc,

                        "file": file
                    })

                except Exception as e:

                    print(f"Error in {file}")

                    print(e)

    return test_data

def evaluate_models(
    test_data,
    kmeans,
    models
):

    y_true = []

    y_pred = []

    for sample in test_data:

        real_command = sample["command"]

        mfcc = sample["mfcc"]

        # Cuantización
        observations = kmeans.predict(mfcc)

        # Reconocimiento
        prediction, score, _ = recognize_command(
            observations,
            models
        )

        y_true.append(real_command)

        y_pred.append(prediction)

        print(
            f"REAL: {real_command}"
            f" | PRED: {prediction}"
            f" | SCORE: {score}"
        )

    return y_true, y_pred

def plot_confusion_matrix(y_true, y_pred):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=COMMANDS
    )

    print("\nConfusion Matrix:")

    print(cm)

    plt.figure(figsize=(8, 6))

    plt.imshow(cm)

    plt.colorbar()

    plt.xticks(
        range(len(COMMANDS)),
        COMMANDS
    )

    plt.yticks(
        range(len(COMMANDS)),
        COMMANDS
    )

    plt.xlabel("Predicted")

    plt.ylabel("True")

    plt.title("Confusion Matrix")

    # Números dentro
    for i in range(len(COMMANDS)):

        for j in range(len(COMMANDS)):

            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.show()

def compute_accuracy(y_true, y_pred):

    correct = sum(
        yt == yp
        for yt, yp in zip(y_true, y_pred)
    )

    accuracy = correct / len(y_true)

    print(f"\nAccuracy: {accuracy:.4f}")

def plot_emission_sparsity(
    models,
    command,
    state
):

    """
    Visualiza sparsity de B
    para un estado específico.
    """

    B = models[command]["B"]

    probs = B[state]

    print(f"\nCommand: {command}")

    print(f"State: {state}")

    print("\nTop probabilities:")

    # Top símbolos más probables
    top_indices = np.argsort(probs)[::-1][:10]

    for idx in top_indices:

        print(
            f"Symbol {idx}: "
            f"{probs[idx]:.6f}"
        )
    
    near_zero = np.sum(probs < 1e-4)

    percentage = 100 * near_zero / len(probs)

    print(
        f"\nNear-zero probabilities: "
        f"{percentage:.2f}%"
    )

    # Plot
    plt.figure(figsize=(12, 4))

    plt.bar(
        range(len(probs)),
        probs
    )

    plt.xlabel("VQ Symbol")

    plt.ylabel("Probability")

    plt.title(
        f"Sparsity of B "
        f"(Command={command}, State={state})"
    )

    plt.tight_layout()

    plt.show()


"""
=======================
Main function
=======================
"""

def main():

    print("\nLoading training data...")

    all_features, training_sequences = load_training_data()

    print("\nTotal feature shape:")

    print(all_features.shape)

    # Entrenar codebook
    kmeans = train_codebook(all_features)

    # Cuantización
    quantized_sequences = quantize_sequences(
        training_sequences,
        kmeans
    )

    # Inicializar HMMs
    models = initialize_models(
        quantized_sequences
    )

    # Guardar
    save_models(
        kmeans,
        models,
        quantized_sequences
    )

    # Verificación final
    print("\n========================")

    print("MODEL CHECK")

    print("========================")

    print(f"\nModel check with example sequence: '{EXAMPLE_COMMAND}'")
    example = models[EXAMPLE_COMMAND]

    print("\nA:")

    print(example["A"])

    print("\npi:")

    print(example["pi"])

    print("\nB shape:")

    print(example["B"].shape)

    print("\nA sums:")

    print(np.sum(example["A"], axis=1))

    print("\nB sums:")

    print(np.sum(example["B"], axis=1))

    print("\nTraining completed successfully")

    print("\n========================")
    print("TEST RECOGNITION")
    print("========================")

    print(f"\nTest recognition with example sequence: '{EXAMPLE_COMMAND}'")
    test_obs = quantized_sequences[EXAMPLE_COMMAND][0]

    prediction, score, scores = recognize_command(
        test_obs,
        models
    )

    print("\nPrediction:")
    print(prediction)

    print("\nLog-likelihood:")
    print(score)

    print("\n========================")
    print("EVALUATION")
    print("========================")

    # Load test data
    test_data = load_test_data()

    # Evaluate
    y_true, y_pred = evaluate_models(
        test_data,
        kmeans,
        models
    )

    # Accuracy
    compute_accuracy(
        y_true,
        y_pred
    )

    # Confusion matrix
    plot_confusion_matrix(
        y_true,
        y_pred
    )

    print("\n========================")
    print("SPARSITY ANALYSIS")
    print("========================")

    plot_emission_sparsity(
        models,
        command=EXAMPLE_COMMAND,
        state=0
    )


if __name__ == "__main__":

    main()