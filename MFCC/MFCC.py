import librosa
import numpy as np

# Función para extraer características MFCC de una señal de audio
def extract_mfcc(file_path, n_mfcc=13):
    """
    Extrae los coeficientes MFCC de una señal de audio.
    :param file_path: Ruta del archivo de audio.
    :param n_mfcc: Número de coeficientes MFCC a extraer.
    :return: Matriz de características MFCC.
    """
    y, sr = librosa.load(file_path, sr=None)  # Carga la señal de audio
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # Calcula los MFCCs
    return mfccs.T  # Transpuesta para que cada fila sea un vector de características

# Entrenamiento de HMMs con Baum-Welch
def train_hmm(mfcc_features_list, n_states=5):
    """
    Entrena un HMM usando el algoritmo Baum-Welch.
    :param mfcc_features_list: Lista de matrices MFCC (una por palabra).
    :param n_states: Número de estados ocultos en el HMM.
    :return: Modelo HMM entrenado.
    """
    # Inicializa el modelo HMM
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100)

    # Combina todas las características para el entrenamiento
    all_features = np.vstack(mfcc_features_list)
    lengths = [features.shape[0] for features in mfcc_features_list]  # Longitudes de cada palabra

    # Entrena el modelo usando Baum-Welch
    model.fit(all_features, lengths)
    return model

# Reconocimiento de palabras con Viterbi
def recognize_word(hmm_models, mfcc_features):
    """
    Reconoce una palabra utilizando el algoritmo Viterbi.
    :param hmm_models: Diccionario con los modelos HMM (uno por palabra).
    :param mfcc_features: Características MFCC de la palabra a reconocer.
    :return: Palabra reconocida.
    """
    max_log_prob = float("-inf")
    recognized_word = None

    # Calcula la probabilidad de las características para cada modelo
    for word, model in hmm_models.items():
        log_prob = model.score(mfcc_features)  # Log-probabilidad con Viterbi
        if log_prob > max_log_prob:
            max_log_prob = log_prob
            recognized_word = word

    return recognized_word

# Ejemplo práctico
if __name__ == "__main__":
    # Rutas de los archivos de audio para entrenamiento
    word_files = {
        "hola": ["audio/hola1.wav", "audio/hola2.wav"],
        "mundo": ["audio/mundo1.wav", "audio/mundo2.wav"],
        "adios": ["audio/adios1.wav", "audio/adios2.wav"]
    }

    # Entrenamiento de modelos HMM
    hmm_models = {}
    for word, files in word_files.items():
        mfcc_features_list = [extract_mfcc(file) for file in files]
        hmm_models[word] = train_hmm(mfcc_features_list)

    # Reconocimiento de una palabra aislada
    test_file = "audio/test_hola.wav"  # Archivo de prueba
    test_mfcc = extract_mfcc(test_file)
    recognized_word = recognize_word(hmm_models, test_mfcc)

    print(f"Palabra reconocida: {recognized_word}")
