import numpy as np
from scipy.io import wavfile
from glob import glob
from LPC import LPC
from LBG import lpc_to_lsf, lsf_to_lpc, lpc_spectrum_batch, itakura_saito_matrix

# cargar codebooks
codebooks = np.load("codebooks_64.npy", allow_pickle=True).item()

commands = [
    "start", "stop", "lift", "pause", "next",
    "listen", "look", "continue", "reboot", "finish"
]

def confusion_matrix(dataset_path="dataset_test"):
    n = len(commands)
    matrix = np.zeros((n, n), dtype=int)

    for i, true_cmd in enumerate(commands):
        files = glob(f"{dataset_path}/{true_cmd}/*.wav")

        for f in files:
            pred = recognize(f)

            j = commands.index(pred)
            matrix[i, j] += 1

    return matrix

def print_confusion_matrix(matrix):
    print("\n=== MATRIZ DE CONFUSIÓN ===\n")

    header = " " * 12 + " ".join(f"{cmd:>10}" for cmd in commands)
    print(header)

    for i, row in enumerate(matrix):
        row_str = " ".join(f"{val:>10}" for val in row)
        print(f"{commands[i]:>10} {row_str}")

def accuracy(matrix):
    correct = np.trace(matrix)
    total = np.sum(matrix)
    return correct / total

def extract_lsf(signal):
    signal = signal / np.max(np.abs(signal))
    lpcs = LPC(signal, orden=12, tamaño_ventana=320, desplazamiento=128)

    lsf_list = []
    for a in lpcs:
        lsf = lpc_to_lsf(a)
        if lsf is None:
            continue
        lsf_list.append(lsf)

    return np.array(lsf_list)


def distortion(test_lsf, codebook_lpc, n_freqs=128):
    """
    Calcula distorsión promedio usando Itakura-Saito
    """
    if len(test_lsf) == 0:
        return np.inf

    # convertir test LSF → LPC
    test_lpc = np.array([lsf_to_lpc(x) for x in test_lsf])

    # espectros
    P_test = lpc_spectrum_batch(test_lpc, n_freqs)
    P_code = lpc_spectrum_batch(codebook_lpc, n_freqs)

    # matriz de distancias
    D = itakura_saito_matrix(P_test, P_code)

    # tomar mínimo por frame y promediar
    return np.mean(np.min(D, axis=1))


def recognize(audio_path, verbose=False):
    fs, signal = wavfile.read(audio_path)
    test_lsf = extract_lsf(signal)

    results = {}

    for cmd, cb in codebooks.items():
        d = distortion(test_lsf, cb["lpc"])
        results[cmd] = d

    predicted = min(results, key=results.get)

    if verbose:
        print(f"{audio_path} → {predicted}")

    return predicted


if __name__ == "__main__":
    cm = confusion_matrix("dataset_test")
    print_confusion_matrix(cm)

    acc = accuracy(cm)
    print(f"\nAccuracy: {acc*100:.2f}%")
