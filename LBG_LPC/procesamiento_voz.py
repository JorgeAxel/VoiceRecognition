import numpy as np
from scipy.io import wavfile
from glob import glob
from LBG import extraer_lsf_señal, lbg_algorithm

commands = [
    "start", "stop", "pause", "next"
]
#users = ["axel", "daniel", "joel", "oscar"]
users = ["axel", "joel"]
dataset_path = "dataset"
codebook_path = "test_codebooks"

FRAME_SIZE = 320
HOP_SIZE = 128
ORDER = 12
ZCR_FRACTION = 0.08
ENERGY_FRACTION = 0.03
LIMITS_MARGIN = 3
CODEBOOK_SIZE = 64 # Cambiar para el tamaño deseado del codebook (16, 32, 64)

def zcr_energy(signal, frame_size=FRAME_SIZE, hop_size=HOP_SIZE):
    num_frames = 1 + (len(signal) - frame_size) // hop_size
    zcr = np.zeros(num_frames)
    energy = np.zeros(num_frames)

    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size
        frame = signal[start:end]

        zcr[i] = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * frame_size)
        energy[i] = np.sum(frame ** 2) / frame_size

    return zcr, energy

def voice_detection(zcr, energy, zcr_fraction=ZCR_FRACTION, energy_fraction=ENERGY_FRACTION):
    zcr_threshold = np.max(zcr) * zcr_fraction
    energy_threshold = np.max(energy) * energy_fraction
    voice_mask = (zcr > zcr_threshold) & (energy > energy_threshold)
    return voice_mask

def find_limits(voice_mask, margin=LIMITS_MARGIN, total_frames=None):
    indices = np.where(voice_mask)[0]
    if len(indices) == 0:
        return None, None # No se detectó voz

    start = max(0, indices[0] - margin)
    end = indices[-1] + margin
    if total_frames is not None:
        end = min(end, total_frames - 1)
    return start, end

def frames_to_samples(first_frame, last_frame, frame_size=FRAME_SIZE, hop_size=HOP_SIZE, len_signal=None):
    start_sample = first_frame * hop_size
    end_sample = last_frame * hop_size + frame_size
    if len_signal is not None:
        end_sample = min(end_sample, len_signal)
    return start_sample, end_sample

def audio_processing(folder_path):
    audio_files = glob(folder_path + "/*.wav")
    matriz_lsf = []

    for audio in sorted(audio_files):
        fs, signal = wavfile.read(audio)
        if signal.ndim > 1:
            signal = signal.mean(axis=1)  # Convertir a mono si es estéreo
        signal = signal.astype(np.float32)

        max_val = np.max(np.abs(signal))
        if max_val > 0:
            signal = signal / max_val  # Normalizar a [-1, 1]

        zcr, energy = zcr_energy(signal)
        voice_mask = voice_detection(zcr, energy)
        start_frame, end_frame = find_limits(voice_mask, total_frames=len(zcr))

        if start_frame is None or end_frame is None:
            print(f"No se detectó voz en el archivo: {audio}")
            continue
        
        start_sample, end_sample = frames_to_samples(start_frame, end_frame, len_signal=len(signal))
        trim_signal = signal[start_sample:end_sample]

        if len(trim_signal) == 0 or np.max(np.abs(trim_signal)) == 0:
            print(f"Sección de voz vacía o silenciosa en el archivo: {audio}")
            continue

        vector_lsf = extraer_lsf_señal(trim_signal, fs)
        validate_lsf = np.any(vector_lsf != 0, axis=1)
        if validate_lsf.any():
            matriz_lsf.append(vector_lsf)
    if not matriz_lsf:
        raise ValueError(f"No se extrajeron LSF válidos de los archivos en: {folder_path}")

    return np.vstack(matriz_lsf)

def main():

    for cmd in commands:
        print(f"\n=== Procesando comando: '{cmd}' ===")

        all_lsf_vectors = []  # acumulador

        for user in users:
            print(f"Usuario: {user}")
            try:
                folder_path = f"{dataset_path}/data_{user}/train/{cmd}"
                #folder_path = f"{dataset_path}/data/train/{cmd}"
                lsf_vector = audio_processing(folder_path)

                print(f"Vectores de {user}: {lsf_vector.shape}")

                all_lsf_vectors.append(lsf_vector)

            except Exception as e:
                print(f"Error con usuario '{user}' en comando '{cmd}': {e}")

        # Combinar todos los usuarios
        if len(all_lsf_vectors) == 0:
            print(f"No hay datos para el comando '{cmd}'")
            continue

        combined_lsf = np.vstack(all_lsf_vectors)
        print(f"Dataset combinado: {combined_lsf.shape}")

        # Entrenar UN solo codebook por comando
        codebook = lbg_algorithm(combined_lsf, tamaño_codebook=CODEBOOK_SIZE)

        save = f"{codebook_path}/codebook_{cmd}_{CODEBOOK_SIZE}.npy"
        np.save(save, codebook)

        print(f"Codebook guardado: {save} shape: {codebook.shape}")

    print("\n=== Fin del procesamiento ===")

if __name__ == "__main__":
    main()
