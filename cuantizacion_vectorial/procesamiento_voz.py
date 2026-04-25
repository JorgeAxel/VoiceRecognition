import numpy as np
from scipy.io import wavfile
from glob import glob
from LBG import lbg_algorithm
from LPC import LPC

commands = [
    "start", "stop", "lift", "pause", "next",
    "listen", "look", "continue", "reboot", "finish"
]

def audio_dataset(folder_path):
    audio_files = glob(folder_path + "/*.wav")
    lpcs = []

    for audio in audio_files:
        fs, signal = wavfile.read(audio)
        signal = signal / np.max(np.abs(signal))
        lpc = LPC(signal, orden=12, tamaño_ventana=320, desplazamiento=128)
        lpcs.append(lpc)

    if len(lpcs) == 0:
        return None

    return np.vstack(lpcs)

def main():
    codebooks = {}

    for cmd in commands:
        print(f"\n=== Procesando comando: '{cmd}' ===")
        folder_path = f"dataset/{cmd}"
        lpcs = audio_dataset(folder_path)

        if lpcs is None:
            print(f"No se encontraron archivos para el comando '{cmd}'.")
            continue

        print(f"Coeficientes LPC calculados para '{cmd}': {lpcs.shape}")
        codebook_lpc, codebook_lsf = lbg_algorithm(lpcs, k=64)

        codebooks[cmd] = {
            "lpc": codebook_lpc,
            "lsf": codebook_lsf
        }
        print(f"Codebook para '{cmd}' generado con {len(codebook_lpc)} entradas.")

    print("\n=== Fin del procesamiento ===")

    np.save("codebooks.npy", codebooks)
    return codebooks

if __name__ == "__main__":
    main()