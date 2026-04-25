import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import time

duration = 4
device_info = sd.query_devices(None, 'input')
freq = 16000
#freq = int(device_info['default_samplerate'])

samples_per_command = 5  # audios por comando

threshold = 0.01          # umbral para detectar voz
margin = int(0.2 * freq) # 200 ms de margen

commands = [
    "start", "stop", "lift", "pause", "next",
    "listen", "look", "continue", "reboot", "finish"
]

def trim_silence(audio, threshold=0.01, margin=0):
    audio = audio.flatten()

    mask = np.abs(audio) > threshold
    if not np.any(mask):
        return audio

    start = np.argmax(mask)
    end = len(mask) - np.argmax(mask[::-1])

    # Añadir margen
    start = max(0, start - margin)
    end = min(len(audio), end + margin)

    return audio[start:end]

for cmd in commands:
    print(f"\n=== Di la palabra: '{cmd}' ===")

    for i in range(samples_per_command):
        print(f"Prepárate... muestra {i}")
        time.sleep(1)

        print("¡Habla!")
        audio = sd.rec(int(duration * freq), samplerate=freq, channels=1)
        sd.wait()

        # =========================
        # RECORTE DE SILENCIO
        # =========================
        trimmed_audio = trim_silence(audio, threshold, margin)

        # =========================
        # NORMALIZACIÓN (opcional pero recomendable)
        # =========================
        if np.max(np.abs(trimmed_audio)) > 0:
            trimmed_audio = trimmed_audio / np.max(np.abs(trimmed_audio))

        # =========================
        # GUARDAR
        # =========================
        filename = f"dataset_test/{cmd}/{cmd}_{i}.wav"
        write(filename, freq, (trimmed_audio * 32767).astype(np.int16))

        print(f"Guardado: {filename}")
        print(f"Duración final: {len(trimmed_audio)/freq:.2f} s")

        time.sleep(0.5)

print("\n Dataset completado")