import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import time
import os
from recorder import apply_audio_cleaning, record_until_silence, get_noise_level

freq = 16000
samples_per_command = 5
MICROFONO_ID = None

commands = ["stop", "pause", "next", "start"]
dataset_path = "dataset/data_user"
train_test = "test" # Change accordingly

# =========================
# CONFIGURACIÓN
# =========================
margin = int(0.2 * freq)  # 200 ms
target_length = freq * 2  # 2 segundos (para ML)

# =========================
# LOOP PRINCIPAL
# =========================
for cmd in commands:
    print(f"\n=== Di la palabra: '{cmd}' ===")

    # Crear carpeta si no existe
    os.makedirs(f"{dataset_path}/{train_test}/{cmd}", exist_ok=True)

    # medir ruido REAL antes de hablar
    noise_level = get_noise_level()
    print(f"Nivel de ruido medido: {noise_level:.4f}")
    dynamic_threshold = max(noise_level * 2.8, 0.025)
    print(f"-> Umbral calibrado establecido en: {dynamic_threshold:.4f}")

    for i in range(samples_per_command):
        print(f"Prepárate... muestra {i}")

        # Purga del búfer para evitar el sonido del teclado
        with sd.InputStream(device=MICROFONO_ID, samplerate=freq, channels=1, dtype='float32') as purge_stream:
            purge_stream.read(int(0.3 * freq))
        
        time.sleep(0.2)

        audio_clip = record_until_silence(dynamic_threshold, max_duration=4)
            
        # FILTRADO ADICIONAL POST-GRABACIÓN: Eliminación del zumbido permanente
        audio_limpio = apply_audio_cleaning(audio_clip, fs=freq)
        audio = np.clip(audio_limpio * 32767, -32768, 32767).astype(np.int16)

        # Verificar si se grabó algo
        if audio is None or len(audio) == 0:
            print("[ERROR] No se capturó audio.")
            raise SystemExit("Stopping here")

        # =========================
        # NORMALIZACIÓN
        # =========================
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))

        sd.play(audio, freq)
        sd.wait()

        # =========================
        # GUARDAR
        # =========================
        filename = f"{dataset_path}/{train_test}/{cmd}/{cmd}_{i}.wav"
        write(filename, freq, (audio * 32767).astype(np.int16))

        print(f"Guardado: {filename}")
        print(f"Duración final: {len(audio)/freq:.2f} s")

        time.sleep(0.5)

print("\nDataset completado")
