import os
import time
import sounddevice as sd
import numpy as np
from scipy.io import wavfile
import glob
from procesamiento_voz import zcr_energy, voice_detection, find_limits, frames_to_samples
from LBG import extraer_lsf_señal, distancia_itakura_saito_lsf
from recorder import apply_audio_cleaning, get_noise_level, record_until_silence
CODEBOOK_SIZE = 64 # Cambiar según el tamaño usado en entrenamiento
REG_IS = 1e-8
MICROFONO_ID = None         

# Start, Pause, Next, Stop
commands = ["start", "pause", "next", "stop"]
codebook_path = "test_codebooks"

def cargar_codebooks(palabras, tamaño, ruta_codebook):
    """
    Carga todos los codebooks entrenados desde los archivos .npy.
    """
    codebooks = {}
    for palabra in palabras:
        nombre = f"codebook_{palabra}_{tamaño}.npy"
        ruta   = os.path.join(ruta_codebook, nombre)
        if os.path.exists(ruta):
            codebooks[palabra] = np.load(ruta)
            print(f"  Codebook cargado: {ruta}  shape: {codebooks[palabra].shape}")
        else:
            print(f"  [ERROR] No encontrado: {ruta}")
    return codebooks

def distancia_vq(vectores_prueba, codebook):
    """
    Cálculo de la distancia de cuantización VQ entre una secuencia de vectores
    y un codebook.
    """
    total_distancia = 0.0
    num_vectores = 0

    for vec in vectores_prueba:
        # Saltar vectores nulos (tramas de silencio)
        if np.all(vec == 0):
            continue

        # Búsqueda de la distancia mínima a cualquier codevector del codebook
        dist_min = np.inf
        for codevec in codebook:
            d = distancia_itakura_saito_lsf(vec, codevec)
            if d < dist_min:
                dist_min = d

        total_distancia += dist_min
        num_vectores += 1

    if num_vectores == 0:
        return np.inf   # Señal vacía -> distancia infinita

    return total_distancia / num_vectores

def reconocer_palabra(fs, señal, codebooks):
    """
    Reconocimiento de la palabra en un archivo .wav usando los codebooks entrenados.
    """

    if señal.ndim > 1:
        señal = señal.mean(axis=1)
    señal = señal.astype(np.float32)

    # NORMALIZACIÓN
    max_val = np.max(np.abs(señal))
    if max_val > 0:
        señal = señal / max_val

    # ZCR + energía
    zcr, energy = zcr_energy(señal)

    # Voice detection
    voice_mask = voice_detection(zcr, energy)

    start_frame, end_frame = find_limits(voice_mask, total_frames=len(zcr))

    if start_frame is None:
        return None, {}

    start_sample, end_sample = frames_to_samples(
        start_frame, end_frame, len_signal=len(señal)
    )

    señal = señal[start_sample:end_sample]

    # Extracción de LSF
    vectores_lsf = extraer_lsf_señal(señal, fs)

    # Cálculo de distancia VQ a cada codebook
    distancias = {}
    for palabra, codebook in codebooks.items():
        distancias[palabra] = distancia_vq(vectores_lsf, codebook)

    # La palabra con distancia mínima es la reconocida
    palabra_reconocida = min(distancias, key=distancias.get)

    return palabra_reconocida, distancias

def main():
    # Cargar codebooks entrenados
    codebooks = cargar_codebooks(commands, tamaño=CODEBOOK_SIZE, ruta_codebook=codebook_path)
    if not codebooks:
        print("[ERROR] No se encontraron codebooks.")
        raise SystemExit("Stopping here")
    
    # =========================
    # CONFIGURACIÓN
    # =========================
    freq = 16000  # Frecuencia de muestreo
    margin = int(0.2 * freq)  # 200 ms
    target_length = freq * 2  # 2 segundos (para ML)

    # medir ruido REAL antes de hablar
    noise_level = get_noise_level()
    print(f"Nivel de ruido medido: {noise_level:.4f}")
    dynamic_threshold = max(noise_level * 2.8, 0.025)
    print(f"-> Umbral calibrado establecido en: {dynamic_threshold:.4f}")

    while True:
        print("\n=== Sistema de reconocimiento de voz ===")
        print("Comandos disponibles: start, pause, next, stop")
        #input("\nPresiona ENTER para comenzar a grabar...")

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
            return

        # =========================
        # TRIM
        # =========================
        #trimmed_audio = trim_silence(audio, dynamic_threshold, margin)
        trimmed_audio = audio  # Para pruebas sin trim, comentar la línea anterior y descomentar esta

        # Protección contra audio vacío después del trim
        if trimmed_audio is None or len(trimmed_audio) == 0:
            print("[ERROR] El audio quedó vacío después de eliminar silencios.")
            return

        # Verificar energía mínima
        if np.max(np.abs(trimmed_audio)) < 1e-5:
            print("[ERROR] Audio demasiado débil.")
            return

        # =========================
        # NORMALIZACIÓN
        # =========================
        if np.max(np.abs(trimmed_audio)) > 0:
            trimmed_audio = trimmed_audio / np.max(np.abs(trimmed_audio))

        # =========================
        # LONGITUD FIJA (ML)
        # =========================
        #if len(trimmed_audio) < target_length:
            #padding = np.zeros(target_length - len(trimmed_audio))
            #trimmed_audio = np.concatenate([trimmed_audio, padding])
        #else:
            #trimmed_audio = trimmed_audio[:target_length]

        sd.play(trimmed_audio, freq)
        sd.wait()

        palabra_pred, distancias = reconocer_palabra(freq, trimmed_audio, codebooks)

        print("\nDistancias VQ:")
        for palabra, d in distancias.items():
            print(f"{palabra:>6}: {d:.4f}")

        if palabra_pred is not None:
            print(f"\nPalabra reconocida: {palabra_pred}")
        else:
            print("\nNo se pudo reconocer la palabra.")
        
        #quit = input("\n¿Deseas probar otra palabra? (s/n): ")
        #if quit.lower() != 's':
        #    print("Saliendo del sistema de reconocimiento.")
        #    break

if __name__ == "__main__":
    main()
