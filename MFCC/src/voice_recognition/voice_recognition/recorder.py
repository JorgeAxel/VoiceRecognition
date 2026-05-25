import sounddevice as sd
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter, iirnotch
import time
from collections import deque

# ==========================================
# CONFIGURACIÓN GLOBAL ÓPTIMA
# ==========================================
FREQ = 16000        
CHUNK_SIZE = 512    
FACTOR_AMPLIFICACION = 2.0  
MICROFONO_ID = None         

commands = [
    "start", "stop", "pause", "next"
]
dataset_path = "dataset/start"  # Cambia a tu ruta deseada para guardar los comandos

# ------------------------------------------
# FILTROS DIGITALES DE AUDIO
# ------------------------------------------
def butter_highpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_audio_cleaning(data, fs=16000):
    
    #Aplica una limpieza profunda al audio grabado para eliminar el zumbido constante.
    #Garantiza una señal cristalina para reproducción o procesamiento posterior.
        
    # 1. Filtro Paso Alto: Corta ruidos sordos por debajo de 120Hz
    b_high, a_high = butter_highpass(240, fs, order=4)
    data_filtered = lfilter(b_high, a_high, data, axis=0)
    
    # 2. Filtro Paso Bajo Agresivo: Corta estática alta por encima de 3000Hz
    # (Si el zumbido persiste, puedes probar bajando este valor a 2500)
    b_low, a_low = butter_lowpass(2000, fs, order=4)
    data_filtered = lfilter(b_low, a_low, data_filtered, axis=0)
    
    # 3. Filtros Notch en cascada (Eliminan la frecuencia fundamental y sus armónicos)
    # Q=30.0 controla qué tan estrecho es el corte para no dañar tu voz
    for freq_ruido in [60.0, 120.0, 180.0]:
        b_notch, a_notch = iirnotch(freq_ruido, 30.0, fs)
        data_filtered = lfilter(b_notch, a_notch, data_filtered, axis=0)
        
    return data_filtered

def apply_highpass_filter(data, cutoff=150, fs=16000):
    b, a = butter_highpass(cutoff, fs, order=4)
    return lfilter(b, a, data, axis=0)

# ------------------------------------------
# FUNCIONES DE CAPTURA
# ------------------------------------------
def get_noise_level(duration=1.2):
    print("\nMidiendo ruido ambiente... QUÉDATE EN SILENCIO.")
    stream = sd.InputStream(device=MICROFONO_ID, samplerate=FREQ, channels=1, dtype='float32')
    with stream:
        data, _ = stream.read(int(duration * FREQ))
    
    data_filtrada = apply_highpass_filter(data, cutoff=150, fs=FREQ)
    data_amplificada = data_filtrada * FACTOR_AMPLIFICACION
    return np.sqrt(np.mean(data_amplificada**2))

def record_until_silence(threshold, max_duration=4):
    buffer = []
    silence_counter = 0
    silence_limit = int(0.5 * FREQ)  
    pre_buffer = deque(maxlen=6) 

    stream = sd.InputStream(device=MICROFONO_ID, samplerate=FREQ, channels=1, dtype='float32')
    
    with stream:
        print("\n=== ESPERANDO COMANDO DE VOZ ===")
        print("Habla ahora...")
        
        while True:
            data, _ = stream.read(CHUNK_SIZE)
            data_filtrada = apply_highpass_filter(data, cutoff=150, fs=FREQ)
            data_amplificada = data_filtrada * FACTOR_AMPLIFICACION
            volume = np.sqrt(np.mean(data_amplificada**2))
            
            pre_buffer.append(data_amplificada.copy())
            
            barras = "█" * int(volume * 120)
            print(f"\rVolumen: [{volume:.4f}] {barras:<40}", end="", flush=True)

            if volume > threshold:
                print("\n\n[¡DISPARADO!] Capturando voz...")
                buffer.extend(list(pre_buffer))
                break
        
        total_samples = len(buffer) * CHUNK_SIZE
        max_samples = FREQ * max_duration
        
        while True:
            data, _ = stream.read(CHUNK_SIZE)
            data_filtrada = apply_highpass_filter(data, cutoff=150, fs=FREQ)
            data_amplificada = data_filtrada * FACTOR_AMPLIFICACION
            
            buffer.append(data_amplificada.copy())
            total_samples += len(data_amplificada)
            
            volume = np.sqrt(np.mean(data_amplificada**2))
            
            if volume < threshold:
                silence_counter += len(data_amplificada)
            else:
                silence_counter = 0  
            
            if silence_counter >= silence_limit:
                print("[INFO] Fin del comando por silencio.")
                break
            if total_samples >= max_samples:
                print("[INFO] Tiempo límite alcanzado.")
                break

    return np.concatenate(buffer, axis=0)

def load_audio(filename, fs=16000):
    sr, data = wavfile.read(filename)
    if sr != fs:
        raise ValueError(f"Frecuencia de muestreo del archivo ({sr} Hz) no coincide con la configurada ({fs} Hz).")
    return data.astype(np.float32) / 32768.0

def recorder():
    try:
        noise_base = get_noise_level()
        dynamic_threshold = max(noise_base * 2.8, 0.025)
        print(f"-> Umbral calibrado establecido en: {dynamic_threshold:.4f}")

        with sd.InputStream(device=MICROFONO_ID, samplerate=FREQ, channels=1, dtype='float32') as purge_stream:
            purge_stream.read(int(0.3 * FREQ))
        time.sleep(0.2)

        audio_clip = record_until_silence(dynamic_threshold, max_duration=4)
        audio_limpio = apply_audio_cleaning(audio_clip, fs=FREQ)
        audio_int16 = np.clip(audio_limpio * 32767, -32768, 32767).astype(np.int16)

        return audio_int16
    
    except Exception as e:
        print(f"\n[ERROR]: {e}")
        return None
    

def save_audio_recorder():
    contador_archivos = 1
    
    while True:
        print("\n" + "="*50)
        
        #seleccion = input("Presiona [ENTER] para grabar una nueva palabra o [q + ENTER] para salir: ").strip().lower()
        
        #if seleccion == 'q':
            #print("\nFinalizando el programa de grabación. ¡Adiós!")
            #break

        audio_int16 = recorder()
        
        # Guardado incremental en formato físico int16
        nombre_archivo = f"{dataset_path}/comando_{contador_archivos}.wav"
        wavfile.write(nombre_archivo, FREQ, audio_int16)
        
        sd.play(audio_int16, FREQ)
        sd.wait()

        print(f"[ÉXITO] Archivo '{nombre_archivo}' guardado y restaurado sin zumbidos.")
        contador_archivos += 1

"""
# ==========================================
# BUCLE PRINCIPAL DE INTERACCIÓN
# ==========================================
if __name__ == "__main__":
    print("=== PROGRAMA DE GRABACIÓN DE COMANDOS DE VOZ ===")
    print("Instrucciones:")
    print("- Presiona [ENTER] para grabar un nuevo comando.")
    print("- Habla claramente después de la señal de inicio.")
    print("- El programa detectará automáticamente el final del comando por silencio.")
    print("- Para salir, presiona [q + ENTER].")
    
    save_audio_recorder()
"""