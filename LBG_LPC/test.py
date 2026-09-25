import os
import numpy as np
from scipy.io import wavfile
import glob
from LBG import extraer_lsf_señal, distancia_itakura_saito_lsf
from procesamiento_voz import find_limits, frames_to_samples, voice_detection, zcr_energy
CODEBOOK_SIZE = 64 # Cambiar según el tamaño usado en entrenamiento
REG_IS = 1e-8
commands = [
    "stop", "pause", "next", "start"
]
#users = ["axel", "daniel", "joel", "oscar"]
users = ["axel", "joel"]
#dataset_path_test = "dataset_equipo4"
dataset_path_test = "dataset"
codebook_path = "test_codebooks"

def distancia_vq(vectores_prueba, codebook):
    """
    Cálculo de la distancia de cuantización VQ entre una secuencia de vectores
    y un codebook.

    Para cada vector de prueba, busca el codevector más cercano (vecino más
    próximo en distancia IS) y acumula esas distancias mínimas.

    La distancia promedio normalizada es la métrica de similitud final:
        d_VQ = (1/T) * sum_t min_c d_IS(x_t, c)

    Un valor menor indica mayor similitud con ese codebook.
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

def reconocer_palabra(ruta_wav, codebooks):
    """
    Reconocimiento de la palabra en un archivo .wav usando los codebooks entrenados.

    Proceso:
    1. Carga y recorte de la señal
    2. Extracción de vectores LSF
    3. Cálculo de distancia VQ a cada codebook
    4. La palabra reconocida es la de menor distancia

    Retorna:
        palabra_reconocida: string
        distancias:         diccionario {palabra: distancia}
    """

    fs, señal = wavfile.read(ruta_wav)
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
    # min(dict, key=dict.get) busca la clave con el valor mínimo
    palabra_reconocida = min(distancias, key=distancias.get)

    return palabra_reconocida, distancias

def cargar_codebooks(palabras, tamaño, ruta_codebook):
    """
    Carga todos los codebooks entrenados desde los archivos .npy.
    Retorna un diccionario: {palabra: arreglo_codebook}
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

def calcular_matriz_confusion(palabras, codebooks):
    """
    Evaluación del sistema sobre todos los archivos de prueba y contrucción de
    la matriz de confusión.

    La matriz de confusión M es de tamaño (n_palabras × n_palabras):
        M[i, j] = número de veces que la palabra real i fue reconocida como j

    La diagonal contiene los aciertos.
    Los elementos fuera de la diagonal son errores.
    """
    
    n = len(palabras)
    # Mapeo de palabra a índice numérico
    idx_palabra = {p: i for i, p in enumerate(palabras)}

    # Inicialización de matriz de confusión con ceros
    # shape: (n_palabras, n_palabras)
    matriz = np.zeros((n, n), dtype=int)

    total = 0
    correctos = 0

    print("\n  Evaluando archivos de prueba:")
    print(f"  {'Archivo':40s}  {'Real':10s}  {'Reconocida':10s}  {'OK?':4s}")
    print("  " + "-"*70)

    for i, palabra_real in enumerate(palabras):

        for user in users:
            carpeta = f"{dataset_path_test}/data_{user}/test/{palabra_real}"
            #carpeta = f"{dataset_path_test}/data/test/{palabra_real}"

            if not os.path.exists(carpeta):
                print(f"  [WARNING] Carpeta no encontrada: {carpeta}")
                continue

            archivos = sorted(glob.glob(os.path.join(carpeta, "*.wav")))

            for ruta_wav in archivos:
                nombre = f"{user}_{os.path.basename(ruta_wav)}"

                palabra_pred, _ = reconocer_palabra(ruta_wav, codebooks)

                j = idx_palabra.get(palabra_pred, -1)
                if j >= 0:
                    matriz[i, j] += 1

                ok = "✓" if palabra_pred == palabra_real else "✗"

                if palabra_pred == palabra_real:
                    correctos += 1
                total += 1

                print(f"  {nombre:40s}  {palabra_real:10s}  {palabra_pred:10s}  {ok}")

    return matriz, correctos, total

def imprimir_matriz(matriz, palabras):
    """
    Visualización de la matriz de confusión en formato texto alineado.
    """
    
    print("\n  MATRIZ DE CONFUSIÓN")
    print("  (filas = palabra real | columnas = palabra reconocida)\n")

    # Encabezado de columnas
    ancho = 8
    print("  " + " " * 12, end="")
    for p in palabras:
        print(f"{p[:ancho]:^{ancho}}", end=" ")
    print()

    # Separador
    print("  " + "-" * (12 + (ancho + 1) * len(palabras)))

    # Filas
    for i, p_real in enumerate(palabras):
        print(f"  {p_real[:12]:12s}", end="")
        for j in range(len(palabras)):
            val = matriz[i, j]
            # Resaltar la diagonal con color ANSI si la terminal lo soporta
            if i == j:
                print(f"\033[92m{val:^{ancho}d}\033[0m", end=" ")   # Verde
            else:
                print(f"{val:^{ancho}d}", end=" ")
        print()

def calcular_metricas(matriz, palabras):
    """
    Cálculo de métricas de desempeño por clase:
    - Precisión por clase: aciertos_clase_i / total_predichos_como_i
    - Recall por clase:    aciertos_clase_i / total_muestras_reales_i
    - Precisión global:    total_aciertos / total_muestras
    """
    print("\n  MÉTRICAS POR CLASE:")
    print(f"  {'Palabra':12s}  {'Recall':8s}  {'Precisión':10s}")
    print("  " + "-"*35)

    for i, palabra in enumerate(palabras):
        total_real = matriz[i, :].sum()
        total_pred = matriz[:, i].sum()
        aciertos   = matriz[i, i]

        recall     = aciertos / total_real  if total_real > 0 else 0
        precision  = aciertos / total_pred  if total_pred > 0 else 0

        print(f"  {palabra:12s}  {recall:7.1%}  {precision:9.1%}")

def main():
    # Cargar codebooks entrenados
    codebooks = cargar_codebooks(commands, tamaño=CODEBOOK_SIZE, ruta_codebook=codebook_path)
    if not codebooks:
        print("[ERROR] No se encontraron codebooks.")
        raise SystemExit("Stopping here")

    # Evaluar el sistema y construir la matriz de confusión
    matriz_confusion, correctos, total = calcular_matriz_confusion(commands, codebooks)

    # Visualización de resultados 
    precision_global = 100 * correctos / total if total > 0 else 0

    print(f"\n  ── Resultados ──")
    print(f"  Total evaluado: {total} archivos")
    print(f"  Correctos:      {correctos}")
    print(f"  Precisión:      {precision_global:.1f}%")

    # Imprimir resultados
    imprimir_matriz(matriz_confusion, commands)
    calcular_metricas(matriz_confusion, commands)

if __name__ == "__main__":
    main()
