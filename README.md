# LABORATORIO 3 
## FRECUENCIA-Y-VOZ
### Objetivo
Capturar y procesar señales de voz masculinas y femeninas para analizar su comportamiento espectral mediante la Transformada de Fourier, extrayendo parámetros característicos como frecuencia fundamental, frecuencia media, brillo, intensidad, jitter y shimmer, con el fin de comparar y concluir las diferencias principales entre ambos géneros.

# PARTE A-ADQUISICIÓN DE LAS SEÑALES DE VOZ


## LIBRERIAS
Las librerias que implementamos fueron las siguientes:

+ **Importación de liberías**
```python
! pip install wfdb
import wfdb
import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import pandas as pd
import find_peaks
```
### Se importaron los archivos de voz desde Google Drive a Google Colab
```python
from google.colab import drive
drive.mount('/content/drive')

from google.colab import files
uploaded = files.upload()
```
### Se guardaron los archivos de voz en formato .wav
Saving Voz_Ali.wav to Voz_Ali.wav
Saving Voz_Karen.wav to Voz_Karen.wav
Saving Voz_Kevin.wav to Voz_Kevin.wav
Saving Voz_Mafe.wav to Voz_Mafe.wav
Saving Voz_Mateus.wav to Voz_Mateus.wav
Saving Voz_Raul.wav to Voz_Raul.wav

### Se convertirtieron archivos de audio .wav en registros WFDB y se muestra la frecuencia de muestreo original del audio
```python
! pip install wfdb
import wfdb
import librosa
import numpy as np
import os

def wav_to_wfdb(path):
    base = os.path.splitext(os.path.basename(path))[0]   # nombre base sin extensión
    y, sr = librosa.load(path, sr=None, mono=True)       # carga el audio en mono, conserva Fs
    y = y / (np.max(np.abs(y)) + 1e-12)                  # normaliza a [-1, 1] por seguridad
    sig = (y * 32767).astype(np.int16).reshape(-1, 1)    # a enteros 16-bit para WFDB

    # Guardar como registro WFDB:
    wfdb.wrsamp(
        record_name=base,        # nombre del registro (crea base.dat y base.hea)
        fs=sr,                   # frecuencia de muestreo
        units=['adu'],           # unidades (digital units)
        sig_name=['voice'],      # nombre del canal
        d_signal=sig,            # señal entera
        fmt=['16'],              # formato 16-bit
        adc_gain=[32767.0],
        baseline=[0]             # nivel de referencia (offset DC) en cuentas
    )
    print(f" Guardado {base}.dat / {base}.hea (Fs={sr} Hz)")
    return base, sr

# Convertimos todos los .wav que subimos
record_names = []
for file in uploaded.keys():
    rec_name, sr = wav_to_wfdb(file)
    record_names.append((rec_name, sr))
```
Guardado Voz_Ali.dat / Voz_Ali.hea (Fs=48000 Hz)
Guardado Voz_Karen.dat / Voz_Karen.hea (Fs=48000 Hz)
Guardado Voz_Kevin.dat / Voz_Kevin.hea (Fs=48000 Hz)
Guardado Voz_Mafe.dat / Voz_Mafe.hea (Fs=48000 Hz)
Guardado Voz_Mateus.dat / Voz_Mateus.hea (Fs=48000 Hz)
Guardado Voz_Raul.dat / Voz_Raul.hea (Fs=48000 Hz)

### Gráficas de las señales de voz en el dominio del tiempo
```python
import matplotlib.pyplot as plt

def plot_wfdb_time(rec_name, sr):
    rec = wfdb.rdrecord(rec_name)
    y = rec.p_signal[:, 0].astype(np.float32)
    t = np.arange(len(y)) / sr

    plt.figure(figsize=(10,3))
    plt.plot(t, y)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud normalizada")
    plt.title(f"{rec_name} — dominio del tiempo")
    plt.tight_layout()
    plt.show()

for rec_name, sr in record_names:
    plot_wfdb_time(rec_name, sr)
```
+ **Voz Alissia**
<img width="887" height="261" alt="image" src="https://github.com/user-attachments/assets/6e890a0d-3e1f-41d8-af49-088e9405b12a" />

+ **Voz Karen**
<img width="879" height="261" alt="image" src="https://github.com/user-attachments/assets/a2e57c99-1523-4b92-9674-8a5de04e62a3" />

+ **Voz Mafe**
<img width="889" height="261" alt="image" src="https://github.com/user-attachments/assets/e4b6de32-d0be-4bc7-93e1-0a4aa04bd723" />

+ **Voz Kevin**
<img width="879" height="265" alt="image" src="https://github.com/user-attachments/assets/8acb63a9-ca12-4c1f-a035-b4b7a21fe9bf" />

+ **Voz Mateus**
<img width="881" height="261" alt="image" src="https://github.com/user-attachments/assets/502a78d0-2a31-439c-be76-e7a1e2bca425" />

+ **Voz Raúl**
<img width="892" height="260" alt="image" src="https://github.com/user-attachments/assets/684f52c4-34aa-4fae-b214-c2da102b8869" />










