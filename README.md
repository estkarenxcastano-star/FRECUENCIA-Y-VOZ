# LABORATORIO 3 
## FRECUENCIA Y VOZ
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
### La frase que elegimos fue: "No nos queremos cambiar de grupo"

### Se importaron los archivos de voz desde Google Drive a Google Colab
```python
from google.colab import drive
drive.mount('/content/drive')

from google.colab import files
uploaded = files.upload()
```
### Se guardaron los archivos de voz en formato .wav
+ Saving Voz_Ali.wav to Voz_Ali.wav
+ Saving Voz_Karen.wav to Voz_Karen.wav
+ Saving Voz_Kevin.wav to Voz_Kevin.wav
+ Saving Voz_Mafe.wav to Voz_Mafe.wav
+ Saving Voz_Mateus.wav to Voz_Mateus.wav
+ Saving Voz_Raul.wav to Voz_Raul.wav

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
+ Guardado Voz_Ali.dat / Voz_Ali.hea (Fs=48000 Hz)
+ Guardado Voz_Karen.dat / Voz_Karen.hea (Fs=48000 Hz)
+ Guardado Voz_Kevin.dat / Voz_Kevin.hea (Fs=48000 Hz)
+ Guardado Voz_Mafe.dat / Voz_Mafe.hea (Fs=48000 Hz)
+ Guardado Voz_Mateus.dat / Voz_Mateus.hea (Fs=48000 Hz)
+ Guardado Voz_Raul.dat / Voz_Raul.hea (Fs=48000 Hz)

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

### Graficamos el espectro de magnitudes frecuenciales de cada voz
```python
# Recorremos los registros WFDB que ya creados
for rec_name, _ in record_names:
    # 1) Leer la señal desde WFDB
    rec = wfdb.rdrecord(rec_name)               }
    if rec.p_signal is not None:
        y = rec.p_signal[:, 0].astype(np.float32)
        sr = rec.fs
    else:
        rec = wfdb.rdrecord(rec_name, physical=False)  # fuerza d_signal
        y_d = rec.d_signal[:, 0].astype(np.float32)
        gain = float(rec.adc_gain[0]) if rec.adc_gain is not None else 32767.0
        baseline = float(rec.baseline[0]) if rec.baseline is not None else 0.0
        y = (y_d - baseline) / gain
        sr = rec.fs

    # 2) Preparar para la FFT
    y = y - np.mean(y)
    N = len(y)
    y_win = y * np.hanning(N)

    # 3) FFT de una sola cara y eje de frecuencias
    n_fft = 1 << int(np.ceil(np.log2(N)))
    n_fft = min(max(2048, n_fft), 65536)
    Y = np.fft.rfft(y_win, n=n_fft)
    f = np.fft.rfftfreq(n_fft, d=1/sr)
    mag_db = 20*np.log10(np.abs(Y) + 1e-12)

    # 4) marcar el pico principal en 50–5000 Hz
    band = (f >= 50) & (f <= 5000)
    f_peak = None
    if np.any(band):
        i = np.argmax(mag_db[band])
        f_peak = f[band][i]
        pk = mag_db[band][i]

    # 5) Gráfico del espectro
    plt.figure(figsize=(10,3))
    plt.plot(f, mag_db)
    if f_peak is not None:
        plt.axvline(f_peak, linestyle='--', alpha=0.6)
        plt.text(f_peak, pk, f'{f_peak:.1f} Hz', ha='left', va='bottom')
    plt.xlim(0, sr/2)
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Magnitud [dB]")
    plt.title(f"{rec_name} — Espectro")
    plt.tight_layout()
    plt.show()
```
+ **Voz Alissia**
<img width="887" height="260" alt="image" src="https://github.com/user-attachments/assets/4f41af13-404c-4014-8e1c-0e87ffe26a0d" />

+ **Voz Karen**
<img width="888" height="263" alt="image" src="https://github.com/user-attachments/assets/ec44298a-a260-46ad-b919-3b30c75a3b7a" />

+ **Voz Mafe**
<img width="890" height="261" alt="image" src="https://github.com/user-attachments/assets/ad446123-5c78-4601-bdd7-d4a75c623fa3" />

+ **Voz Kevin**
<img width="887" height="265" alt="image" src="https://github.com/user-attachments/assets/000fc88f-1567-45f5-8d3c-6cc5d7440b0b" />

+ **Voz Mateus**
<img width="888" height="263" alt="image" src="https://github.com/user-attachments/assets/4b9dd698-2e44-42bd-a610-8ab98c51eb26" />

+ **Voz Raúl**
<img width="892" height="258" alt="image" src="https://github.com/user-attachments/assets/594cdb35-8d6b-4419-a18e-fb9493996c99" />

### Obtenemos las siguientes características de cada señal
+ Frecuencia fundamental. 
+ Frecuencia media. 
+ Brillo. 
+ Intensidad (energía). 

```python
#  Define una constante muy pequeña llamada EPS
EPS = 1e-12

# 1) Leer señal WFDB como float + Fs
def read_wfdb_float(rec_name):
    rec = wfdb.rdrecord(rec_name)
    if rec.p_signal is not None:
        return rec.p_signal[:,0].astype(np.float32), rec.fs
    rec = wfdb.rdrecord(rec_name, physical=False)
    y = rec.d_signal[:,0].astype(np.float32)
    gain = float(rec.adc_gain[0]) if rec.adc_gain is not None else 32767.0
    base = float(rec.baseline[0]) if rec.baseline is not None else 0.0
    return (y - base) / gain, rec.fs

# 2) F0 por autocorrelación (simple)
def f0_autocorr(y, sr, fmin=50, fmax=500):
    y = y - np.mean(y)
    ac = np.correlate(y, y, mode='full')[len(y)-1:]
    ac = ac / (ac[0] + EPS)
    lag_min = int(sr / fmax); lag_max = int(sr / fmin)
    if lag_max <= lag_min or lag_max >= len(ac):
        return np.nan
    k = lag_min + np.argmax(ac[lag_min:lag_max+1])
    return sr / k if ac[k] > 0.2 else np.nan

# 3) Frecuencia_media y brillo (>1500 Hz)
def FM_y_brillo(y, sr, cutoff=1500):
    N = len(y)
    y = y - np.mean(y)
    Y = np.fft.rfft(y * np.hanning(N))
    f = np.fft.rfftfreq(N, d=1/sr)
    mag = np.abs(Y); pow_spec = mag**2
    FM = np.sum(f * mag) / (np.sum(mag) + EPS)
    brillo = np.sum(pow_spec[f >= cutoff]) / (np.sum(pow_spec) + EPS)
    return float(FM), float(brillo)

# 4) Intensidad (RMS y dBFS)
def rms_y_dbfs(y):
    rms = float(np.sqrt(np.mean(y**2)))
    dbfs = float(20 * np.log10(rms + EPS))
    return rms, dbfs

# 5) Recorre tus registros ya existentes
import pandas as pd
rows = []
for rec_name, _ in record_names:   # record_names ya debe existir
    y, sr = read_wfdb_float(rec_name)
    F0 = f0_autocorr(y, sr)
    C, B = FM_y_brillo(y, sr)
    RMS, DB = rms_y_dbfs(y)
    rows.append({
        "archivo": rec_name,
        "Fs_Hz": sr,
        "dur_s": round(len(y)/sr, 3),
        "F0_Hz": None if np.isnan(F0) else round(F0, 2),
        "Frecuencia_media_Hz": round(C, 2),
        "brillo_ratio(>1500Hz)": round(B, 4),
        "RMS": round(RMS, 6),
        "RMS_dBFS": round(DB, 2),
    })

df_punto5 = pd.DataFrame(rows)
from IPython.display import display
display(df_punto5)
df_punto5.to_csv("punto5_resultados.csv", index=False)
print("Guardado: punto5_resultados.csv")
```
### RESULTADOS

<img width="703" height="199" alt="image" src="https://github.com/user-attachments/assets/63553cc9-5ddb-48e6-b3d1-e964fb4460a2" />

# PARTE B - MEDICIÓN DE JITTER Y SHIMMER

+ **FILTRO PASA-BANDA**
```python
# ================== PARTE B · Paso 1: Filtro pasa-banda (con tu mapeo) ==================
import wfdb, numpy as np, matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# --- 1) Leer WFDB y devolver señal en float + Fs ---
def read_wfdb_float(rec_name):
    rec = wfdb.rdrecord(rec_name)                      # abre <rec>.hea/.dat
    if rec.p_signal is not None:                       # señal en unidades físicas
        return rec.p_signal[:, 0].astype(np.float32), rec.fs
    # si no existe p_signal, usamos d_signal y la convertimos con la cabecera
    rec = wfdb.rdrecord(rec_name, physical=False)
    y = rec.d_signal[:, 0].astype(np.float32)
    gain = float(rec.adc_gain[0]) if rec.adc_gain is not None else 32767.0
    base = float(rec.baseline[0]) if rec.baseline is not None else 0.0
    return (y - base) / gain, rec.fs

# --- 2) Filtro pasa-banda Butterworth (orden 4) con fase cero ---
def bandpass(y, sr, f1, f2, order=4):
    nyq = 0.5 * sr
    b, a = butter(order, [f1/nyq, f2/nyq], btype="bandpass")
    return filtfilt(b, a, y)   # filtra ida/vuelta → sin desfase

# --- 3) Tu mapeo de sexo por archivo (usamos claves en minúscula por seguridad) ---
sexo_map = {
    "voz_ali":   "mujer",
    "voz_karen": "mujer",
    "voz_mafe":  "mujer",
    "voz_kevin": "hombre",
    "voz_mateus":"hombre",
    "voz_raul":  "hombre",
}

# --- 4) Helper para graficar antes vs después ---
def plot_time_before_after(y, y_filt, sr, title):
    t = np.arange(len(y)) / sr
    plt.figure(figsize=(10,4))
    plt.plot(t, y, label="Original", alpha=0.6)
    plt.plot(t, y_filt, label="Filtrada", linewidth=1)
    plt.xlabel("Tiempo [s]"); plt.ylabel("Amplitud")
    plt.title(title); plt.legend(); plt.tight_layout(); plt.show()

# --- 5) Aplicamos el filtro a todos y guardamos para el paso de jitter ---
filtered_signals = {}  # aquí guardamos para reutilizar en la siguiente celda

for rec_name, _ in record_names:
    y, sr = read_wfdb_float(rec_name)

    # sexo por tu mapeo (insensible a mayúsculas/minúsculas)
    sexo = sexo_map.get(rec_name.lower(), None)
    if sexo is None:
        raise ValueError(
            f"'{rec_name}'. "
        )

    # bandas exigidas por la guía
    f1, f2 = (80, 400) if sexo == "hombre" else (150, 500)
    y_filt = bandpass(y, sr, f1, f2, order=4)

    print(f"{rec_name}: sexo={sexo}, Fs={sr} Hz → filtro {f1}-{f2} Hz (orden 4)")
    plot_time_before_after(y, y_filt, sr, f"{rec_name} — Antes/Después del filtro")

    # guardamos para el paso de jitter/shimmer
    filtered_signals[rec_name] = {"y": y_filt.astype(np.float32), "sr": sr, "sexo": sexo}
```
+ **Voz Alissia: sexo=mujer, Fs=48000 Hz → filtro 150-500 Hz (orden 4)**
<img width="888" height="347" alt="image" src="https://github.com/user-attachments/assets/3b06d383-a12b-48d3-87e2-fb321ee662fe" />

+ **Voz Karen: sexo=mujer, Fs=48000 Hz → filtro 150-500 Hz (orden 4)**
<img width="887" height="346" alt="image" src="https://github.com/user-attachments/assets/0ceb8bf5-2615-424d-9bc5-558d43fb3dc5" />

+ **Voz Mafe: sexo=mujer, Fs=48000 Hz → filtro 150-500 Hz (orden 4)**
<img width="889" height="348" alt="image" src="https://github.com/user-attachments/assets/0174df8a-377d-4eb3-a88d-8be077c83d82" />

+ **Voz Kevin: sexo=hombre, Fs=48000 Hz → filtro 80-400 Hz (orden 4)**
<img width="889" height="350" alt="image" src="https://github.com/user-attachments/assets/0a79b440-6c43-4dac-8b83-d1ade7b36c14" />

+ **Voz Mateus: sexo=hombre, Fs=48000 Hz → filtro 80-400 Hz (orden 4)**
<img width="887" height="350" alt="image" src="https://github.com/user-attachments/assets/5e076bbb-483f-49c9-8d92-81bb85e0a6d2" />

+ **Voz Raúl: sexo=hombre, Fs=48000 Hz → filtro 80-400 Hz (orden 4)**
<img width="888" height="348" alt="image" src="https://github.com/user-attachments/assets/316232ed-86d4-4a37-983b-ef516f6c5171" />

























