# LABORATORIO 3 
## FRECUENCIA-Y-VOZ
### Objetivo
Capturar y procesar señales de voz masculinas y femeninas para analizar su comportamiento espectral mediante la Transformada de Fourier, extrayendo parámetros característicos como frecuencia fundamental, frecuencia media, brillo, intensidad, jitter y shimmer, con el fin de comparar y concluir las diferencias principales entre ambos géneros.

#PARTE A-ADQUISICIÓN DE LAS SEÑALES DE VOZ


##LIBRERIAS
Las librerias que implementamos fueron las siguientes:

+**Importación de liberías**
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



