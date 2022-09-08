
import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt

fs,x_1m = wav.read('med1m.wav')
fs,x_2m_ruido = wav.read('med2mruido.wav')
fs,cal = wav.read('Calibracion.wav')
cal=np.float64(cal)

x_2m_ruido=np.concatenate((x_2m_ruido,np.zeros(3)),axis=None)

# %% Conversión a presión [Pa] [dB]

#### Calibración

n_cal = len(cal)    
cal_rms = np.sqrt((1/n_cal)*np.sum((cal[0:n_cal-1])**2))

#### Medición a 1 m

x_1m = (x_1m/cal_rms)             # Señal a 1 m en Pascales
x_1m_dB = 10*np.log10((x_1m)**2/((20*(10**(-6)))**2))  # Pasaje a dB
Leq_1m=10*np.log10(np.mean(10**(x_1m_dB/10)))  # Nivel contínuo equivalente

#### Medición a 2m con ruido

x_2m_ruido = (x_2m_ruido/cal_rms)      # Señal a 2 m con ruido en Pascales
x_2m_ruido_dB = 10*np.log10((x_2m_ruido)**2/((20*(10**(-6)))**2))  # Pasaje a dB
Leq_2m_ruido=10*np.log10(np.mean(10**(x_2m_ruido_dB/10)))  # Nivel contínuo equivalente

# %% Ecuación señal temporal 

n_2m = len(x_2m_ruido)/fs
Ar = np.array([0.7, 0.4, 0.5, 0.2, 0.3])
f0 = 100

# Variación temporal de la señal original reproducida 

y = 0

for i in range(5):
    y = y + Ar[i]*np.cos(2*np.pi*(i+1)*f0*np.arange(0,n_2m,1/fs))
#%%                         Filtro Kalman

# %%PRIMER CONJUNTO DE MUESTRAS: 
    
"""
Estimación tomando como modelo matemático la divergencia de la onda esférica, 
partiendo de la medición a 1 m estimamos la medición a 2 m con una caida de 6 dB.
Expresada como la mitad de la presión eficaz al cuadrado en 1 m 
(pef(2m)^2 = [pef(1m)^2]/2)
"""
x_2m_priori = x_1m[0:50]/2

K1 = (np.var(x_2m_priori))**2 / ((np.var(x_2m_priori))**2 + (np.var(x_2m_ruido))**2)

x_2m = x_2m_priori + K1*(x_2m_ruido[0:50] - x_2m_priori)


# %%SEGUNDO CONJUNTO DE MUESTRAS (el resto de la señal):
"""
Aca se utiliza la ecuacion de la señal que se reprodujo para llegar a un 
estimador a priori contemplando la evolucion temporal de la señal y lograr
un estimador optimo de la medicion a 2 metros
"""


M=50 # Tamaño de ventana a filtrar

v=np.arange(50,len(x_2m_ruido),50) # Vector cuyos numeros se utilizan para ventanear
y_1m=y*((np.mean((x_1m)**2))/(np.mean(y**2))) # Valor de la señal generada a 1m
y_2m=y_1m/2 # Valor de la señal generada a 2m
x_2m_priori2=np.concatenate((x_2m,y_2m[50:len(y_2m)]),axis=None) # Estimador a priori de la segunda etapa
x_2m_etapa2=np.concatenate((x_2m,np.ones((len(x_2m_priori2)-50))),axis=None) # Vector medición a 2m que se va a filtrar



for u in range(len(v)) :
    w=v[u]
    x_2m_etapa2[w:w+M]=x_2m_priori2[w-M:w] 
    K = (np.var(x_2m_etapa2[w:w+M]))**2 / ((np.var(x_2m_etapa2[w:w+M]))**2 + (np.var(x_2m_ruido[w:w+M]))**2)
    x_2m_etapa2[w:w+M] = x_2m_etapa2[w:w+M] + K*(x_2m_ruido[w:w+M] - x_2m_etapa2[w:w+M])


x_2m_etapa2_dB=10*np.log10((x_2m_etapa2)**2/((20*(10**(-6)))**2)) # Pasaje a dB de la señal a 2m filtrada
Leq_2m=10*np.log10(np.mean(10**(x_2m_etapa2_dB/10))) # Nivel contínuo equivalente de la señal a 2m filtrada


# Varianzas
var_1m = np.var(x_1m)
var_2m = np.var(x_2m)
var_2m_r = np.var(x_2m_ruido)


# Cálculos de FFT para comparar las mediciones y la señal filtrada
x_2m_r_fft=np.fft.fft(x_2m_ruido)
x_2m_fft = np.fft.fft(x_2m_etapa2)
x_1m_fft = np.fft.fft(x_1m)

print('\n')
print('Nivel contínuo equivalente a 1 m: Leq_1m =',Leq_1m, ' dB' )

print('\n')
print('Nivel contínuo equivalente a 2 m con ruido: Leq_2m_ruido =',Leq_2m_ruido, ' dB' )

print('\n')
print('Nivel contínuo equivalente a 2 m filtrada: Leq_2m =',Leq_2m, ' dB' )

print('\n')
print ('Varianzas:')

print('\n')
print ('Varianza a 2 m con ruido: Var(2m_r) = ',var_2m_r)

print('\n')
print ('Varianza a 2 m filtrada: Var(2m) = ',var_2m)

# %% PLOTS

# Temporal
plt.figure()
plt.plot(x_2m_ruido,label='2 m con ruido')
plt.plot(x_2m_etapa2, label='2 m filtrada')
plt.xlim(10000,15000)
plt.legend()
plt.grid()
plt.xlabel('Muestras')
plt.ylabel('Amplitud')

plt.figure()
plt.plot(x_1m, label='1 m')
plt.xlim(10000,15000)
plt.legend()
plt.grid()
plt.xlabel('Muestras')
plt.ylabel('Amplitud')

#Espectral
plt.figure()
plt.subplot(3,1,1)
plt.plot(x_1m_fft[0:20000], label='1 m')
plt.legend(loc=0, shadow=True, fontsize='x-large')
plt.grid()
plt.title('Respuesta en frecuencia')
plt.ylabel('Amplitud')
plt.xlabel('Frecuencia')
plt.subplot(3,1,2)
plt.plot(x_2m_r_fft[0:20000], label='2 m con ruido')
plt.legend(loc=0, shadow=True, fontsize='x-large')
plt.grid()
plt.ylabel('Amplitud')
plt.xlabel('Frecuencia')
plt.subplot(3,1,3)
plt.plot(x_2m_fft[0:20000], label='2 m filtrada')
plt.legend(loc=0, shadow=True, fontsize='x-large')
plt.grid()
plt.ylabel('Amplitud')
plt.xlabel('Frecuencia')
plt.tight_layout(0.926,0.121,0.117)
plt.show()

