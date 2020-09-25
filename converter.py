from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
import scipy
import scipy.fftpack as fftpk
import wave
import timidity
from scipy.signal import find_peaks

def combineSignals(original,newSignal):


    if(len(original)>len(newSignal)):
        newSignal =np.concatenate([newSignal, np.zeros(len(original)-len(newSignal))])
    elif(len(newSignal)>len(original)):
        original = np.concatenate([original, np.zeros(len(newSignal)-len(original))])

    original = original + newSignal
    return original


testfile = "sounds/C4.wav"
testfileB = "sounds/G4.wav"
testfileC = "sounds/C5.wav"

s_rate, signal = wavfile.read(testfile) #file to FFT
if wave.open(testfile).getnchannels()==2:
    signal = signal.sum(axis=1)/2 #if stereo convert to mono https://stackoverflow.com/questions/30401042/stereo-to-mono-wav-in-python

s_rateB, signalB = wavfile.read(testfileB) #file to FFT
if wave.open(testfileB).getnchannels()==2:
    signalB = signalB.sum(axis=1)/2 #if stereo convert to mono https://stackoverflow.com/questions/30401042/stereo-to-mono-wav-in-python

s_rateC, signalC = wavfile.read(testfileC) #file to FFT
if wave.open(testfileC).getnchannels()==2:
    signalC = signalC.sum(axis=1)/2 #if stereo convert to mono https://stackoverflow.com/questions/30401042/stereo-to-mono-wav-in-python


newSignal =combineSignals(signal,signalB)
newSignal =combineSignals(newSignal,signalC)
signal = newSignal



FFT = abs(scipy.fft.fft(signal)) 
freqs = scipy.fft.fftfreq(len(FFT), (1.0/s_rate)) 


peaks, _ = find_peaks(FFT,distance=25) #find the peaks of audio
peaks = [x for x in peaks if freqs[x]>=0] #get rid of negative peaks.

#normalize FFT where biggest peak's amplitude is 1.0
largest = 0
largestFreq = 0
for x in range(0,len(peaks)):
    if (largest< FFT[peaks[x]]):
        largest = FFT[peaks[x]]
        largestFreq = freqs[peaks[x]]
FFT = FFT/largest


peaks, _ = find_peaks(FFT,prominence=0.1, height=0.05) #find the peaks of audio
peaks = [x for x in peaks if freqs[x]>=0] #get rid of negative peaks.




plt.plot(freqs[range(len(FFT)//2)], FFT[range(len(FFT)//2)]) 
plt.plot(freqs[peaks],FFT[peaks], "x")  #mark peaks with x

axes = plt.gca()
axes.set_xlim([0,freqs[peaks[len(peaks)-1]]+250])   #limit x axis     
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (Relative)')

plt.show()
