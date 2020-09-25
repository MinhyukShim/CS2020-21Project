from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
import scipy
import scipy.fftpack as fftpk
import wave
import utils
from scipy.signal import find_peaks

def combineSignals(original,newSignal):

    if(len(original)>len(newSignal)):
        newSignal =np.concatenate([newSignal, np.zeros(len(original)-len(newSignal))])
    elif(len(newSignal)>len(original)):
        original = np.concatenate([original, np.zeros(len(newSignal)-len(original))])

    original = original + newSignal
    return original


def makeMono(signal):
    if(signal.ndim==2):
        return signal.sum(axis=1)/2 #if stereo convert to mono https://stackoverflow.com/questions/30401042/stereo-to-mono-wav-in-python
    return signal



#needs fixing: make code more general to support different amount of sounds to combine.
testfile = "sounds/C4.wav"
testfileB = "sounds/G4.wav"
testfileC = "sounds/C5.wav"

s_rate, signal = wavfile.read(testfile) #file to FFT
signal = makeMono(signal)

s_rateB, signalB = wavfile.read(testfileB) #file to FFT
signalB = makeMono(signalB)

s_rateC, signalC = wavfile.read(testfileC) #file to FFT
signalC = makeMono(signalC)


newSignal = combineSignals(signal,signalB)
newSignal = combineSignals(newSignal,signalC)
signal = newSignal



FFT = abs(scipy.fft.fft(signal)) 
freqs = scipy.fft.fftfreq(len(FFT), (1.0/s_rate)) 


peaks, _ = find_peaks(FFT,distance=25) #find the peaks of audio
peaks = [x for x in peaks if freqs[x]>=0] #get rid of negative peaks.

#normalize FFT where biggest peak's amplitude is 1.0
FFT = utils.normalizeFFT(FFT,peaks)


peaks, _ = find_peaks(FFT,prominence=0.1, height=0.05) #find the peaks of audio
peaks = [x for x in peaks if freqs[x]>=0] #get rid of negative peaks.



plt.plot(freqs[range(len(FFT)//2)], FFT[range(len(FFT)//2)]) 
plt.plot(freqs[peaks],FFT[peaks], "x")  #mark peaks with x

axes = plt.gca()
axes.set_xlim([0,freqs[peaks[len(peaks)-1]]+250])   #limit x axis     
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (Relative)')

plt.show()
