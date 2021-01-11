  
import scipy.io.wavfile as wavfile
import scipy
import scipy.fftpack as fftpk
import numpy as np
import wave
import naiveGuesser
import utils
import librosa
import geneticGuesser
import KeySignatureID
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

testfile = "notes/A-4.wav"
bpm = 60    

s_rate, signal = wavfile.read(testfile) #read the file and extract the sample rate and signal.
#if wave.open(testfile).getnchannels()==2:
#    signal = signal.sum(axis=1)/2 
FFT = abs(scipy.fft.fft2(signal)) #FFT the signal
freqs = scipy.fft.fftfreq(len(FFT), (1.0/s_rate)) #get increments of frequencies scaled with the sample rate of the audio

#FFT = utils.normalizeFFT(FFT,freqs) #normalize the FFT graph so that the largest peak has an amplitude of 1.0

plt.plot(freqs[range(len(FFT)//2)], FFT[range(len(FFT)//2)])       #x = frequencies, y = FFT amplitude
#peaks, _ = find_peaks(FFT,prominence=0.05, height=0.05) 
#peaks = [x for x in peaks if freqs[x]>=0] 

#plt.plot(freqs[peaks],FFT[peaks], "x")  #mark peaks with x

axes = plt.gca()
#axes.set_xlim([0,freqs[peaks[len(peaks)-1]]+250])   #limit x axis                         
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (Relative)')
plt.show()