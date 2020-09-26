  
import scipy.io.wavfile as wavfile
import scipy
import scipy.fftpack as fftpk
import numpy as np
import wave
import naiveGuesser
import utils
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

#dir_path = os.path.dirname(os.path.realpath(__file__))
listFrequencies = utils.generateFrequencies() #[27.5 ... 4186.009]
frequencyNames = utils.generateFrequencyNames(listFrequencies) #['A-0' ... 'C-8']



testfile = "sounds/CGC.wav"


s_rate, signal = wavfile.read(testfile) #read the file and extract the sample rate and signal.


if wave.open(testfile).getnchannels()==2:
    signal = signal.sum(axis=1)/2 #if stereo convert to mono https://stackoverflow.com/questions/30401042/stereo-to-mono-wav-in-python

FFT = abs(scipy.fft.fft(signal)) #FFT the signal
freqs = scipy.fft.fftfreq(len(FFT), (1.0/s_rate)) #get increments of frequencies scaled with the sample rate of the audio


FFT = utils.normalizeFFT(FFT,freqs) #scale the FFT so that the largest peak has an amplitude of 1.0


#find the peaks of the normalized graph
peaks, _ = find_peaks(FFT,prominence=0.05, height=0.05) 
peaks = [x for x in peaks if freqs[x]>=0] 


#create list of peak frequencies with their relative amplitudes.
freqAmp = [] 
for x in range(0,len(peaks)):
    freqAmp.append([freqs[peaks[x]],FFT[peaks[x]]])   


freqAmp = sorted(freqAmp, key = lambda x: x[1], reverse=1) # [[freq of peak, amp]] sorted by relative amplitude descending.

#use freqAmp and find the closest matching note for each element. [[noteName, noteNumber, amp]]
peakList = []
for y in range(len(freqAmp)):
    frequency = min(listFrequencies, key=lambda x:abs(x-freqAmp[int(y)][0])) #get closest frequency
    index = listFrequencies.index(frequency) #get the index/note number
    peakList.append([frequencyNames[index],index,freqAmp[int(y)][1]])
#print(peakList)
print(naiveGuesser.makeGuess(peakList))


plt.plot(freqs[range(len(FFT)//2)], FFT[range(len(FFT)//2)])       #x = frequencies, y = FFT amplitude


plt.plot(freqs[peaks],FFT[peaks], "x")  #mark peaks with x


axes = plt.gca()
axes.set_xlim([0,freqs[peaks[len(peaks)-1]]+250])   #limit x axis                         
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (Relative)')
plt.show()

