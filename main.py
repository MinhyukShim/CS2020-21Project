  
import scipy.io.wavfile as wavfile
import scipy
import scipy.fftpack as fftpk
import numpy as np
import audioop
import wave
import contextlib
import mingus
import os 
import naiveGuesser
from mingus.containers import NoteContainer
from mingus.midi import midi_file_out
from matplotlib import pyplot as plt
from scipy.signal import find_peaks


def generateFrequencies():
    listFrequencies = []
    aFourTuning = 440.0
    for x in range(1,89,1): #get frequencies of typical 88 key piano https://en.wikipedia.org/wiki/Piano_key_frequencies
        listFrequencies.append(aFourTuning* (2**((x-49)/12)))
    return listFrequencies


def generateFrequencyNames():
    noteNames = ["A","A#/Bb","B","C", "C#/Db", "D", "D#/Eb", "E", "F","F#/Gb", "G", "G#/Ab"]
    frequencyNames = []


    currentNote = 0 #iterates through note names
    octave = 0 #iterates if current note is a "C"
    for x in range (len(listFrequencies)):
        if (noteNames[currentNote] == "C"):
            octave = octave+1
        frequencyNames.append(noteNames[currentNote]+"-"+str(octave))
        currentNote= currentNote+1
        if(currentNote >= len(noteNames)):
            currentNote = 0    

    return frequencyNames

def normalizeFFT(FFT):
    #normalize FFT where biggest peak's amplitude is 1.0
    largest = 0
    largestFreq = 0
    for x in range(0,len(peaks)):
        if (largest< FFT[peaks[x]]):
            largest = FFT[peaks[x]]
            largestFreq = freqs[peaks[x]]
    return FFT/largest



dir_path = os.path.dirname(os.path.realpath(__file__))
listFrequencies = generateFrequencies()
frequencyNames = generateFrequencyNames()



testfile = "sounds/CGC.wav"


s_rate, signal = wavfile.read(testfile) #file to FFT
if wave.open(testfile).getnchannels()==2:
    signal = signal.sum(axis=1)/2 #if stereo convert to mono https://stackoverflow.com/questions/30401042/stereo-to-mono-wav-in-python

FFT = abs(scipy.fft.fft(signal)) #FFT the signal

freqs = scipy.fft.fftfreq(len(FFT), (1.0/s_rate)) #get increments of frequencies scaled with the sample rate of the audio
peaks, _ = find_peaks(FFT,distance=25) #find the peaks of audio
peaks = [x for x in peaks if freqs[x]>=0] #get rid of negative peaks.




FFT = normalizeFFT(FFT)

peaks, _ = find_peaks(FFT,prominence=0.05, height=0.05) #find the peaks of audio
peaks = [x for x in peaks if freqs[x]>=0] #get rid of negative peaks.


freqAmp = [] #create list of frequencies with their relative amplitudes.
for x in range(0,len(peaks)):
    freqAmp.append([freqs[peaks[x]],FFT[peaks[x]]])   

freqAmp = sorted(freqAmp, key = lambda x: x[1], reverse=1)
#print(freqAmp)


peakList= []
for y in range(len(freqAmp)):
    frequency = min(listFrequencies, key=lambda x:abs(x-freqAmp[int(y)][0]))
    index = listFrequencies.index(frequency)
    #print(frequencyNames[index])
    peakList.append([frequencyNames[index],index,freqAmp[int(y)][1]])


print(naiveGuesser.makeGuess(peakList))


plt.plot(freqs[range(len(FFT)//2)], FFT[range(len(FFT)//2)])       #x = frequencies, y = FFT amplitude


plt.plot(freqs[peaks],FFT[peaks], "x")  #mark peaks with x


axes = plt.gca()
axes.set_xlim([0,freqs[peaks[len(peaks)-1]]+250])   #limit x axis                         
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (Relative)')
plt.show()

