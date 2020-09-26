import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

#list of frequencies of piano notes from [27.5 ... 4186.009]
def generateFrequencies():
    listFrequencies = []
    aFourTuning = 440.0
    for x in range(1,89,1): #get frequencies of typical 88 key piano https://en.wikipedia.org/wiki/Piano_key_frequencies
        listFrequencies.append(aFourTuning * (2**((x-49)/12)))
    return listFrequencies

#list of notenames ['A-0' ... 'C-8']
def generateFrequencyNames(listFrequencies):
    noteNames = ["A","A#/Bb","B","C", "C#/Db", "D", "D#/Eb", "E", "F","F#/Gb", "G", "G#/Ab"]
    frequencyNames = []


    currentNote = 0 #iterates through note names
    octave = 0 #iterates if current note is a "C"
    for _ in range (len(listFrequencies)):
        if (noteNames[currentNote] == "C"):
            octave +=1
        frequencyNames.append(noteNames[currentNote]+"-"+str(octave))
        currentNote +=1
        if(currentNote >= len(noteNames)):
            currentNote = 0    

    return frequencyNames

#normalize FFT where biggest peak's amplitude is 1.0
def normalizeFFT(FFT,freqs):
    
    peaks, _ = find_peaks(FFT,distance=25) #find the peaks of audio
    peaks = [x for x in peaks if freqs[x]>=0] #get rid of negative peaks.
    largest = 0
    for x in range(0,len(peaks)):
        if (largest< FFT[peaks[x]]):
            largest = FFT[peaks[x]]
    return FFT/largest
