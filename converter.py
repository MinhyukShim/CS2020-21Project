from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
import scipy
import scipy.fftpack as fftpk
import wave
import utils
import random
from scipy.signal import find_peaks

def combineSignals(original,newSignal):

    if (original == []):
        return newSignal
    if(len(original)>len(newSignal)):
        newSignal =np.concatenate([newSignal, np.zeros(len(original)-len(newSignal))])
    elif(len(newSignal)>len(original)):
        original = np.concatenate([original, np.zeros(len(newSignal)-len(original))])

    original = original + newSignal
    return original

def makeSignal(noteList):
    directory = "notes/"
    finalSignal = []
    for x in range(len(noteList)):
        directoryNote = noteList[x]

        sound = directory + directoryNote + ".wav"
        s_rate, signal = wavfile.read(sound) #read the file and extract the sample rate and signal.

        if wave.open(sound).getnchannels()==2:
            signal = signal.sum(axis=1)/2 #if stereo convert to mono https://stackoverflow.com/questions/30401042/stereo-to-mono-wav-in-python
        finalSignal = combineSignals(finalSignal, signal)
    
    return finalSignal, s_rate





def calculateAccuracy(originalPeaks, generatedPeaks):
    noMatchPeaks = 0
    frequencyDifference = 0
    amplitudeDifference = 0
    for x in range(len(generatedPeaks)):
        currentPeak = generatedPeaks[x]
        peakExists = False
        for y in range(len(originalPeaks)):
            if (currentPeak[1] == originalPeaks[y][1]):
                peakExists = True
                frequencyDifference += abs(currentPeak[3] - originalPeaks[y][3])
                amplitudeDifference += abs(currentPeak[2] - originalPeaks[y][2])
            
        if(peakExists==False):
            noMatchPeaks += 1
    return noMatchPeaks, frequencyDifference, amplitudeDifference

def generateClosestNoteList(signal,s_rate,listFrequencies,frequencyNames):

    FFT = abs(scipy.fft.fft(signal)) #FFT the signal
    freqs = scipy.fft.fftfreq(len(FFT), (1.0/s_rate)) #get increments of frequencies scaled with the sample rate of the audio


    FFT = utils.normalizeFFT(FFT,freqs) #scale the FFT so that the largest peak has an amplitude of 1.0


    #find the peaks of the normalized graph
    peaks, _ = find_peaks(FFT,prominence=0.05, height=0.05) 
    peaks = [x for x in peaks if freqs[x]>=0] 

    freqAmp = utils.createListOfPeaks(peaks,freqs,FFT) # [[Freq,Amplitude]] #sorted by ascending frequency like peaks


    #use freqAmp and find the closest matching note for each element. [[noteName, noteNumber, amp, hz]]
    closestNoteList = utils.matchFreqToNote(freqAmp,frequencyNames,listFrequencies)
    return closestNoteList

def generateRandomNotes(originalPeaks):

    numberOfNotes = random.randint(1,10)
    notes = []
    print(numberOfNotes)
    x = 0
    while x < numberOfNotes:
        x += 1
        noteIndex = random.randint(0,len(originalPeaks)-1)
        noteName = originalPeaks[noteIndex][0]
        print(noteName)
        if (noteName in notes):
            x -= 1
            if(len(notes) >= len(originalPeaks)):
                x += 10000
        else:

            notes.append(noteName)
    print(notes)
    return notes

def crossBreed(notesA,notesB):
    length = int((len(notesA) + len(notesB) )/2)
    x=0
    while x < length:
        x += 1
        

def makeGuess(originalPeaks):
    listFrequencies = utils.generateFrequencies() #[27.5 ... 4186.009]
    frequencyNames = utils.generateFrequencyNames(listFrequencies) #['A-0' ... 'C-8']
    generations = 100
    population = 100

    notes = generateRandomNotes(originalPeaks)
    signal, s_rate = makeSignal(notes)
    closestNoteList = generateClosestNoteList(signal,s_rate,listFrequencies,frequencyNames)
    print(calculateAccuracy(originalPeaks,closestNoteList))