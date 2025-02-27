  
import scipy.io.wavfile as wavfile
import scipy
import scipy.fftpack as fftpk
import numpy as np
import wave
import naiveGuesser
import utils
import librosa
import math
import librosa.display
import geneticGuesser
import KeySignatureID
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from music21 import *

def plotFFT(freqs,FFT,peaks):
    plt.plot(freqs[range(len(FFT)//2)], FFT[range(len(FFT)//2)])       #x = frequencies, y = FFT amplitude


    plt.plot(freqs[peaks],FFT[peaks], "x")  #mark peaks with x

    axes = plt.gca()
    axes.set_xlim([0,2750])   #limit x axis                         
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (Relative)')
    plt.show()

#naive simple guess using thresholding and hand distance.
def naiveGuess(closestNoteListNoHarmonics,guessedNotes,namedNotes):
    guess,guessB = naiveGuesser.makeGuess(closestNoteListNoHarmonics)

    print("Predicted Notes: ")
    stringGuess = ""
    for x in range(len(guess)):
        stringGuess += guess[x][0] + " "
    print("Hand 1: " + stringGuess)

    stringGuess = ""
    for x in range(len(guessB)):
        stringGuess += guessB[x][0] + " "
    #print("Hand 2: " + stringGuess)
    finalGuess = [row[1] for row in guess] + [row[1] for row in guessB]
    guessedNotes.append(finalGuess)
    nameGuess= [row[0] for row in guess] + [row[0] for row in guessB]
    namedNotes.append(nameGuess)


#attempted genetic geuss algorithm (it doesnt work too well and works way too slowly.)
def geneticGuess(closestNoteListSorted,guessedNotes,namedNotes):
    #print(closestNoteListSorted)
    guess = geneticGuesser.makeGuess(closestNoteListSorted)

    #finalGuess = [row[1] for row in guess]
    #guessedNotes.append(finalGuess)
    nameGuess= [row[0] for row in guess]
    namedNotes.append(guess)


#used for display purposes.
def HPS(inputSignal,iterations):
    downsamples = []
    for x in range(iterations):
        downsampled = np.abs(scipy.signal.decimate(inputSignal,x+2,axis=0))
        pad_size = len(inputSignal) - len(downsampled)
        downsampled = np.pad(downsampled,((0,pad_size)),constant_values=0)
        downsamples.append(downsampled)

    total = np.copy(inputSignal)
    for x in range(len(downsamples)):
        total = total* downsamples[x]
    return total



def signalToNote(s_rate, signal,listFrequencies,frequencyNames,guessedNotes,namedNotes):


    FFT = abs(scipy.fft.fft(signal)) #FFT the signal
    
    freqs = scipy.fft.fftfreq(len(FFT), (1.0/s_rate)) #get increments of frequencies scaled with the sample rate of the audio
    print("freqs")
    print(freqs[10])
    FFT = utils.normalizeFFT(FFT,freqs) #normalize the FFT graph so that the largest peak has an amplitude of 1.0

    peaks, _ = find_peaks(FFT,prominence=0.05, height=0.05,distance=20) 
    peaks = [x for x in peaks if freqs[x]>=0] 
    #[[Freq,Amplitude]] get the frequency of the peaks and the amplitutde. Ascending frequency.
    freqAmp = utils.createListOfPeaks(peaks,freqs,FFT) 


    #use freqAmp and find the closest matching note for each element. [[noteName, noteNumber, amp, hz]]

    closestNoteList = utils.matchFreqToNote(freqAmp,frequencyNames,listFrequencies)


    closestNoteListSorted = sorted(closestNoteList.copy(),key=lambda x: x[2], reverse=True)
    naiveGuess(closestNoteList,guessedNotes,namedNotes)
    #geneticGuess(closestNoteListSorted,guessedNotes,namedNotes)
    
    plotFFT(freqs,FFT,peaks)

def generateBeatTimings(bpm):
    quarterNote = 60/(bpm)
    print(quarterNote)
    return quarterNote

def getClosestTiming(timeOfNotes,index, quarterNoteLength):
    quarterMultiplier = [1/16,1/8,1/4,1/2,1,2,4,8,16]
    if(index == len(timeOfNotes)-1 or len(timeOfNotes)==0):
        return 1.0
    lengthOfNote = timeOfNotes[index+1] - timeOfNotes[index]
    closestMultiplier = 0
    closestDistance = 100000000

    for x in range(len(quarterMultiplier)):
        distance = abs(quarterMultiplier[x]*quarterNoteLength - lengthOfNote)
        if(distance< closestDistance):
            closestDistance = distance
            closestMultiplier = quarterMultiplier[x]
    
    return closestMultiplier

def convertToXML(namedNotes,bpm,keySignature,frequencyNames,timeOfNotes):

    quarterNoteLength =generateBeatTimings(bpm)
    trebleStream = stream.Stream()
    bassStream = stream.Stream()
    trebleStream.clef = clef.TrebleClef()
    bassStream.clef = clef.BassClef()
    tmp = tempo.MetronomeMark(number=int(bpm))
    tsFourFour = meter.TimeSignature('4/4')
    keySign = key.Key(keySignature)
    trebleStream.append(tsFourFour)
    trebleStream.append(tmp)
    trebleStream.append(keySign)
    bassStream.append(tsFourFour)
    bassStream.append(keySign)



    for x in range(len(namedNotes)):
        noteList =[]
        bassList = []
        timing = getClosestTiming(timeOfNotes,x,quarterNoteLength)
        for y in range(len(namedNotes[x])):
            
            #midi numbers offset by +20 https://www.inspiredacoustics.com/en/MIDI_note_numbers_and_center_frequencies
            currentNote = utils.noteNameToNumber(namedNotes[x][y],frequencyNames) + 20
            currentNote = pitch.Pitch(currentNote)
            if(currentNote.accidental.name == "natural"):
                currentNote.accidental = None
            #print(currentNote.accidental.name)
            if(utils.noteNameToNumber(namedNotes[x][y],frequencyNames) + 20 <60):
                bassList.append(currentNote)
            else:
                noteList.append(currentNote)


        if(len(bassList)==0):
            r = note.Rest()
            r.duration.quarterLength = timing
            bassList.append(r)
            bassStream.append(bassList)
        else:
            c2 = chord.Chord(bassList)
            c2.duration.quarterLength = timing
            bassStream.append(c2)

        if(len(noteList)==0):
            r = note.Rest()
            r.duration.quarterLength = timing
            noteList.append(r)
            trebleStream.append(noteList)
        else:        
            c1 = chord.Chord(noteList)
            c1.duration.quarterLength = timing
            trebleStream.append(c1)

    s = stream.Score()
    s.insert(0, trebleStream)
    s.insert(0, bassStream)
    staffGroup1 = layout.StaffGroup([trebleStream,bassStream],name='Piano', abbreviation='Pno.', symbol='brace')
    s.insert(staffGroup1)
    s.write("musicxml", "test")
    #bassStream.write("musicxml", "test")


def main():
    us = environment.UserSettings()
    us['musicxmlPath'] = 'C:\\Program Files\\MuseScore 3\\bin\\MuseScore3.exe'
    listFrequencies = utils.generateFrequencies() #[27.5 ... 4186.009]
    frequencyNames = utils.generateFrequencyNames(listFrequencies) #['A-0' ... 'C-8']
    geneticGuesser.loadNoteSounds()

    guessedNotes = []
    namedNotes = []
    timeOfNotes = []   
    #0 if need to do multi slice analysis. (long files)
    singleSlice = 0

    testfile = "sounds/Singletons/Fdim.wav"
    bpm = 60    



    s_rate, signal = wavfile.read(testfile) #read the file and extract the sample rate and signal.
    signal = np.transpose(signal)
    signal = np.pad(signal,pad_width=[500,500], mode='constant')
    signal = np.transpose(signal)

    #if stereo convert to mono https://stackoverflow.com/questions/30401042/stereo-to-mono-wav-in-python
    if wave.open(testfile).getnchannels()==2:
        signal = signal.sum(axis=1)/2 

    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)



    if(singleSlice):
        signalToNote(s_rate,signal,listFrequencies,frequencyNames,guessedNotes,namedNotes)
    else:

        #used to analyse pieces rather than a single slice
        splits = librosa.onset.onset_detect(y=signal,sr=44100,hop_length=512, units='samples',backtrack=True) #uses onset detection to find where to split
        bpm = librosa.beat.tempo(y=signal, sr=44100,hop_length=256)

        splitSignals= np.array_split(signal, splits)
        for x in range(len(splitSignals)):
            print("  ")

            if(x!=0):
                print("Sample: " + str(splits[x-1]) + "  Time: " + str(float(splits[x-1]/s_rate)) + "  Note: " + str(x))
                timeOfNotes.append(float(splits[x-1]/s_rate))
                signalToNote(s_rate,splitSignals[x],listFrequencies,frequencyNames,guessedNotes,namedNotes)


    #print(namedNotes)
    print("BPM: " + str(bpm))
    keySignature = KeySignatureID.matchKeySignature(guessedNotes)
    convertToXML(namedNotes,bpm,keySignature,frequencyNames,timeOfNotes)



main()
