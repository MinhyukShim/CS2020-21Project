     
import scipy.io.wavfile as wavfile
import scipy
import scipy.fftpack as fftpk
import numpy as np
import wave
import naiveGuesser
import utils
import librosa
import librosa.display
import geneticGuesser
import KeySignatureID
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from music21 import *
from collections import Counter
   

def matchFrequencyToNote(frequency):
    frequency = min(listFrequencies, key=lambda x:abs(x-frequency))
    index = listFrequencies.index(frequency)
    return index, frequencyNames[index]
    
def cleanNoteList(noteList):
    if(len(noteList)!=0):
        newList = [noteList[0]]
        for x in range(len(noteList)):

            sameFound = False
            for y in range(len(newList)):
                if(noteList[x]["noteNum"]==newList[y]["noteNum"]):
                    sameFound = True
                    if(noteList[x]["decibel"] >newList[y]["decibel"]):
                        newList[y]["decibel"]=noteList[x]["decibel"]


            if(sameFound==False):
                newList.append(noteList[x])
        
        return newList
    return []


def identifyNotes(noteSlice):
    noteList = []
    peaks, _ = find_peaks(noteSlice+80,prominence=40,height=30) 


    for y in range(len(noteSlice)):
    

        if(y in peaks):
            noteNum, noteName = matchFrequencyToNote(freqs[y])
            dictNote = {
                "noteNum": noteNum,
                "noteName": noteName,
                "decibel": noteSlice[y]
            }
            noteList.append(dictNote)


    #eliminate duplicates
    noteList = cleanNoteList(noteList)
    noteList = eliminateHarmonics(noteList)
    return noteList

def eliminateHarmonics(noteList):
    harmonics = [12,7,5,4,3,3,2] #https://www.earmaster.com/music-theory-online/ch04/chapter-4-5.html
    big_threshold = 2.5
    singleton_threshold = -35.0
    loud_threshold = -8
    harmonics =np.cumsum(harmonics)
    indexOfHarmonics = []

    for x in range(len(noteList)):
        tentative_fundamental = noteList[x]
        current_decibel = noteList[x]["decibel"]
        harmonic_found = False


        numberOfMatches = 0
        for y in range(len(harmonics)):
            noteNumber = tentative_fundamental["noteNum"]#fundamental note number
            noteNumber += harmonics[y]; #harmonic to check
            for z in range(len(noteList)):

                #if match check decibels
                if(noteList[z]["noteNum"]==noteNumber):
                    harmonic_found = True
                    numberOfMatches = numberOfMatches+1
                    if(noteList[z]["decibel"] <current_decibel-(big_threshold) ):
                        indexOfHarmonics.append(z)
        
        if(harmonic_found==False or current_decibel<singleton_threshold):
            indexOfHarmonics.append(x)

        if(numberOfMatches>=4):
            for y in range(len(harmonics)):
                noteNumber = tentative_fundamental["noteNum"]#fundamental note number
                noteNumber += harmonics[y]; #harmonic to check
                for z in range(len(noteList)):

                    #if match check decibels
                    if(noteList[z]["noteNum"]==noteNumber and noteList[z]["decibel"] < loud_threshold):
                        indexOfHarmonics.append(z)
    noteList = [i for j, i in enumerate(noteList) if j not in indexOfHarmonics]#https://stackoverflow.com/questions/11303225/how-to-remove-multiple-indexes-from-a-list-at-the-same-time/41079803
    return noteList

def loopThroughOnset(D_trans):

    sample_length = (len(D_trans)*hop_length) #total length of file in samples (44100 samples per second)

    for split_index in range(len(splits)):#

        percentage = splits[split_index]/sample_length #find how far in the file the onset is.
        index = int(len(D_trans)*percentage) #get the corresponding index for the onset in D.
        #go through freqs at the slice. filter lower decibels.
        identifyNotes(D_trans[index])

        if(split_index<len(splits)-1):
            nextPercentage = splits[split_index+1]/sample_length 
            newPercentage = (nextPercentage-percentage) / 16
            noteCount = []
            noteDecibels = []

            for x in range(15):
                percentage += newPercentage
                index = int(len(D_trans)*(percentage)) 
                noteList = identifyNotes(D_trans[index])
                for y in range(len(noteList)):
                    noteCount.append(noteList[y]["noteName"])
                    noteDecibels.append([noteList[y]["noteName"],noteList[y]["decibel"]])
            count = Counter(noteCount)
            threshold_number = (count.most_common(1)[0][1] )/2
            finalNotes = []

            iteratable_count = count.most_common(len(count))
            for x in range(len(iteratable_count)):
                if(iteratable_count[x][1]>=threshold_number):
                    finalNotes.append(iteratable_count[x][0])

            print(finalNotes)
            print(count)
            #print(noteDecibels)

        else:
            newPercentage = (1-percentage)/16
            noteCount = []
            for x in range(15):
                percentage += newPercentage
                index = int(len(D_trans)*(percentage)) 
                noteList = identifyNotes(D_trans[index])
                for y in range(len(noteList)):
                    noteCount.append(noteList[y]["noteName"])
            print(Counter(noteCount))
        print(" ")


def displaySpectrogram():
    img = librosa.display.specshow(D, y_axis='log', sr=44100, hop_length=hop_length,
            x_axis='time', ax=ax)
    plt.vlines(splits/44100,0,20000,colors=[0.2,1.0,0.2,0.4])
    ax.set(title='Log-frequency power spectrogram')
    ax.label_outer()
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.show()

us = environment.UserSettings()
us['musicxmlPath'] = 'C:\\Program Files\\MuseScore 3\\bin\\MuseScore3.exe'
listFrequencies = utils.generateFrequencies() #[27.5 ... 4186.009]
frequencyNames = utils.generateFrequencyNames(listFrequencies) #['A-0' ... 'C-8']
geneticGuesser.loadNoteSounds()

guessedNotes = []
namedNotes = []
timeOfNotes = []   

testfile = "sounds/DearYou.wav"
bpm = 60    




s_rate, signal = wavfile.read(testfile) #read the file and extract the sample rate and signal.


#if stereo convert to mono https://stackoverflow.com/questions/30401042/stereo-to-mono-wav-in-python
if wave.open(testfile).getnchannels()==2:
    signal = signal.sum(axis=1)/2 

fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)



hop_length = 256 #increment of sample steps
window_size= 8192  #detail of fft
splits = librosa.onset.onset_detect(y=signal,sr=44100,hop_length=hop_length, units='samples',backtrack=True) #uses onset detection to find where to split
FFT = np.abs(librosa.stft(signal, n_fft=window_size, hop_length=hop_length,
              center=False))
freqs = librosa.fft_frequencies(sr=44100,n_fft=window_size)
D = librosa.amplitude_to_db(FFT,
                        ref=np.max)           




#transpose matrix so that time goes along x axis and range of freqs goes y axis. D_trans[x][y]
D_trans = np.transpose(D)
loopThroughOnset(D_trans)



#displaySpectrogram()



