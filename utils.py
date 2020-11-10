import numpy as np
import math
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

#list of frequencies of piano notes from [27.5 ... 4186.009] using 12-tone equal temperament (12-TET)
def generateFrequencies():
    listFrequencies = []
    aFourTuning = 440.0
    for x in range(1,89,1): #get frequencies of typical 88 key piano https://en.wikipedia.org/wiki/Piano_key_frequencies
        listFrequencies.append(aFourTuning * (2**((x-49)/12)))
    return listFrequencies

#list of notenames ['A-0' ... 'C-8']
def generateFrequencyNames(listFrequencies):
    noteNames = ["A","A#Bb","B","C", "C#Db", "D", "D#Eb", "E", "F","F#Gb", "G", "G#Ab"]
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


def generateKeySignatures():
    noteNames = ["A","A#Bb","B","C", "C#Db", "D", "D#Eb", "E", "F","F#Gb", "G", "G#Ab"]
    noteNamesLen = len(noteNames)
    majorSteps = [2,2,1,2,2,2]
    #keySignatures = []
    majorScale = []
    for j in range(len(noteNames)):

        currentNote = j
        newScale = [noteNames[j]]
        for i in range(len(majorSteps)):
            currentNote = (currentNote+majorSteps[i])%noteNamesLen
            newScale.append(noteNames[currentNote])
        majorScale.append(newScale)

    print(majorScale)
    print(len(majorScale))


#normalize FFT where biggest peak's amplitude is 1.0
def normalizeFFT(FFT,freqs):
    
    peaks, _ = find_peaks(FFT,distance=25) #find the peaks of audio
    peaks = [x for x in peaks if freqs[x]>=0] #get rid of negative peaks.
    largest = 0
    for x in range(0,len(peaks)):
        if (largest< FFT[peaks[x]]):
            largest = FFT[peaks[x]]
    return FFT/largest


# helps discriminate against peaks that are high but quite far from an actual note (helps eliminate potential harmonics)
# havent used https://en.wikipedia.org/wiki/Cent_(music) but could use for more accuracy?
def multiplyDifference(freqAmp, closestNoteList, listFrequencies): 
    for x in range(len(freqAmp)):
        difference = abs(float(freqAmp[x][0])-listFrequencies[int(closestNoteList[x][1])])
        percentageDifference = difference/listFrequencies[int(closestNoteList[x][1])]
        #print(float(freqAmp[x][0]), listFrequencies[int(closestNoteList[x][1])])

        #print(" cent: ")
        centDifference = abs(1200 * math.log(float(freqAmp[x][0])/listFrequencies[int(closestNoteList[x][1])], 2))
        #print(centDifference)
        closestNoteList[x][2] = float(closestNoteList[x][2]) /(centDifference + 1 )

    closestNoteList = sorted(closestNoteList, key = lambda x: x[2], reverse=1) # [[freq of peak, amp]] sorted by relative amplitude descending.
    print(closestNoteList)
    return closestNoteList


#Somewhat works in removing false octave harmonics. Sets each peak as a fundamental then gets its harmonics. 
#If there is a note that is closer to a harmonic frequency rather than a note frequency then remove it as its likely to be a harmonic and not a played note.
def removeHarmonics(closestNoteList,listFrequencies):
    harmonics = 5
    potentialHarmonics = []
    for x in range(len(closestNoteList)):
        fundamentalFrequency = float(closestNoteList[x][3])
        #print("")
        #print("fundamental: " + str(fundamentalFrequency))
        harmonicsList = []
        for y in range(2,harmonics+2):
            harmonicsList.append(y*fundamentalFrequency)
        #print("harmonics: " + str(harmonicsList))
        for y in range((len(closestNoteList))):
            closestNoteFrequency = min(listFrequencies, key=lambda z:abs(z-closestNoteList[int(y)][3])) #get closest frequency
            closestHarmonicFrequency = min(harmonicsList, key=lambda z:abs(z-closestNoteList[int(y)][3]))
            
            #could use cent difference here instead.
            noteDiff = abs(closestNoteFrequency-closestNoteList[y][3])
            harmonicDiff = abs(closestHarmonicFrequency-closestNoteList[y][3])
            if(harmonicDiff<noteDiff):
                #print("note: " + str(closestNoteList[y][3]) + " | closest freq:"+ str(closestNoteFrequency) + " |  harmonic: " + str(closestHarmonicFrequency))
                potentialHarmonics.append(closestNoteList[y])
    #print(potentialHarmonics)
    closestNoteList = [i for i in closestNoteList + potentialHarmonics if i not in closestNoteList or i not in potentialHarmonics] 
    return closestNoteList

#create list of peak frequencies sorted by their relative amplitudes descending
def createListOfPeaks(peaks,freqs,FFT):
    freqAmp = [] 
    for x in range(0,len(peaks)):
        freqAmp.append([freqs[peaks[x]],FFT[peaks[x]]])   

    #freqAmp = sorted(freqAmp, key = lambda x: x[1], reverse=1) # [[freq of peak, amp]] sorted by relative amplitude descending.
    return freqAmp

#create list of peak frequencies sorted by their frequencies ascending
def createListPeaksFreqs(peaks,freqs,FFT):
    freqAmp = [] 
    for x in range(0,len(peaks)):
        freqAmp.append([freqs[peaks[x]],FFT[peaks[x]]])   

    freqAmp = sorted(freqAmp, key = lambda x: x[0], reverse=0) # [[freq of peak, amp]] sorted by relative amplitude descending.
    return freqAmp

#gets the list of freqAmp and matches it to the closest frequencies in listFrequencies. applies the index to the corresponding in frequencyNames
def matchFreqToNote(freqAmp, frequencyNames, listFrequencies):
    closestNoteList = []
    for y in range(len(freqAmp)):
        frequency = min(listFrequencies, key=lambda x:abs(x-freqAmp[int(y)][0])) #get closest frequency
        index = listFrequencies.index(frequency) #get the index/note number
        closestNoteList.append([frequencyNames[index],index,freqAmp[int(y)][1], freqAmp[int(y)][0]])
    return closestNoteList