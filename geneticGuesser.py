from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
import scipy
import scipy.fftpack as fftpk
import wave
import utils
import random
import math
from scipy.signal import find_peaks
from collections import OrderedDict

listFrequencies = utils.generateFrequencies() #[27.5 ... 4186.009]
frequencyNames = utils.generateFrequencyNames(listFrequencies) #['A-0' ... 'C-8']
noteSounds = {} #data structure for sound files/signals
s_rate = 44100





#Go through all the notes and add the corresponding note file. These nmote files are used to combine signals together to produce sounds.
def loadNoteSounds():
    directory = "notes/"
    for x in range(len(frequencyNames)):        
        sound = directory + frequencyNames[x] + ".wav"
        try:
            _, signal = wavfile.read(sound)
            #if stereo convert to mono https://stackoverflow.com/questions/30401042/stereo-to-mono-wav-in-python
            if wave.open(sound).getnchannels()==2:
                signal = signal.sum(axis=1)/2 
            noteSounds[frequencyNames[x]] = signal
        except:
            noteSounds[frequencyNames[x]] = []


#Gets two signals and returns one signal as the combination of the two.
def combineSignals(original,newSignal):

    if (original == []):
        return newSignal
    if(len(original)>len(newSignal)):
        newSignal =np.concatenate([newSignal, np.zeros(len(original)-len(newSignal))])
    elif(len(newSignal)>len(original)):
        original = np.concatenate([original, np.zeros(len(newSignal)-len(original))])

    original = original + newSignal
    return original


#Gets list of note list and then returns the combined signal
def makeSignal(noteList):
    finalSignal = []
    for x in range(len(noteList)):
        currentNote = noteList[x]

        signal = noteSounds[currentNote]
        finalSignal = combineSignals(finalSignal, signal)
    
    return finalSignal




def differenceInNotes(originalPeaks,generatedPeaks):
    copyOriginalPeaks = originalPeaks.copy()
    for x in range(len(generatedPeaks)):
        note = generatedPeaks[x][1]
        for y in range(len(copyOriginalPeaks)):
            if(copyOriginalPeaks[y][1] == note):
                del copyOriginalPeaks[y]
                break
    return len(copyOriginalPeaks)


def calculateScore(givenList):
    return (((givenList[0]+1)**3) + givenList[1]**2 +(givenList[2]*10)**2)

def calculateAccuracy(originalPeaks, generatedPeaks):
    noMatchPeaks = 0
    centDifference = 0
    amplitudeDifference = 0
    for x in range(len(generatedPeaks)):
        currentPeak = generatedPeaks[x]
        peakExists = False


        #find smallest difference. better than difference of all matching, cent difference better than frequency difference to avoid skew
        tempCent = 10000
        tempAmplitude = 10000
        for y in range(len(originalPeaks)):

            if (currentPeak[1] == originalPeaks[y][1]):
                peakExists = True
                tempCent = min( tempCent,abs(1200* math.log((currentPeak[3] /originalPeaks[y][3]), 2)) )
                tempAmplitude = min( tempAmplitude,abs(currentPeak[2] - originalPeaks[y][2]))
        
        if(peakExists==False):
            noMatchPeaks +=1
        else:
            centDifference +=tempCent
            amplitudeDifference += tempAmplitude
    noMatchPeaks += differenceInNotes(originalPeaks,generatedPeaks)
    return [noMatchPeaks, centDifference, amplitudeDifference]

def generateClosestNoteList(signal,s_rate):

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

    numberOfNotes = int(random.triangular(2,8,4))
    notes = []
    x = 0


    while x < numberOfNotes:
        x += 1
        #noteIndex = int(random.triangular(0,len(originalPeaks)-1,0))
        noteIndex = int(random.randint(0,len(originalPeaks)-1))
        noteName = originalPeaks[noteIndex][0]
        if (noteName in notes):
            x -= 1
            if(len(notes) >= len(originalPeaks)):
                #print("Test")
                x += 10000
        else:

            notes.append(noteName)
    return notes


def mutate(originalPeaks,notes):
    chooseMutate  = random.randint(1,100)
    if(chooseMutate <= 75):
        randomMutate = random.randint(0,len(notes)-1)
        if(len(notes)>2):
            notes.remove(notes[randomMutate])
    else:
        shuffledPeaks = originalPeaks.copy()
        #random.shuffle(shuffledPeaks)
        for y in range(len(shuffledPeaks)):
            if(not (shuffledPeaks[y][0] in notes)):
                notes.append(shuffledPeaks[y][0])
                break
    return notes


def crossBreed(notesA,notesB):

    #length = int((len(notesA) + len(notesB) )/2)
    length = random.randint(min(len(notesA),len(notesB))-1, max(len(notesA),len(notesB)))
    if(length<2):
        length =2
    x=0
    tooManyAttempts  =0
    newNotes = []
    while x < length and tooManyAttempts<50:
        x += 1
        chooseList = random.randint(0,1)
        chosenList = []
        if(chooseList==0):
            chosenList = notesA.copy()
        else:
            chosenList = notesB.copy()
        
        chooseNote = random.randint(0, len(chosenList)-1)
        newNote = chosenList[chooseNote]
        if(newNote in newNotes):
            x -= 1
            tooManyAttempts +=1
        else:
            newNotes.append(newNote)
    return newNotes

def makeOne(originalPeaks,notes):            

    signal = makeSignal(notes)
    closestNoteList = generateClosestNoteList(signal,s_rate)
    accuracy = calculateAccuracy(originalPeaks,closestNoteList)
    return [notes,accuracy]


def testNotes(originalPeaks):
    signal = makeSignal(["F#Gb-3","A-3","C#Db-4"])
    closestNoteList = generateClosestNoteList(signal,s_rate)
    accuracy = calculateAccuracy(originalPeaks,closestNoteList)
    print("acrcriacy::")
    print(accuracy)
    print(calculateScore(accuracy))
    input("")



def sortPopulation(populationList):
    #return sorted(populationList,key=lambda x: (x[1][0],x[1][1]*x[1][2])) #good
    return sorted(populationList,key=lambda x: (calculateScore(x[1])))





def makeGuess(originalPeaks):

    #print(testNotes(originalPeaks))


    #GA numbers
    generations = 5
    population = 200
    crossBreedAmount = 75
    numberToKeep = 1
    mutationNumber = 75


    populationList = []
    bestCandidates = []
    for x in range(0,generations):

        #make new group of notes for the population
        while len(populationList) <population:
            notes = generateRandomNotes(originalPeaks)
            if(len(originalPeaks) == 0):
                print("problem!")
            newNotes = makeOne(originalPeaks,notes)
            populationList.append(newNotes)


        #sort by accuracy compared to the Target signal.
        populationList = sortPopulation(populationList)
        newPopulation = []
        print("-----------------")
        print(populationList[0])

        #cross breed for next generation
        for a in range(0,crossBreedAmount):
            randomNumber = int(random.triangular(0,population-1,0))
            notesA = populationList[randomNumber][0]

            randomNumber = int(random.triangular(0,population-1,0))
            notesB = populationList[randomNumber][0]
            newNotes = crossBreed(notesA,notesB)
            newNotes = makeOne(originalPeaks, newNotes)
            newPopulation.append(newNotes)

        #keep best ones
        for a in range (0,numberToKeep):
            newPopulation.append(populationList[a])
        

        #mutate some
        for a in range(0,mutationNumber):
            randomNumber = int(random.triangular(0,population-1,0))
            notes = populationList[randomNumber][0].copy()
            
            newNotes = mutate(originalPeaks,notes)
            newNotes = makeOne(originalPeaks, newNotes)
            newPopulation.append(newNotes)


        print(makeOne(originalPeaks, populationList[0][0]))
        print(calculateScore(populationList[0][1]))
        bestCandidates = [row[0] for row in populationList[:20]].copy()
        print("  ")

        populationList = []
        populationList = newPopulation.copy()
    
    #print(bestCandidates)
    print("")
    #print(calculateScore(bestCandidates[0][1]))
    for y in range(len((bestCandidates))):
        bestCandidates[y] = tuple(sorted(bestCandidates[y]))
    bestCandidates = list(OrderedDict.fromkeys(bestCandidates))

    for y in range(len((bestCandidates))):
        bestCandidates[y] = list(bestCandidates[y])
    return bestCandidates[0]
