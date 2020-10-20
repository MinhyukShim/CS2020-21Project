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

listFrequencies = utils.generateFrequencies() #[27.5 ... 4186.009]
frequencyNames = utils.generateFrequencyNames(listFrequencies) #['A-0' ... 'C-8']
noteSounds = {}
s_rate = 44100

def loadNoteSounds():
    directory = "notes/"
    for x in range(len(frequencyNames)):        
        sound = directory + frequencyNames[x] + ".wav"
        try:
            _, signal = wavfile.read(sound)
            if wave.open(sound).getnchannels()==2:
                signal = signal.sum(axis=1)/2 #if stereo convert to mono https://stackoverflow.com/questions/30401042/stereo-to-mono-wav-in-python
            noteSounds[frequencyNames[x]] = signal
        except:
            noteSounds[frequencyNames[x]] = []

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
            noMatchPeaks +=1    
    noMatchPeaks += differenceInNotes(originalPeaks,generatedPeaks)
    return [noMatchPeaks, frequencyDifference, amplitudeDifference]

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
        noteIndex = int(random.triangular(0,len(originalPeaks)-1,0))
        noteName = originalPeaks[noteIndex][0]
        if (noteName in notes):
            x -= 1
            if(len(notes) >= len(originalPeaks)):
                x += 10000
        else:

            notes.append(noteName)
    return notes


def mutate(notes):
    randomMutate = random.randint(0,len(notes)-1)
    if(len(notes)>1):
        notes.remove(notes[randomMutate])
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
        if(chooseList==0):
            chooseNote = random.randint(0, len(notesA)-1)
            newNote =notesA[chooseNote]
            if(newNote in newNotes):
               x -= 1
               tooManyAttempts +=1
            else:
                newNotes.append(newNote)
        else:
            chooseNote = random.randint(0, len(notesB)-1)
            newNote =notesB[chooseNote]
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
    signal = makeSignal(["C-4", "G-4","C-5"])
    closestNoteList = generateClosestNoteList(signal,s_rate)
    accuracy = calculateAccuracy(originalPeaks,closestNoteList)
    print("acrcriacy::")
    print(accuracy)
    input("")

def makeGuess(originalPeaks):
    loadNoteSounds()
    print(originalPeaks)
    generations = 10
    population = 150
    crossBreedAmount = 40
    numberToKeep = 1
    mutationNumber = 40
    populationList = []
    for x in range(0,generations):

        #make new ones
        while len(populationList) <population:
            notes = generateRandomNotes(originalPeaks)
            newNotes = makeOne(originalPeaks,notes)
            populationList.append(newNotes)


        #sort by accuracy
        populationList = sorted(populationList,key=lambda x: (x[1][0],x[1][1]*x[1][2]))
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
            newNotes = mutate(notes)
            newNotes = makeOne(originalPeaks, newNotes)
            newPopulation.append(newNotes)


        #print(populationList)
        #print(populationList)
        print(makeOne(originalPeaks, populationList[0][0]))
        print(populationList[:20])
        print("  ")

        populationList = []
        populationList = newPopulation.copy()