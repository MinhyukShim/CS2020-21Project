# The best algorithm tested so far.
# Uses the HPS algorithm talked about in the report.
# Reduces a lot of false positives.  


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

#Get the frequency note number and frequency name given a frequency by finding the closest known frequency.
def matchFrequencyToNote(frequency):
    frequency = min(listFrequencies, key=lambda x:abs(x-frequency))
    index = listFrequencies.index(frequency)
    return index, frequencyNames[index]


#remove duplicate notes and keep the loudest decibel.
def cleanNoteList(noteList):
    if(len(noteList)==1):
        return noteList
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

#same as above function but returns quietest
def cleanNoteQuiet(noteList):
    if(len(noteList)!=0):
        newList = [noteList[0]]
        for x in range(len(noteList)):

            sameFound = False
            for y in range(len(newList)):
                if(noteList[x]["noteNum"]==newList[y]["noteNum"]):
                    sameFound = True
                    if(noteList[x]["decibel"] <newList[y]["decibel"]):
                        newList[y]["decibel"]=noteList[x]["decibel"]


            if(sameFound==False):
                newList.append(noteList[x])
        
        return newList
    return []



#Given a slcie from the FFT, find the peaks and assign the correct notes for each peak.
def identifyNotes(noteSlice):
    noteList = []
    peaks, _ = find_peaks(noteSlice,prominence=prominence,height=height) 

    for y in range(len(noteSlice)):
    

        if(y in peaks):
            noteNum, noteName = matchFrequencyToNote(freqs[y])
            dictNote = {
                "noteNum": noteNum,
                "noteName": noteName,
                "decibel": noteSlice[y]
            }
            noteList.append(dictNote)



    noteList = cleanNoteList(noteList)
    return noteList





#used to display the spectrogram for testing and report writing (currently not used).
def displaySpectrogram(spectrogram,ax):
    img = librosa.display.specshow(spectrogram, y_axis='log', sr=s_rate, hop_length=hop_length,
            x_axis='time', ax=ax)
    #plt.vlines(splits/44100,0,20000,colors=[0.2,1.0,0.2,0.4])
    ax.set(title='Log-frequency power spectrogram')
    ax.label_outer()
    fig.colorbar(img, ax=ax)

#used to calculated the closest note timing to be used in the sheet music.
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


#60/bpm finds the time for each quarter note.
def generateBeatTimings(bpm):
    quarterNote = 60/(bpm)
    return quarterNote

#function to convert all the detected features into sheet music.
def convertToXML(namedNotes,bpm,keySignature,frequencyNames,timeOfNotes,heldList):
    
    #define the sheet music in music21
    quarterNoteLength =generateBeatTimings(bpm)
    trebleStream = stream.Stream()
    bassStream = stream.Stream()
    trebleStream.clef = clef.TrebleClef()
    bassStream.clef = clef.BassClef()
    tmp = tempo.MetronomeMark(number=int(bpm))
    tsFourFour = meter.TimeSignature('4/4')
    keySign = key.KeySignature(keySignature)
    trebleStream.append(tsFourFour)
    trebleStream.append(tmp)
    trebleStream.append(keySign)
    bassStream.append(tsFourFour)
    bassStream.append(keySign)



    #go through all notes and assign to sheet music.
    for x in range(len(namedNotes)):
        noteList =[]
        bassList = []
        timing = getClosestTiming(timeOfNotes,x,quarterNoteLength)
        if(x==len(namedNotes)-1):
            timing = 4
        for y in range(len(namedNotes[x][0])):
            #midi numbers offset by +20 https://www.inspiredacoustics.com/en/MIDI_note_numbers_and_center_frequencies
            currentNote = utils.noteNameToNumber(namedNotes[x][0][y],frequencyNames) + 20
            hold = False
            for z in range(len(heldList[x])):
                if(heldList[x][z][0]["noteNum"]==currentNote-21 and heldList[x][z][1]=="hold"):
                    hold = True
            if(hold==False):
                currentNote = pitch.Pitch(currentNote)
                if(currentNote.accidental.name == "natural"):
                    currentNote.accidental = None
                if(utils.noteNameToNumber(namedNotes[x][0][y],frequencyNames) + 20 <60):
                    bassList.append(currentNote)
                else:
                    noteList.append(currentNote)


        #if empty then assign a rest otherwise append for bass and treble clef.
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



#key signature detection algorithm as written in the report. "Krumhansl-Schmuckler key-finding algorithm"
def keySignatureIdentification(guessedNotes): 
    majorProfile =[6.35,	2.23,	3.48,	2.33,	4.38,	4.09,	2.52,	5.19,	2.39,	3.66,	2.29,	2.88] #http://rnhart.net/articles/key-finding/
    minorProfile = [6.33,	2.68,	3.52,	5.38,	2.60,	3.53,	2.54,	4.75,	3.98,	2.69,	3.34,	3.17]

    #count number of each notes starting with A.
    noteCount = []

    for x in range(12):

        count= 0
        for y in range(len(guessedNotes)):
            index = frequencyNames.index(guessedNotes[y])
            index = index%12
            if(index==x):
                count = count+1

        noteCount.append(count)
    

    #test major
    coefficients = []
    for x in range(12):
        coefficients.append(np.amin(np.corrcoef(majorProfile,noteCount)))
        noteCount =np.roll(noteCount,-1)

    #test minor
    for x in range(12):
        coefficients.append(np.amin(np.corrcoef(minorProfile,noteCount)))
        noteCount =np.roll(noteCount,-1)

    #get the best index
    index = np.argmax(coefficients)

    #return if major or minor and the respective key.
    majmin = "maj"
    if(index>12):
        majmin = "min"

    return index, majmin



#harmonic product spectrum algorithm. downsamples and multiplies them all together by the number of iterations.
def HPS(inputSignal,iterations):
    downsamples = []
    for x in range(iterations):
        downsampled = np.abs(scipy.signal.decimate(inputSignal,x+2,axis=0))
        pad_size = len(inputSignal) - len(downsampled)
        downsampled = np.pad(downsampled,((0,pad_size),(0,0)),constant_values=0)
        downsamples.append(downsampled)

    total = np.copy(inputSignal)
    for x in range(len(downsamples)):
        total = total* downsamples[x]
    return total

#remove any duplicate notes within an amplitude threshold.
def removeDuplicateNotes(outputList):

    newNoteList = []
    x=1
    tempList =[]
    for a in range(len(outputList[0][1])):
        tempList.append([outputList[0][1][a],"new"])

    newNoteList.append(tempList)
    for x in range(len(outputList)-1):
        tempList =[]
        currentSlice = outputList[x][1]
        previousSlice = outputList[x-1][1]
        for y in range(len(currentSlice)):
            currentNote = currentSlice[y]["noteNum"]
            currentDecibel = currentSlice[y]["decibel"]
            hold = False
            for z in range(len(previousSlice)):
                if(previousSlice[z]["noteNum"] == currentNote):
                    if(previousSlice[z]["decibel"]-10>currentDecibel):
                        hold = True

            if(hold==True):
                tempList.append([currentSlice[y],"hold"])
            else:
                tempList.append([currentSlice[y],"new"])
        newNoteList.append(tempList)
    return newNoteList



us = environment.UserSettings()
us['musicxmlPath'] = 'C:\\Program Files\\MuseScore 3\\bin\\MuseScore3.exe'
listFrequencies = utils.generateFrequencies() #[27.5 ... 4186.009]
frequencyNames = utils.generateFrequencyNames(listFrequencies) #['A-0' ... 'C-8']

guessedNotes = []
namedNotes = []
timeOfNotes = [] 


#input test file here.
testfile = "sounds/furelise.wav"
bpm = 60    



s_rate, signal = wavfile.read(testfile) #read the file and extract the sample rate and signal.

signal = np.transpose(signal)
signal = np.pad(signal,pad_width=[250,250], mode='constant') # pad
signal = np.transpose(signal)
#if stereo convert to mono https://stackoverflow.com/questions/30401042/stereo-to-mono-wav-in-python
if wave.open(testfile).getnchannels()==2:
    signal = signal.sum(axis=1)/2 



bpm = librosa.beat.tempo(y=signal, sr=s_rate,hop_length=256)
print("TEMPO: " + str(bpm))
hop_length = 256 #increment of sample steps
window_size= 8192*2 #detail of fft


splits = librosa.onset.onset_detect(y=signal,sr=s_rate,hop_length=hop_length, units='samples',backtrack=True) #uses onset detection to find where to split

prominence = 10
height = 10
hps_iteration_list = [5,15]
#hps_iteration_list = [1,1000]
splitSignals= np.array_split(signal, splits)
output = []
FFT = np.abs(librosa.stft(signal, n_fft=window_size, hop_length=hop_length,center=False))  


#go through each onset.
for x in range(1,len(splitSignals)):
    
    #pad if signal is too small.
    if(len(splitSignals[x])<window_size):
        padding_size = window_size-len(splitSignals[x])
        splitSignals[x] = np.transpose(splitSignals[x])
        splitSignals[x] = np.pad(splitSignals[x],pad_width=[0,padding_size], mode='constant') # pad
        splitSignals[x] = np.transpose(splitSignals[x])
    
    #apply FFT
    FFT = np.abs(librosa.stft(splitSignals[x], n_fft=window_size, hop_length=hop_length,center=False))
    freqs = librosa.fft_frequencies(sr=s_rate,n_fft=window_size)

    initial = np.copy(FFT)
    initial *= 100.0/initial.max() # normalize
    test = np.transpose(initial)

    #find number of peaks to dynamically apply HPS depending on peak number.
    peaks, _ = find_peaks(test[0],prominence=5,height=15) 
    if(len(peaks)<=hps_iteration_list[0]):
        hps_iteration= 1
    elif(len(peaks)<=hps_iteration_list[1]):
        hps_iteration=2
    else:
        hps_iteration=3

    hps_signal = HPS(initial,hps_iteration)
    hps_signal *= 100.0/hps_signal.max() # normalize
    d_trans = np.transpose(hps_signal)


    #store notes detected.
    noteDecibels = []
    finalNotes = []

    noteList = identifyNotes(d_trans[x])
    for y in range(len(noteList)):
        noteDecibels.append(noteList[y])
        finalNotes.append(noteList[y]["noteName"])    
    output.append([finalNotes,cleanNoteList(noteDecibels),cleanNoteQuiet(noteDecibels)])
    
heldList = removeDuplicateNotes(output)


#calculate onset times.
timeOfNotes = []
for x in range(len(splits)):
    timeOfNotes.append(float(splits[x]/s_rate))
extractNotesOnly = []
for x in range(len(output)):
    for y in range(len(output[x][0])):
        extractNotesOnly.append(output[x][0][y])


#key signature identification.
keyValue, majmin = keySignatureIdentification(extractNotesOnly)
noteNames = ["A","A#Bb","B","C", "C#Db", "D", "D#Eb", "E", "F","F#Gb", "G", "G#Ab"]
circleOfFifthsMaj =[3,-2,5,0,-5,2,-3,4,-1,6,1,-4]
circleOfFifthsMin = [0,-5,2,-3,4,-1,-6,1,-4,3,-2,5]

circleOfFifths = np.concatenate([circleOfFifthsMaj, circleOfFifthsMin])

print(circleOfFifths[keyValue],majmin)

#convert to xml
convertToXML(output,bpm,int(circleOfFifths[keyValue]),frequencyNames,timeOfNotes,heldList)