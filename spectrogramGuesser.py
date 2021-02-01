     
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


def identifyNotes(noteSlice):
    noteList = []
    peaks, _ = find_peaks(noteSlice+80,prominence=prominence,height=height) 
    #peaks, _ = find_peaks(noteSlice,prominence=0.5,height=1) 

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
    #print(noteList)
    noteList = cleanNoteList(noteList)
    noteList = eliminateHarmonics(noteList)
    return noteList

def eliminateHarmonics(noteList):
    harmonics = [12,7,5,4,3,3,2] #https://www.earmaster.com/music-theory-online/ch04/chapter-4-5.html
    big_threshold = 2.5
    singleton_threshold = -20.0
    loud_threshold = -20.0
    harmonics =np.cumsum(harmonics)
    indexOfHarmonics = []

    if(len(noteList)<=1):
        return noteList
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
    
    #print(noteList)
    #print("")
    noteList = [i for j, i in enumerate(noteList) if j not in indexOfHarmonics]#https://stackoverflow.com/questions/11303225/how-to-remove-multiple-indexes-from-a-list-at-the-same-time/41079803
    return noteList


def loopThroughOnset(D_trans):
    
    sample_length = (len(D_trans)*hop_length) #total length of file in samples (44100 samples per second)
    output = []
    for split_index in range(len(splits)):#
        percentage = splits[split_index]/sample_length #find how far in the file the onset is.
        index = int(len(D_trans)*percentage) #get the corresponding index for the onset in D.
        if (index > len(D_trans)-1):
            index = len(D_trans)-1
            percentage = index/len(D_trans)
        #go through freqs at the slice. filter lower decibels.
        identifyNotes(D_trans[index])

        nextPercentage =0
        newPercentage = 0
        noteCount = []
        noteDecibels = []
        if(split_index<len(splits)-1):
            nextPercentage = splits[split_index+1]/sample_length 
            newPercentage = (nextPercentage-percentage) / 64



            #print(noteDecibels)

        else:
            newPercentage = (1-percentage)/64


        for x in range(32):
            percentage += newPercentage
            index = int(len(D_trans)*(percentage)) 
            noteList = identifyNotes(D_trans[index])
            for y in range(len(noteList)):
                noteCount.append(noteList[y]["noteName"])
                noteDecibels.append(noteList[y])
                
        count = Counter(noteCount)
        if(len(count)>0):
            threshold_number = max((count.most_common(1)[0][1] )/2 , count.most_common(1)[0][1]-10)
            finalNotes = []

            iteratable_count = count.most_common(len(count))
            for x in range(len(iteratable_count)):
                if(iteratable_count[x][1]>=threshold_number):
                    finalNotes.append(iteratable_count[x][0])

            print(finalNotes)
            print(count)
            #print(noteDecibels)
            print(cleanNoteList(noteDecibels))
            print(cleanNoteQuiet(noteDecibels))
            print(" ")
            output.append([finalNotes,cleanNoteList(noteDecibels),cleanNoteQuiet(noteDecibels)])

    return output


def displaySpectrogram(spectrogram,ax):
    img = librosa.display.specshow(spectrogram, y_axis='log', sr=44100, hop_length=hop_length,
            x_axis='time', ax=ax)
    plt.vlines(splits/44100,0,20000,colors=[0.2,1.0,0.2,0.4])
    ax.set(title='Log-frequency power spectrogram')
    ax.label_outer()
    fig.colorbar(img, ax=ax, format="%+2.f dB")


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
        if(x==len(namedNotes)-1):
            timing = 4
        for y in range(len(namedNotes[x][0])):
            #midi numbers offset by +20 https://www.inspiredacoustics.com/en/MIDI_note_numbers_and_center_frequencies
            currentNote = utils.noteNameToNumber(namedNotes[x][0][y],frequencyNames) + 20
            currentNote = pitch.Pitch(currentNote)
            if(currentNote.accidental.name == "natural"):
                currentNote.accidental = None
            #print(currentNote.accidental.name)
            if(utils.noteNameToNumber(namedNotes[x][0][y],frequencyNames) + 20 <60):
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





us = environment.UserSettings()
us['musicxmlPath'] = 'C:\\Program Files\\MuseScore 3\\bin\\MuseScore3.exe'
listFrequencies = utils.generateFrequencies() #[27.5 ... 4186.009]
frequencyNames = utils.generateFrequencyNames(listFrequencies) #['A-0' ... 'C-8']

guessedNotes = []
namedNotes = []
timeOfNotes = [] 

prominence = 40
height = 30

padding = "sounds/padding.wav"
testfile = "sounds/MaryPoly.wav"
bpm = 60    



s_rate, signal = wavfile.read(testfile) #read the file and extract the sample rate and signal.
signal = np.transpose(signal)
signal = np.pad(signal,pad_width=[250,250], mode='constant')
signal = np.transpose(signal)
#if stereo convert to mono https://stackoverflow.com/questions/30401042/stereo-to-mono-wav-in-python
if wave.open(testfile).getnchannels()==2:
    signal = signal.sum(axis=1)/2 

fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)

bpm = librosa.beat.tempo(y=signal, sr=44100,hop_length=256)
beats = librosa.beat.beat_track(y=signal, sr=44100,hop_length=256,units='samples')
hop_length = 256 #increment of sample steps
window_size= 8192  #detail of fft
splits = librosa.onset.onset_detect(y=signal,sr=44100,hop_length=hop_length, units='samples',backtrack=True) #uses onset detection to find where to split
FFT = np.abs(librosa.stft(signal, n_fft=window_size, hop_length=hop_length,
              center=False))
freqs = librosa.fft_frequencies(sr=44100,n_fft=window_size)
D = librosa.amplitude_to_db(FFT,
                        ref=np.max)           

D_power = librosa.db_to_power(D,80)
ax1 =plt.subplot(1, 2, 1)
displaySpectrogram(D,ax1)

ax2=plt.subplot(1, 2, 2)
displaySpectrogram(D_power,ax2)
#plt.show()

#transpose matrix so that time goes along x axis and range of freqs goes y axis. D_trans[x][y]
D_trans = np.transpose(D)
output=loopThroughOnset(D_trans)
#print(output)
semitone_filterbank, sample_rates = librosa.filters.semitone_filterbank()

timeOfNotes = []
for x in range(len(splits)):
    timeOfNotes.append(float(splits[x]/s_rate))

convertToXML(output,bpm,"C",frequencyNames,timeOfNotes)
'''
print("SECOND")

prominence = 1
height = 5
D_trans = np.transpose(D_power)
output = loopThroughOnset(D_trans)
convertToXML(output,bpm,"E-",frequencyNames,timeOfNotes)'''
