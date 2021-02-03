#Naive guess / "Programmed guess"

# Input: list of notes detected by FFT, sorted by amplitude (Descending) in this form: [NoteName, NoteNumber, Amplitude]. e.g. ['C-5' '51' '0.9048670809876272']
# Output: A list of notes that the program guessed what notes were played

# 1. Get the note with the highest amplitude. Add it to the 'guess' list as it is most likely this note was played.
# 2. Go down the list from loudest notes and see if it's within finger range.
# 3. Add it to the list and repeat until out of fingers or peaks.


# Pretty accuate. Usually gets all notes but also adds octaves that weren't played due to piano overtones/harmonics.
# Peaks of octaves are hard to tell if actually played or just a harmonic when looking at peak amplitudes.


# checkOctaves attempts to remove false positives on octaves. by checking if the octave is loud enough or has multiple harmonics. Removes quiet octaves. (can lead to false negatives)
# Arguably, having only false positives for sheet music is better than false negatives. since a person could probably figure out whether that octave is played or not.
# However, in terms of "pure accuracy", removing octaves may be better.

# Breaks down at really low notes: e.g. D-2. The normalized FFT is really messy. (see A-0). the harmonics of low notes creates lots of peaks.
# Some of these harmonic peaks are louder than the fundamental frequency. https://en.wikipedia.org/wiki/Harmonic_series_(music)

import numpy as np

amplitudeThreshold = 0.1
fingerNumbers = 5
fingerRange = 13 # 13 = octave

#returns true if the tentative note given already exists in the guess notes list
def checkIfNoteExists(note,noteList):
    for x in range(0,len(noteList)):
        if (int(note[1])==int(noteList[x][1])):
            return True


            
    return False


#returns true if the tentative note is within the finger range of every other 'guessed' note. 
def checkLargestDifference(note,noteList,fingerRange):
    
    for x in range(0,len(noteList)):
        #print (note, noteList)
        if(abs(int(note[1])-int(noteList[x][1]))>fingerRange):
            return False

    return True


# Attempt to remove false positive octaves.
def checkOctaves(notes,peakList):

    #the sum of amplitudes in the peaklist shouldnt exceed octaveAmpLimit, if it does then an octave most likely is played
    octaveAmpLimit = 1.25
    deleteList=[]
    for x in range (0, len(notes)):
        total = 0
        for y in range(0, len(peakList)):

            #add amplitudes of all the peaks at the octave.
            if( int(notes[x][1])+12 == int(peakList[y][1])):
                total += float(peakList[y][2])

        #if the sum of amplitudes is less than 1.5 then add it to the delete list with the notenumber + octave to index the correct note to remove
        if(total<octaveAmpLimit):
            deleteList.append([int(notes[x][1])+12,total])

    # go through the delete lsit
    #print(deleteList)
    indexes = []
    for x in range(len(deleteList)):
        for y in range(len(notes)):

            #check if the notenumber is in the list of potential notes and if the loudest peak is quiet.
            if((deleteList[x][0] == int(notes[y][1]) ) and float(notes[y][2]) < octaveAmpLimit/2):
                indexes.append(y)
    notes = np.delete(notes, indexes, 0)
    return notes             



#implemented second hand check
# lower notes heavily impact higher notes, way more false positives here.
def checkSecondHand(peakList,fingerNumbers, fingerRange,takenNotes):

    notes = np.array([["","","",""]]) 
    for x in range(len(peakList)):
        found = 0
        for y in range(len(takenNotes)):
            if((int(takenNotes[y][1]) == int(peakList[x][1]))):
                found = 1
                break
        if(found==0):
            notes = np.append(notes, np.array([peakList[x]]),axis=0)
    notes = np.delete(notes,0,axis=0)
    #notesCopy =notes.copy()
    #notesCopy = sorted(notesCopy, key = lambda x: x[2], reverse=1)
    #print(notes)
    if(len(notes)==0 or float(notes[0][2])<amplitudeThreshold):
        return []

    notes = checkHand(notes, fingerNumbers, fingerRange,2)
    
    return notes

def checkHand(peakList, fingerNumbers, fingerRange,hand):
    peakList = sorted(peakList, key = lambda x: x[2], reverse=1) # [[freq of peak, amp]] sorted by relative amplitude descending.
    #print(peakList)


    #add loudest note

    notes = np.array([peakList[0]]) 
    fingerNumbers -= 1
    for x in range(1,len(peakList)):
        testNote = peakList[x]
        currentAmplitude = float(testNote[2])
        if (float(testNote[2])>amplitudeThreshold): #if the amplitude is greater than 15% the max peak
            if (checkLargestDifference(testNote, notes, fingerRange)): #check finger range
                if(fingerNumbers>0): #check if you have enough fingers to play
                    if(checkIfNoteExists(testNote,notes)==False):
                        fingerNumbers -= 1
                        notes = np.append(notes,np.array([testNote]),axis=0)




    #check to remove if an octave needs to be removed.
    newNotes = notes
    notes = checkOctaves(notes,peakList)

    return notes

#returns a naive guess of the notes given.
def makeGuess(peakList):

    notesA = []
    notesB = []
    if(len(peakList)>0):
        notesA = checkHand(peakList,fingerNumbers,fingerRange,1)

        notesB = checkSecondHand(peakList,fingerNumbers,fingerRange,notesA)

    #print("check")
    #print(notesB)
    #return notesA
    return notesA, notesB
   
