#Naive guess / "Programmed guess"

# Input: list of notes detected by FFT, sorted by amplitude (Descending) in this form: [NoteName, NoteNumber, Amplitude]. e.g. ['C-5' '51' '0.9048670809876272']
# Output: A list of notes that the program guessed what notes were played

# 1. Get the note with the highest amplitude. Add it to the 'guess' list as it is most likely this note was played.
# 2. Go down the list from loudest notes and see if it's within finger range.
# 3. Add it to the list and repeat until out of fingers or peaks.

# Pretty accuate. Usually gets all notes but also adds octaves that weren't played due to piano overtones/harmonics.
# Peaks of octaves are hard to tell if actually played or a harmonic when looking at peak amplitudes.


import numpy as np

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
    octaveAmpLimit = 1.5
    deleteList=[]
    for x in range (0, len(notes)):
        total = 0
        for y in range(0, len(peakList)):

            #add amplitudes of all the peaks at the octave.
            if( int(notes[x][1])+12 == int(peakList[y][1])):
                total += float(peakList[y][2])

        #if the sum of amplitudes is less tan 1.5 then add it to the delete list with the notenumber + octave to index the correct note to remove
        if(total<1.5):
            deleteList.append([int(notes[x][1])+12,total])

    # go through the delete lsit
    for x in range(len(deleteList)):
        for y in range(len(notes)):

            #check if the number is in the list of potential notes and if the loudest peak is quiet.
            if((deleteList[x][0] == int(notes[y][1]) ) and float(notes[y][2]) < octaveAmpLimit/2):
                notes = np.delete(notes, y, 0)
    return notes             

#returns a naive guess of the notes given.
def makeGuess(peakList):
    fingerNumbers = 5
    fingerRange = 13 # 13 = octave

    #add loudest note
    notes = np.array([peakList[0]]) 
    fingerNumbers -= 1

    for x in range(1,len(peakList)):
        testNote = peakList[x]
        
        if (float(testNote[2])>0.1): #if the amplitude is greater than 10% the max peak
            if (checkLargestDifference(testNote, notes, fingerRange)): #check finger range
                if(fingerNumbers>0): #check if you have enough fingers to play
                    if(checkIfNoteExists(testNote,notes)==False):
                        fingerNumbers -= 1
                        notes = np.append(notes,np.array([testNote]),axis=0)

    #check to remove if an octave needs to be removed.
    notes = checkOctaves(notes,peakList)
    return notes
