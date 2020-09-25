import numpy as np
#Naive guess / "Programmed guess"

# Input: list of notes detected by FFT, sorted by amplitude (Descending) in this form: [NoteName, NoteNumber, Amplitude]. e.g. ['C-5' '51' '0.9048670809876272']
# Output: A list of notes that the program guessed what notes were played

# 1. Get the note with the highest amplitude. Add it to the 'guess' list as it is most likely this note was played.
# 2. 



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


#returns a naive guess of the notes given.
def makeGuess(peakList):
    fingerNumbers = 5
    fingerRange = 13

    notes = np.array([peakList[0]]) 

    for x in range(1,len(peakList)):
        testNote = peakList[x]
        
        if (float(testNote[2])>0.1):
            if (checkLargestDifference(testNote, notes, fingerRange)):
                if(fingerNumbers>0):
                    if(checkIfNoteExists(testNote,notes)==False):
                        fingerNumbers = fingerNumbers-1
                        notes = np.concatenate((notes,np.array([testNote])),axis=0)


    return notes
