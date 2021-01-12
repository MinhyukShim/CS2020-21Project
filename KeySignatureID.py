#Once the notes are detected, in order to convert to sheet music we need extra details such as Key signature and tempo.

#Detection of key signature relies on the detection of notes to be accurate. IF all the notes are wrong then the wrong key is going to be detected.


#The most simplest way to implement this feature is to find which notes match the best scale.

# 1: Have a list of all the notes A-G#Ab and count the number of guessed notes.
# 2: Key signatures are a specific subset of the range of notes
# 3: Go through each key and add up the count for each note in the scale.
# 4: return the best matching key which should most likely be the correct key signature. 


# major and minor scales have equivalents. for example. C major and A minor contain all the same notes but start at different notes.






#key signatures using each note as the tonic and then building major scales
def generateKeySignatures():
    #noteNames = ["A","A#Bb","B","C", "C#Db", "D", "D#Eb", "E", "F","F#Gb", "G", "G#Ab"]
    noteNames = ["A","A#","B","C", "C#", "D", "D#", "E", "F","F#", "G", "G#"]
    noteNamesLen = len(noteNames)
    majorSteps = [2,2,1,2,2,2]
    #keySignatures = []
    majorScale = []
    for initalNote in range(len(noteNames)):

        currentNote = initalNote
        newScale = [noteNames[initalNote]]
        for i in range(len(majorSteps)):
            currentNote = (currentNote+majorSteps[i])%noteNamesLen
            newScale.append(noteNames[currentNote])
        majorScale.append(newScale)

   # print(majorScale)
    return majorScale


#get the guessed notes and count which notes are played.
def countNoteNames(finalGuess):
    #First note in piano is A-0
    #noteNames = ["A","A#Bb","B","C", "C#Db", "D", "D#Eb", "E", "F","F#Gb", "G", "G#Ab"]
    noteNames = ["A","A#","B","C", "C#", "D", "D#", "E", "F","F#", "G", "G#"]
    noteCount = {}
    for x in range(len(noteNames)):
        noteCount[noteNames[x]] = 0

    
    for x in range(len(finalGuess)):
        for y in range(len(finalGuess[x])):
            noteNumber = int(finalGuess[x][y]) %12
            noteCount[noteNames[noteNumber]] += 1

    # [[NoteName,NoteCount]]
    #print(noteCount)
    return noteCount


def matchKeySignature(finalGuess):
    noteCount = countNoteNames(finalGuess)
    keys = generateKeySignatures()
    bestMatch = 0
    bestKey = ""
    for x in range(len(keys)):
        currentCount = 0
        for y in range(len(keys[x])):
            currentCount += noteCount[keys[x][y]]
        
        if (currentCount>bestMatch):
            bestKey = keys[x]
            bestMatch=currentCount
    
    print("Key: " + str(bestKey[0]) + " major")
    return bestKey[0]

