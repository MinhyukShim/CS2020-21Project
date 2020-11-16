

#key signatures
def generateKeySignatures():
    noteNames = ["A","A#Bb","B","C", "C#Db", "D", "D#Eb", "E", "F","F#Gb", "G", "G#Ab"]
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

    print(majorScale)
    return majorScale

def countNoteNames(finalGuess):
    #First note in piano is A-0
    noteNames = ["A","A#Bb","B","C", "C#Db", "D", "D#Eb", "E", "F","F#Gb", "G", "G#Ab"]
    noteCount = [0]*12
    
    for x in range(len(finalGuess)):
        for y in range(len(finalGuess[x])):
            noteNumber = int(finalGuess[x][y]) %12
            noteCount[noteNumber] += 1

    noteNameCount = []
    for x in range(len(noteNames)):
        noteNameCount.append([noteNames[x],noteCount[x]])
    return noteNameCount


def matchKeySignature(finalGuess):
    noteNameCount = countNoteNames(finalGuess)
    majorScale = generateKeySignatures()
    #for x in range(len(majorScale))