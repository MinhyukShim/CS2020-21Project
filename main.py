  
import scipy.io.wavfile as wavfile
import scipy
import scipy.fftpack as fftpk
import numpy as np
import wave
import naiveGuesser
import utils
import librosa
import geneticGuesser
import KeySignatureID
from matplotlib import pyplot as plt
from scipy.signal import find_peaks


def plotFFT(freqs,FFT,peaks):
    plt.plot(freqs[range(len(FFT)//2)], FFT[range(len(FFT)//2)])       #x = frequencies, y = FFT amplitude


    plt.plot(freqs[peaks],FFT[peaks], "x")  #mark peaks with x

    axes = plt.gca()
    axes.set_xlim([0,freqs[peaks[len(peaks)-1]]+250])   #limit x axis                         
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (Relative)')
    #plt.show()


def naiveGuess(closestNoteListNoHarmonics,guessedNotes,namedNotes):
    guess,guessB = naiveGuesser.makeGuess(closestNoteListNoHarmonics)

    print("Predicted Notes: ")
    stringGuess = ""
    for x in range(len(guess)):
        stringGuess += guess[x][0] + " "
    print("Hand 1: " + stringGuess)

    stringGuess = ""
    for x in range(len(guessB)):
        stringGuess += guessB[x][0] + " "
    print("Hand 2: " + stringGuess)
    finalGuess = [row[1] for row in guess] + [row[1] for row in guessB]
    guessedNotes.append(finalGuess)
    nameGuess= [row[0] for row in guess] + [row[0] for row in guessB]
    namedNotes.append(nameGuess)

def geneticGuess(closestNoteListSorted,guessedNotes,namedNotes):
    print(closestNoteListSorted)
    guess = geneticGuesser.makeGuess(closestNoteListSorted)
    finalGuess = [row[1] for row in guess]
    guessedNotes.append(finalGuess)
    nameGuess= [row[0] for row in guess]
    namedNotes.append(nameGuess)

def signalToNote(s_rate, signal,listFrequencies,frequencyNames,guessedNotes,namedNotes):


    FFT = abs(scipy.fft.fft(signal)) #FFT the signal
    freqs = scipy.fft.fftfreq(len(FFT), (1.0/s_rate)) #get increments of frequencies scaled with the sample rate of the audio

    FFT = utils.normalizeFFT(FFT,freqs) #normalize the FFT graph so that the largest peak has an amplitude of 1.0


    #find the peaks of the normalized graph and get rid of negative peaks.
    peaks, _ = find_peaks(FFT,prominence=0.05, height=0.05) 
    peaks = [x for x in peaks if freqs[x]>=0] 


    #[[Freq,Amplitude]] get the frequency of the peaks and the amplitutde. Ascending frequency.
    freqAmp = utils.createListOfPeaks(peaks,freqs,FFT) 


    #use freqAmp and find the closest matching note for each element. [[noteName, noteNumber, amp, hz]]

    closestNoteList = utils.matchFreqToNote(freqAmp,frequencyNames,listFrequencies)
    #print(closestNoteList)
    #closestNoteList= utils.multiplyDifference(freqAmp,closestNoteList,listFrequencies)
    #closestNoteListNoHarmonics = utils.removeHarmonics(closestNoteList,listFrequencies)

    closestNoteListSorted = sorted(closestNoteList.copy(),key=lambda x: x[2], reverse=True)
    #print(closestNoteListNoHarmonics)
    #naiveGuess(closestNoteList,guessedNotes,namedNotes)
    geneticGuess(closestNoteListSorted,guessedNotes,namedNotes)
    
    plotFFT(freqs,FFT,peaks)




def main():

    listFrequencies = utils.generateFrequencies() #[27.5 ... 4186.009]
    frequencyNames = utils.generateFrequencyNames(listFrequencies) #['A-0' ... 'C-8']
    geneticGuesser.loadNoteSounds()

    guessedNotes = []
    namedNotes = []
    #0 if need to do multi slice analysis. (long files)
    singleSlice = 0

    testfile = "sounds/shelter.wav"
    bpm = 60    

    s_rate, signal = wavfile.read(testfile) #read the file and extract the sample rate and signal.

    #if stereo convert to mono https://stackoverflow.com/questions/30401042/stereo-to-mono-wav-in-python
    if wave.open(testfile).getnchannels()==2:
        signal = signal.sum(axis=1)/2 

    if(singleSlice):
        signalToNote(s_rate,signal,listFrequencies,frequencyNames,guessedNotes)
    else:

        #used to analyse pieces rather than a single slice
        splits = librosa.onset.onset_detect(y=signal,sr=44100, units='samples',backtrack=True) #uses onset detection to find where to split
        bpm = librosa.beat.tempo(y=signal, sr=44100,hop_length=256)

        splitSignals= np.array_split(signal, splits)
        for x in range(len(splitSignals)):
            print("  ")
            if(x==0):
                print("Sample: 0  Time: 0  Note: 0")
            else:
                print("Sample: " + str(splits[x-1]) + "  Time: " + str(float(splits[x-1]/s_rate)) + "  Note: " + str(x))
            signalToNote(s_rate,splitSignals[x],listFrequencies,frequencyNames,guessedNotes,namedNotes)


    print(namedNotes)

    print("BPM: " + str(bpm))
    KeySignatureID.matchKeySignature(guessedNotes)

main()