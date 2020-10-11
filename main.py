  
import scipy.io.wavfile as wavfile
import scipy
import scipy.fftpack as fftpk
import numpy as np
import wave
import naiveGuesser
import utils
import librosa
from matplotlib import pyplot as plt
from scipy.signal import find_peaks


def signalToNote(s_rate, signal,listFrequencies,frequencyNames):
    FFT = abs(scipy.fft.fft(signal)) #FFT the signal
    freqs = scipy.fft.fftfreq(len(FFT), (1.0/s_rate)) #get increments of frequencies scaled with the sample rate of the audio


    FFT = utils.normalizeFFT(FFT,freqs) #scale the FFT so that the largest peak has an amplitude of 1.0


    #find the peaks of the normalized graph
    peaks, _ = find_peaks(FFT,prominence=0.05, height=0.05) 
    peaks = [x for x in peaks if freqs[x]>=0] 

    freqAmp = utils.createListOfPeaks(peaks,freqs,FFT) # [[Freq,Amplitude]] #sorted by ascending frequency like peaks

    #print(freqAmp)

    #use freqAmp and find the closest matching note for each element. [[noteName, noteNumber, amp, hz]]
    closestNoteList = utils.matchFreqToNote(freqAmp,frequencyNames,listFrequencies)
    #closestNoteList= utils.multiplyDifference(freqAmp,closestNoteList,listFrequencies)
    closestNoteList = utils.removeHarmonics(closestNoteList,listFrequencies)
    print(closestNoteList)


    #guess = naiveGuesser.makeGuess(closestNoteList)
    guess,guessB = naiveGuesser.makeGuess(closestNoteList)
    #print(" Note | NoteNum. | Amp | Freq")
    #print("Hand 1:")
    #print(guess)
    #print("Hand 2:")
    #print(guessB)
    print("Predicted Notes: ")
    stringGuess = ""
    for x in range(len(guess)):
        stringGuess += guess[x][0] + " "
    print("Hand 1: " + stringGuess)

    stringGuess = ""
    for x in range(len(guessB)):
        stringGuess += guessB[x][0] + " "
    print("Hand 2: " + stringGuess)


    plt.plot(freqs[range(len(FFT)//2)], FFT[range(len(FFT)//2)])       #x = frequencies, y = FFT amplitude


    plt.plot(freqs[peaks],FFT[peaks], "x")  #mark peaks with x

    axes = plt.gca()
    axes.set_xlim([0,freqs[peaks[len(peaks)-1]]+250])   #limit x axis                         
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (Relative)')
    plt.show()




#dir_path = os.path.dirname(os.path.realpath(__file__))
listFrequencies = utils.generateFrequencies() #[27.5 ... 4186.009]
frequencyNames = utils.generateFrequencyNames(listFrequencies) #['A-0' ... 'C-8']


#0 if need to do multi slice analysis. (long files)
singleSlice = 0

testfile = "sounds/CmajScale.wav"
bpm = 60    

s_rate, signal = wavfile.read(testfile) #read the file and extract the sample rate and signal.

if wave.open(testfile).getnchannels()==2:
    signal = signal.sum(axis=1)/2 #if stereo convert to mono https://stackoverflow.com/questions/30401042/stereo-to-mono-wav-in-python

if(singleSlice):
    signalToNote(s_rate,signal,listFrequencies,frequencyNames)
else:
    #used for long file to split individiual lines.
    splits = librosa.onset.onset_detect(y=signal, units='samples') #uses onset detection
    splitSignals= np.array_split(signal, splits)
    #print(len(splitSignal))
    for x in range(len(splitSignals)):
        print("  ")
        if(x==0):
            print("sample: 0  time: 0")
        else:
            print("sample: " + str(splits[x-1]) + "  time: " + str(float(splits[x-1]/s_rate)))
            print("Note: " + str(x))
        signalToNote(s_rate,splitSignals[x],listFrequencies,frequencyNames)  


