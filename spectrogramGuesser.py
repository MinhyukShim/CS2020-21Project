     
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
   
   
us = environment.UserSettings()
us['musicxmlPath'] = 'C:\\Program Files\\MuseScore 3\\bin\\MuseScore3.exe'
listFrequencies = utils.generateFrequencies() #[27.5 ... 4186.009]
frequencyNames = utils.generateFrequencyNames(listFrequencies) #['A-0' ... 'C-8']
geneticGuesser.loadNoteSounds()

guessedNotes = []
namedNotes = []
timeOfNotes = []   
#0 if need to do multi slice analysis. (long files)
singleSlice = 0

testfile = "sounds/AmajScale.wav"
bpm = 60    




s_rate, signal = wavfile.read(testfile) #read the file and extract the sample rate and signal.


#if stereo convert to mono https://stackoverflow.com/questions/30401042/stereo-to-mono-wav-in-python
if wave.open(testfile).getnchannels()==2:
    signal = signal.sum(axis=1)/2 

fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
splits = librosa.onset.onset_detect(y=signal,sr=44100,hop_length=512, units='samples',backtrack=True) #uses onset detection to find where to split
hop_length = 1024
window_size= 8192
FFT = np.abs(librosa.stft(signal, n_fft=window_size, hop_length=hop_length,
              center=False))
freqs = librosa.fft_frequencies(sr=44100,n_fft=window_size)
D = librosa.amplitude_to_db(FFT,
                        ref=np.max)           
img = librosa.display.specshow(D, y_axis='log', sr=44100, hop_length=hop_length,
        x_axis='time', ax=ax)


#Get decibels + frequency of first slice.
for x in range(len(test)):
    print(str(D[x][0]) + " " + str(freqs[x]))

plt.vlines(splits/44100,0,20000,colors=[0.2,1.0,0.2,0.4])
ax.set(title='Log-frequency power spectrogram')
ax.label_outer()
fig.colorbar(img, ax=ax, format="%+2.f dB")
plt.show()



