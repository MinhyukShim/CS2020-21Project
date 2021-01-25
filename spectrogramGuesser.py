     
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

testfile = "sounds/CmajScale.wav"
bpm = 60    




s_rate, signal = wavfile.read(testfile) #read the file and extract the sample rate and signal.


#if stereo convert to mono https://stackoverflow.com/questions/30401042/stereo-to-mono-wav-in-python
if wave.open(testfile).getnchannels()==2:
    signal = signal.sum(axis=1)/2 

fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
splits = librosa.onset.onset_detect(y=signal,sr=44100,hop_length=512, units='samples',backtrack=True) #uses onset detection to find where to split


hop_length = 512 #increment of sample steps
window_size= 8192 #detail of fft
FFT = np.abs(librosa.stft(signal, n_fft=window_size, hop_length=hop_length,
              center=False))
freqs = librosa.fft_frequencies(sr=44100,n_fft=window_size)
D = librosa.amplitude_to_db(FFT,
                        ref=np.max)           
img = librosa.display.specshow(D, y_axis='log', sr=44100, hop_length=hop_length,
        x_axis='time', ax=ax)

#which onset to detect notes at.
split_index = 0

#transpose matrix so that time goes along x axis and range of freqs goes y axis. D_trans[x][y]
D_trans = np.transpose(D)


sample_length = (len(D_trans)*hop_length) #total length of file in samples (44100 samples per second)
percentage = splits[split_index]/sample_length #find how far in the file the onset is.
index = int(len(D_trans)*percentage) #get the corresponding index for the onset in D.
print(index)

#go through freqs at the slice. filter lower decibels.
for y in range(len(freqs)):
    if(D_trans[index][y]>=-15.0):
        print(str(D_trans[index][y]) + " " + str(freqs[y]))

plt.vlines(splits/44100,0,20000,colors=[0.2,1.0,0.2,0.4])
ax.set(title='Log-frequency power spectrogram')
ax.label_outer()
fig.colorbar(img, ax=ax, format="%+2.f dB")
plt.show()



