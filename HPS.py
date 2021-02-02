     
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

def displaySpectrogram(spectrogram,ax):
    img = librosa.display.specshow(spectrogram, y_axis='log', sr=44100, hop_length=hop_length,
            x_axis='time', ax=ax)
    #plt.vlines(splits/44100,0,20000,colors=[0.2,1.0,0.2,0.4])
    ax.set(title='Log-frequency power spectrogram')
    ax.label_outer()
    fig.colorbar(img, ax=ax)


us = environment.UserSettings()
us['musicxmlPath'] = 'C:\\Program Files\\MuseScore 3\\bin\\MuseScore3.exe'
listFrequencies = utils.generateFrequencies() #[27.5 ... 4186.009]
frequencyNames = utils.generateFrequencyNames(listFrequencies) #['A-0' ... 'C-8']

guessedNotes = []
namedNotes = []
timeOfNotes = [] 

prominence = 40
height = 30

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
hop_length = 256 #increment of sample steps
window_size= 8192*2 #detail of fft


splits = librosa.onset.onset_detect(y=signal,sr=44100,hop_length=hop_length, units='samples',backtrack=True) #uses onset detection to find where to split
FFT = np.abs(librosa.stft(signal, n_fft=window_size, hop_length=hop_length,
              center=False))
freqs = librosa.fft_frequencies(sr=44100,n_fft=window_size)


FFT = librosa.amplitude_to_db(FFT,ref=np.max)   
total = np.copy(FFT)        
total = total+80


ax1 =plt.subplot(1, 2, 1)
displaySpectrogram(total,ax1)


d_down =scipy.signal.decimate(total,2,axis=0)

pad_size = len(total) - len(d_down)
d_down = np.pad(d_down,((0,pad_size),(0,0)),constant_values=0)
total = total*d_down




ax4=plt.subplot(1, 2, 2)
#total = librosa.amplitude_to_db(total,ref=np.max)   
displaySpectrogram(total,ax4)
plt.show()
#transpose matrix so that time goes along x axis and range of freqs goes y axis. D_trans[x][y]

