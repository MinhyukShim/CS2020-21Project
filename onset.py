import numpy as np
import scipy
import scipy.io.wavfile as wavfile
import librosa
import wave


testfile = "sounds/CmajScale.wav"
bpm = 60

s_rate, signal = wavfile.read(testfile) #read the file and extract the sample rate and signal.
if wave.open(testfile).getnchannels()==2:
    signal = signal.sum(axis=1)/2 #if stereo convert to mono https://stackoverflow.com/questions/30401042/stereo-to-mono-wav-in-python
