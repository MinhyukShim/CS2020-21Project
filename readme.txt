In order to run this program, python 3.8+ 64 bit must be installed(tested on python 3.8.5 64 bit).

Certain libraries need to be installed can be installed using pip. 
Libraries include: scipy, numpy, librosa,music21,matplotlib, wave

In order to input a recording, the file must be of .wav format and placed somewhere in the directory (recommended to be in the sound folder).
Look for the variable "testfile" and modify the directory so that the string is equal to the file you want converted.
For example: testfile = "sounds/furelise.wav"
See the sound folder to test on already downloaded audio files.

To run the HPS algorithm use "python HPS.py" or "python main.py" for the simple detection using thresholding and hand range.

The output should be called "test.xml" in the same directory as the python codes.

This can be opened using any sheet notation program. For a free one use musescore: https://musescore.org/en
Note that running the program again will overwrite the current .musicxml file so be sure to save it elsewhere.