
import librosa
import os
import soundfile as sf
#from python_speech_features import mfcc
import numpy as np


file_path="C:\\Users\\Maaz Ahmed\\project\\audiodataset\\output_audio.2.wav"

from pydub import AudioSegment
import os

# set your input and output paths
# input_path = 'C:\\Users\\Maaz Ahmed\\Desktop\\dataset3\\clean2'
# output_path = '/path/to/wav/file.wav'


input_path = 'C:\\Users\\Maaz Ahmed\\Desktop\\try.mp3'
output_path = 'C:\\Users\\Maaz Ahmed\\Desktop\\try2.wav'

# create an AudioSegment object from the mp3 file
sound = AudioSegment.from_mp3(input_path)

# export the AudioSegment object as a wav file
sound.export(output_path, format='wav')

# j=0
# for i, (dirpath,dirnames,filenames) in enumerate(os.walk(input_path)): #loop thru all data
#     sorted_files = sorted(filenames, key=lambda x:(len(str(x)),x))
#             #loop thru all files and load their signals
#     for f in sorted_files:
# # create an AudioSegment object from the mp3 file
#                 path = os.path.join(input_path,f)
#                 sosund = AudioSegment.from_mp3(path)

#                 # export the AudioSegment object as a wav file
#                 sosund.export("C:\\Users\\Maaz Ahmed\\Desktop\\dataset3\\clean_wav2\\heavy"+str(j)+"output.wav", format='wav')
#                 j=j+1
