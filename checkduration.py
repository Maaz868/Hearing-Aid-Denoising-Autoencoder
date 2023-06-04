
import librosa
import os
import soundfile as sf
#from python_speech_features import mfcc
import numpy as np


file_path="C:\\Users\\Maaz Ahmed\\project\\audiodataset\\output_audio.2.wav"

from pydub import AudioSegment
import os
j=0

input_path="C:\\Users\\Maaz Ahmed\\Desktop\\dataset3\\clean2"

for i, (dirpath,dirnames,filenames) in enumerate(os.walk(input_path)): #loop thru all data
    sorted_files = sorted(filenames, key=lambda x:(len(str(x)),x))
            #loop thru all files and load their signals
    for f in sorted_files:
                if j<2000:
                        path = os.path.join(input_path,f)
                        x, sr = librosa.load(path, sr=16000)
                        duration = librosa.get_duration(y=x,sr=sr)
                
                        if duration>9 and duration <10.2:
                                sosund = AudioSegment.from_mp3(path)
                                sosund.export("C:\\Users\\Maaz Ahmed\\Desktop\\dataset3\\clean_wav2\\heavy"+str(j)+"output.wav", format='wav')
                                j=j+1
                        

                        # export the AudioSegment object as a wav file
                        
                        