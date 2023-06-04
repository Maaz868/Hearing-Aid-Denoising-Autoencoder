
import librosa
import os
import soundfile as sf
from pydub import AudioSegment
#from python_speech_features import mfcc
import numpy as np
import random as rand
#from autoencoder import Autoencoder




from pydub import AudioSegment

dataset= "C:\\Users\\Maaz Ahmed\\Desktop\\dataset3\\clean_wav"



j=0
for dirpath,dirnames,filenames in os.walk(dataset):
    if dirpath is dataset:
        dirpath_componenets = dirpath.split("\\") 
        label=dirpath_componenets[-1]
           
        sorted_files = sorted(filenames, key=lambda x:(len(str(x)),x))
            #loop thru all files and load their signals
        
        for f in sorted_files:
                file_path=os.path.join(dataset,f)
                audio_file1 = AudioSegment.from_file(file_path, format="wav")


                for dir,dirname,noise_files in os.walk("C:\\Users\\Maaz Ahmed\\Desktop\\dataset3\\noisenew"):
                    
                        integer = rand.randint(0,6)
                       
                        path= os.path.join("C:\\Users\\Maaz Ahmed\\Desktop\\dataset3\\noisenew",noise_files[j])
                        audio_file2 = AudioSegment.from_file(path, format="wav")
                        output_audio = audio_file1.overlay(audio_file2, position=integer*1000)

                        

                        output_audio.export("C:\\Users\\Maaz Ahmed\\Desktop\\dataset3\\newclean\\heavy"+str(j)+"output.wav", format="wav")
                        audio_file1.export("C:\\Users\\Maaz Ahmed\\Desktop\\dataset3\\clean\\heavy"+str(j)+"output.wav", format="wav")
                        
                        j=j+1
                
                
                    


                   
                











# for i, (dirpath,dirnames,filenames) in enumerate(os.walk(dataset)): #loop thru all data
#         dirpath_componenets = dirpath.split("\\") 
#         label=dirpath_componenets[-1]
#            # data["mapping"].append(label)
#             #print("\nProcessing {}".format(label))
#         sorted_files = sorted(filenames, key=lambda x:(len(str(x)),x))
#             #loop thru all files and load their signals
#         j=369
#         for f in sorted_files:
#             # integer = rand.randint(0,16)
#             # path= os.path.join(dataset,filenames[i])
#             # audio_file2 = AudioSegment.from_file(path, format="wav")
#             # output_audio = audio_file1.overlay(audio_file2, position=integer*1000)
#             # #output_path= os.path.join(")
#             # output_audio.export("C:\\Users\\Maaz Ahmed\\project\\kharab.wav", format="wav")
         
#             file_path = os.path.join(dirpath, f)
#             audio, sr = librosa.load(file_path,sr=22050)
#             duration = librosa.get_duration(y=audio, sr=sr)
#             if (duration < min):
#                   min=duration
#                   filename=file_path


#                 # file_path = os.path.join(dirpath, f)
#                 # mp3_file=file_path
#                 # audio = AudioSegment.from_file(mp3_file, format="mp3")
#                 # audio.export("C:\\Users\\Maaz Ahmed\\Desktop\\dataset3\\noisenew\\output."+str(j)+".wav", format="wav")
#                 # j=j+1
#                 # #signal, sr =librosa.load(file_path,sr=sample_rate)
#         print(min,filename)

# read the mp3 file
#audio = AudioSegment.from_file(mp3_file, format="mp3")

# export the audio in wav format
#audio.export(wav_file, format="wav")





# signal, sr =librosa.load(file_path,sr=22050)
#     #print(signal.shape)
# mfcc = librosa.feature.mfcc(y=signal,sr=sr,n_fft=2048,n_mfcc=32)

# audio_reconstructed = librosa.feature.inverse.mfcc_to_audio(mfcc, sr=sr, n_fft=2048, hop_length=512)


# sf.write('reconstructed_1audio.wav', audio_reconstructed, sr)


# print(mfcc.shape)

# mfcc=mfcc.T
# #return imageio.imread(path)[:128, :128, 0].reshape((128, 128, 1)).astype('float32') / 255.
# mfcc=mfcc[:128,:32]
# print(mfcc.shape)
# mfcc=mfcc.T
# audio_reconstructed = librosa.feature.inverse.mfcc_to_audio(mfcc, sr=sr, n_fft=2048, hop_length=512)


# sf.write('reconstructed_1audio.wav', audio_reconstructed, sr)


# # print(mfcc)
# # mean = np.mean(mfcc, axis=0)
# # std = np.std(mfcc, axis=0)
# # norm_mfcc = (mfcc - mean) / std
# # mfcc= norm_mfcc.T                 


# # mfcc = mfcc[...,np.newaxis]
# # mfccs = np.tile(mfcc, (100, 1, 1, 1))
# # mfccs = mfccs[:, :-13, :, :]

# # #pred=mfccs
# # pred = autoencoder.model.predict(mfccs)


# # pred = pred[0]
# # pred = np.squeeze(pred)

# # pred=pred.T
# # pred_norm = np.zeros_like(pred)



# # for i in range(pred.shape[1]):
# #     pred_norm[:, i] = (pred[:, i] * std[i]) + mean[i]

# # print(pred_norm)
# # #pred = (pred * std) + mean

# # print(pred_norm.shape)
# # #audio_reconstructed = librosa.feature.inverse.mfcc_to_audio(pred_norm, sr=sr, n_fft=2048, hop_length=512, win_length=2048)


# # #sf.write('reconstructed_1audio.wav', audio_reconstructed, sr)

