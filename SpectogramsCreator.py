import os
import librosa
import numpy as np
from scipy import signal
import soundfile as sf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pickle





sample_rate = 16000
input_dataset= "C:\\Users\\Maaz Ahmed\\Desktop\\dataset3\\clean"
output_dataset= "C:\\Users\\Maaz Ahmed\\Desktop\\dataset3\\clean"

def save_spectograms(input_dataset,output_dataset):
    
    lost=[]
    data_type = np.float32
    input_spectrogram = np.empty((2000,1025,256), dtype=data_type)
    output_spectrogram = np.empty((2000,1025,256), dtype=data_type)

   
    j=0
    for i, (dirpath,dirnames,filenames) in enumerate(os.walk(input_dataset)): #loop thru all data
        if dirpath is input_dataset: #this will not let me work in dataset2 root folder

            dirpath_componenets = dirpath.split("\\") 
            label=dirpath_componenets[-1]
            print("\nProcessing {}".format(label))
            file_index=[]
            sorted_files = sorted(filenames, key=lambda x:(len(str(x)),x))
            #loop thru all files and load their signals
            for f in sorted_files:
                    file_path = os.path.join(dirpath, f)
                    signal, sr =librosa.load(file_path,sr=sample_rate)

                
                    Spectogram = np.abs(librosa.stft(signal))
                    X_decibels = librosa.amplitude_to_db(Spectogram)
                    shape_spec= X_decibels.shape[1]
                    to_be_subtracted=shape_spec-256
                    X_decibels=X_decibels[:,:-to_be_subtracted]
                    
                    if (X_decibels.shape[1]!=0):
                        scaler = MinMaxScaler(feature_range=(-1, 1))
                        scaler.fit(X_decibels.T)
                        norm_x = scaler.transform(X_decibels.T).T
                        
                    if (norm_x.shape[1]<1):
                        lost.append(j)
                    else:
                        print(f)
                        input_spectrogram[j, :, :] = norm_x
                    
                    j=j+1
                


    j=0

    for i, (dirpath,dirnames,filenames) in enumerate(os.walk(output_dataset)): #loop thru all data
        if dirpath is output_dataset: #this will not let me work in dataset2 root
            sorted_files = sorted(filenames, key=lambda x:(len(str(x)),x))
            #loop thru all files and load their signals
            for f in sorted_files:
                    file_path = os.path.join(dirpath, f)
                    signal, sr =librosa.load(file_path,sr=sample_rate)

                    Spectogram = np.abs(librosa.stft(signal))
                    X_decibels = librosa.amplitude_to_db(Spectogram)
                    shape_spec= X_decibels.shape[1]
                    to_be_subtracted=shape_spec-256
                    X_decibels=X_decibels[:,:-to_be_subtracted]
                    
                    if (X_decibels.shape[1]!=0):
                        scaler.fit(X_decibels.T)
                        norm_x = scaler.transform(X_decibels.T).T
                  
                    if j not in lost:
                        output_spectrogram[j, :, :] = norm_x    
                    #if (norm_mfcc.shape[0]>0):
                    j=j+1
                    
                    

                
                

   
    np.save("input_spectrogram.npy", input_spectrogram)
    np.save("output_spectrogram.npy", output_spectrogram)



save_spectograms(input_dataset,output_dataset)
    


   

 