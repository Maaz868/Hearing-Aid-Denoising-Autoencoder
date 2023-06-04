
from pydub import AudioSegment
import librosa
import os
import json
import soundfile as sf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from autoencoder import Autoencoder
from sklearn.preprocessing import MinMaxScaler


filename="C:\\Users\\Maaz Ahmed\\project\\test.wav"
    

#load the file
x , sr = librosa.load(filename,sr=16000)
Spectogram = np.abs(librosa.stft(x))
X_decibels = librosa.amplitude_to_db(Spectogram)

Spectogram =  Spectogram[::2]

#resize
shape_spec= X_decibels.shape[1]
to_be_subtracted=shape_spec-256
X_decibels = X_decibels[:-1,:-to_be_subtracted]


#normalize
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(X_decibels.T)
X_decibels = scaler.transform(X_decibels.T).T



#load the model
autoencoder = Autoencoder.load("model","parameter50.pkl","weights50.h5")

#reshape for prediction
X_decibels = X_decibels[...,np.newaxis]
X_decibel = np.tile(X_decibels, (1, 1, 1, 1))

#predict
pred = autoencoder.model.predict(X_decibel)
pred = pred[0]
pred = np.squeeze(pred)

#denormalize
Xdb_denorm = scaler.inverse_transform(pred.T).T

#convert to audio
S2 = librosa.db_to_amplitude(Xdb_denorm)
y_inv = librosa.griffinlim(S2)
sf.write('reconstructed_audio.wav', y_inv, sr)



# if np.isnan(S2).any() or np.isinf(S2).any():
#     # Handle NaN or infinite values
#     mean_val = np.nanmean(S2)
#     print("hi1")
#     S2 = np.nan_to_num(S2, nan=mean_val, posinf=mean_val, neginf=mean_val)
#     print(np.isfinite(S2))
#     S2 = np.where(np.isinf(S2), 0, S2)
#     zero_indices = np.where(S2 == 0)
#     print(S2)
#     print("hello")
#     abc = librosa.griffinlim(S2)
#     sf.write('reconstructed_1audio.wav', abc, sr)

#     # if np.isnan(librosa.griffinlim(S2)).any() or np.isinf(librosa.griffinlim(S2)).any():
#     #         mean_val = np.nanmean(librosa.griffinlim(S2))
#     #         y_inv = np.nan_to_num(librosa.griffinlim(S2), nan=mean_val, posinf=mean_val, neginf=mean_val)
#     #         abc = librosa.griffinlim(y_inv)
#     #         sf.write('reconstructed_1audio.wav', abc, sr)
#     # else:
#     #     print("error")
#     # # y_inv = librosa.griffinlim(S2)
#     # # sf.write('reconstructed_1audio.wav', y_inv, sr)
#     # print('yes')
# else:
#     #mean_val = np.nanmean(spec)
#     print("hi2")
    
# #y_inv = librosa.griffinlim(S2)
# #sf.write('reconstructed_1audio.wav',data=y_inv, sr=sr)
# # shape_spec= X_decibels.shape[1]
# # to_be_subtracted=shape_spec-256
# # X_decibels=X_decibels[:,:-to_be_subtracted]

# # scaler = MinMaxScaler(feature_range=(-1, 1))
# # X_norm = scaler.fit_transform(X_decibels.T).T
# # print(X_norm.shape)
# # # # normalize the data
# # # Xdb_norm = scaler.transform(X_decibels)

# # # to denormalize, use the inverse_transform method
# # Xdb_denorm = scaler.inverse_transform(X_norm.T).T

# # #print(Xdb_denorm)

# # S2 = librosa.db_to_amplitude(Xdb_denorm)
# # y_inv = librosa.griffinlim(S2)
# # sf.write('reconstructed_1audio.wav', y_inv, sr)