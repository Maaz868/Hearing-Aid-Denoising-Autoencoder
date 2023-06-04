import tensorflow as tf
import numpy as np
from autoencoder import Autoencoder
from sklearn.model_selection import train_test_split
import soundfile as sf
from sklearn.preprocessing import MinMaxScaler
import librosa


learning_rate= 0.001
batch_size=32


def load_data(input_spectrogram,output_spectrogram):
    input = np.load(input_spectrogram)
    output=np.load(output_spectrogram)
    return input, output

def train(x_train,mfcc,x_test,y_test,learning_rate,batch_size):
    

    autoencoder = Autoencoder(
        input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]),
        filters=(64,128,256),
        kernels=((5,5),(5,5),(5,5)),
        strides=((2,2),(2,2),(2,2)),
        latent_dim=256
    )
    autoencoder.input_shape = (x_train.shape[1],x_train.shape[2],x_train.shape[3])
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    autoencoder.train(x_train,mfcc,x_test,y_test,batch_size,50)
    autoencoder.save("model","parameter50.pkl","weights50.h5")
    return autoencoder




x,y = load_data("input_spectrogram.npy","output_spectrogram.npy")
x_train, x_test,y_train, y_test = train_test_split(x,y,test_size=0.25)

    

x_train = x_train[...,np.newaxis]
y_train = y_train[...,np.newaxis]
x_train = x_train[:,:-1,:,:]
y_train = y_train[:,:-1,:,:]
print(x_train.shape)
print(y_train.shape)

x_test = x_test[...,np.newaxis]
y_test = y_test[...,np.newaxis]
x_test = x_test[:,:-1,:,:]
y_test = y_test[:,:-1,:,:]
print(x_test.shape)
print(y_test.shape)
print(x_train[12])
print("-"*90)
print(y_train[12])

filename="C:\\Users\\Maaz Ahmed\\Desktop\\dataset3\\newclean\\heavy11output.wav"

x , sr = librosa.load(filename,sr=16000)
Spectogram = np.abs(librosa.stft(x))
X_decibels = librosa.amplitude_to_db(Spectogram)

#resize
shape_spec= X_decibels.shape[1]
to_be_subtracted=shape_spec-256
# X_decibels=X_decibels[:,:-to_be_subtracted]
X_decibels = X_decibels[:-1,:-to_be_subtracted]
print(X_decibels.shape)

#normalize
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(X_decibels.T)
X_decibels = scaler.transform(X_decibels.T).T

pred = x_train[999]
pred = np.squeeze(pred)

#denormalize
Xdb_denorm = scaler.inverse_transform(pred.T).T

#convert to audio
S2 = librosa.db_to_amplitude(Xdb_denorm)
y_inv = librosa.griffinlim(S2)
sf.write('reconstructed_1audio.wav', y_inv, 16000)

pred = y_train[999]
pred = np.squeeze(pred)

#denormalize
Xdb_denorm = scaler.inverse_transform(pred.T).T

#convert to audio
S2 = librosa.db_to_amplitude(Xdb_denorm)
y_inv = librosa.griffinlim(S2)
sf.write('reconstructed_2audio.wav', y_inv, 16000)

# autoencoder = train(x_train,y_train,x_test,y_test,learning_rate,batch_size)











