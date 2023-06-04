import tensorflow as tf
import numpy as np
import os
import pickle

class Autoencoder:
    
    def __init__(self,input_shape,filters,kernels,strides,latent_dim):
        
        self.input_shape=input_shape #[1,2,3] width, height and number of kernels
        #lists containing components for each layer
        self.filters=filters
        self.kernels=kernels
        self.strides=strides
        self.latent_dim=latent_dim

        self.encoder=None
        self.decoder=None
        self.model=None

        self.num_layers=len(filters)
        self.shape_before_bottleneck=None
        self.model_input=None

        #self.build()
        self.build_encoder()
        self.build_decoder()
        self.build_autoencoder()
        
    def summary(self):
        self.encoder.summary()
        print("-------------"*20)
        self.decoder.summary()
        print("-------------"*20)
        self.model.summary()
    

# --------------------------------------- AUTOENCODER ---------------------------------------

    def build_autoencoder(self):
        model_input = self.model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = tf.keras.Model(model_input,model_output)

    def compile(self, learning_rate=0.0001):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        mse_loss = tf.keras.losses.MeanSquaredError()
        self.model.compile(optimizer=optimizer,loss=mse_loss, metrics=['accuracy'])
    
    def train(self, x_train,mfcc,x_test,y_test, batch_size, num_epochs):
        self.model.fit(x_train,mfcc,validation_data=(x_test,y_test),batch_size=batch_size,epochs=num_epochs,shuffle=True)
        


# --------------------------------------- DECODER ---------------------------------------

    def build_decoder(self): #reverse of encoder steps
        decoder_input = self.add_decoder_input()
        dense_layer = self.add_dense_layer(decoder_input)
        reshape_layer = self.add_reshape_layer(dense_layer)
        conv_transpose_layers = self.add_conv_transpose_layers(reshape_layer)
        decoder_output = self.add_decoder_output(conv_transpose_layers)
        self.decoder = tf.keras.Model(decoder_input,decoder_output,name="decoder")

    def add_decoder_input(self):
        return tf.keras.layers.Input(shape=self.latent_dim, name="decoder_input")
    
    def add_dense_layer(self,decoder_input):
        num_neurons = np.prod(self.shape_before_bottleneck)
        dense_layer= tf.keras.layers.Dense(num_neurons,name="decoder_dense")(decoder_input)
        return dense_layer
    
    def add_reshape_layer(self, dense_layer):
        return tf.keras.layers.Reshape(self.shape_before_bottleneck)(dense_layer)
    
    def add_conv_transpose_layers(self,x):
        for layer in reversed(range(1,self.num_layers)):
            x=self.add_conv_transpose_layer(layer, x)
        return x
    
    def add_conv_transpose_layer(self,layer, x):
        conv_transpose_layer=tf.keras.layers.Conv2DTranspose(
            filters=self.filters[layer],
            kernel_size=self.kernels[layer],
            strides=self.strides[layer],
            padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            name=f"decoder_convtranspose_layer_{self.num_layers-layer}"
        )
        x = conv_transpose_layer(x)
        x = tf.keras.layers.ReLU(name=f"decoder_relu_{self.num_layers-layer}")(x)
        x = tf.keras.layers.BatchNormalization(name=f"decoder_bn_{self.num_layers-layer}")(x)
        return x
    
    def add_decoder_output(self,x):
        conv_transpose_layer=tf.keras.layers.Conv2DTranspose(
            filters=1,
            kernel_size=self.kernels[0],
            strides=self.strides[0],
            padding="same",
            name=f"decoder_convtranspose_layer_{self.num_layers}"
        )

        x = conv_transpose_layer(x)
        output_layer = tf.keras.layers.Activation("sigmoid",name="output_sigmoid_layer")(x)
        return x
    

    



# --------------------------------------- ENCODER ---------------------------------------

    def build_encoder(self):
        encoder_input=self.add_encoder_input()
        self.model_input=encoder_input
        conv_layers=self.add_conv_layers(encoder_input)
        bottleneck=self.add_bottleneck(conv_layers)
        self.encoder=tf.keras.Model(encoder_input,bottleneck,name="encoder")

    def add_encoder_input(self):
        return tf.keras.layers.Input(shape=self.input_shape,name="encoder_input")
    
    def add_conv_layers(self,encoder_input):
        x=encoder_input
        for layer in range(self.num_layers):
            x=self.add_conv_layer(layer,x)
        return x
    
    def add_conv_layer(self, layer,x):
        conv_layer = tf.keras.layers.Conv2D(
            filters=self.filters[layer],
            kernel_size=self.kernels[layer],
            strides=self.strides[layer],
            padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            name=f"encoder_conv_layer_{layer+1}"
        )

        x=conv_layer(x)
        x = tf.keras.layers.ReLU(name=f"encoder_relu_{layer+1}")(x)
        x = tf.keras.layers.BatchNormalization(name=f"encoder_bn_{layer+1}")(x)

        return x
    
    def add_bottleneck(self,x):
        #this shape will be used in decoder
        self.shape_before_bottleneck= tf.keras.backend.int_shape(x)[1:] #ignore batch size from 4 dimensional array, [1,2,3,4] take only 2,3,4
        x=tf.keras.layers.Flatten()(x)
       # x = tf.keras.layers.Dropout(0.05)(x)
        x=tf.keras.layers.Dense(self.latent_dim,name=f"encoder_output")(x)      
        return x



# --------------------------------------- MODEL SAVE AND LOAD ---------------------------------------

    def save(self, folder=".",parameter=".",weights="."):
        self.create_folder(folder)
        self.save_parameters(folder,parameter)
        self.save_weights(folder,weights)

    def create_folder(self,folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    def save_parameters(self, folder,parameter):
        parameters = [
            self.input_shape,
            self.filters,
            self.kernels,
            self.strides,
            self.latent_dim
        ]
        path = os.path.join(folder,parameter)
        with open(path,"wb") as f:
            pickle.dump(parameters, f) 

    def save_weights(self,folder,weights):
        path = os.path.join(folder,weights)
        self.model.save_weights(path)

    def load_weights(self,weights):
        self.model.load_weights(weights)

    @classmethod
    def load(cls, folder,parameter,weights):
        parameters=os.path.join(folder,parameter)
        weights=os.path.join(folder,weights)
        with open(parameters, "rb") as f:
            parameters = pickle.load(f)
        autoencoder = Autoencoder(*parameters)
        autoencoder.load_weights(weights)
        return autoencoder
     









