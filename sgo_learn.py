import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import h5py
import numpy as n
import matplotlib.pyplot as plt

import sgo_data as igd

import time
#from tensorflow.python.keras.callbacks import TensorBoard

def teach_network(n_type="label",bs=32, n_epochs=2,plot=True):
    # scalings:
    # fof2, fe, h'mf, h'e
    
    if n_type == "label":
        dataset=igd.sgo_normalized_data(prob=True,batch_size=bs,fr0=0.0,fr1=0.8)
        validation_dataset=igd.sgo_normalized_data(prob=True,batch_size=bs,fr0=0.8,fr1=1.0)
    if n_type == "f2":
        # find ionograms with an f2-region
        dataset=igd.sgo_normalized_data(prob=False,batch_size=bs,fr0=0.0,fr1=0.8,output_type="f2")
        validation_dataset=igd.sgo_normalized_data(prob=False,batch_size=bs,fr0=0.8,fr1=1.0,output_type="f2")
    if n_type == "es":
        # find ionograms with an f2-region
        dataset=igd.sgo_normalized_data(prob=False,batch_size=bs,fr0=0.0,fr1=0.8,output_type="es")
        validation_dataset=igd.sgo_normalized_data(prob=False,batch_size=bs,fr0=0.8,fr1=1.0,output_type="es")
    if n_type == "e":
        # find ionograms with an f2-region
        dataset=igd.sgo_normalized_data(prob=False,batch_size=bs,fr0=0.0,fr1=0.8,output_type="e")
        validation_dataset=igd.sgo_normalized_data(prob=False,batch_size=bs,fr0=0.8,fr1=1.0,output_type="e")

    
    # multi-gpu
    ms=tf.distribute.MirroredStrategy()
    # multi-gpu
    with ms.scope():
        model = tf.keras.models.Sequential([
            #(262, 295)(174, 295)
            tf.keras.layers.Conv2D(64, (3, 3), strides=(1,1), activation='relu', input_shape=(174, 295,1)),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),            
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024,activation="relu"),
            tf.keras.layers.Dense(1024,activation="relu"),
        ])
        if n_type == "label":
            model.add(tf.keras.layers.Dense(dataset.n_pars,activation="sigmoid"))
        elif n_type == "f_scale" or n_type == "e_scale":
            model.add(tf.keras.layers.Dense(2,activation="relu"))
        else:
            print("n_type not recognized. exiting")
            exit(0)

    #
    # probability of feature in image: 'binary_crossentropy'
    #
    # regression: "mse"
    #
    if n_type == "label":
        model.compile(loss="binary_crossentropy",
                      optimizer=tf.keras.optimizers.Adam())
    else:
        model.compile(loss="mse",
                      optimizer=tf.keras.optimizers.Adam())
        
    model.summary()

    history = model.fit(dataset,
                        batch_size=bs,
                        validation_data=validation_dataset,
                        epochs=n_epochs)

    model.save("sgo_model/%s"%(n_type))

            
# three networks to solve all problems
# 1) determine the presence of F and E traces
#teach_network(n_type="label",bs=32, n_epochs=10)
# 2) scale f-region trace h'f and fof2
teach_network(n_type="f2",bs=32, n_epochs=10,plot=False)
# 3) scale e-region trace h'es and fes
#teach_network(n_type="es",bs=32, n_epochs=10,plot=False)
# 4) scale e-region trace h'e and fe
#teach_network(n_type="e",bs=32, n_epochs=10,plot=False)

