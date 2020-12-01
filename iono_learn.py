import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import h5py
import numpy as n
import matplotlib.pyplot as plt

import ionogram_data as igd

import time
from tensorflow.python.keras.callbacks import TensorBoard

def teach_network(n_type="label",bs=64, n_epochs=2,N=10,plot=True):
    # scalings:
    # fof2, fe, h'mf, h'e
    if n_type == "label":
        dataset=igd.get_ionogram_data(dirname="./sod_ski",N=N,shift=True,bs=bs,fr0=0.0,fr1=0.8,prob=True)
        validation_dataset=igd.get_ionogram_data(dirname="./sod_ski",N=1,shift=False,bs=bs,fr0=0.8,fr1=1.0,prob=True)
    if n_type == "f_scale":
        # find ionograms with an f-region
        dataset=igd.get_ionogram_data(dirname="./sod_ski",N=N,shift=True,bs=bs,fr0=0.0,fr1=0.8,region="f")
        validation_dataset=igd.get_ionogram_data(dirname="./sod_ski",N=1,shift=False,bs=bs,fr0=0.8,fr1=1.0,region="f")
    if n_type == "e_scale":
        # find ionograms with an e-region
        dataset=igd.get_ionogram_data(dirname="./sod_ski",N=N,shift=True,bs=bs,fr0=0.0,fr1=0.8,region="e")
        validation_dataset=igd.get_ionogram_data(dirname="./sod_ski",N=1,shift=False,bs=bs,fr0=0.8,fr1=1.0,region="e")
    
    # multi-gpu
    ms=tf.distribute.MirroredStrategy()
    # multi-gpu
    with ms.scope():
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(64, (3, 3), strides=(2,2), activation='relu', input_shape=(200, 498,1)),
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
            model.add(tf.keras.layers.Dense(2,activation="sigmoid"))
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

    model.save("model/%s"%(n_type))

    if plot:
        n_input=len(validation_dataset)
        for i in range(n_input):
            im,sc=validation_dataset[i]
            pr=model.predict(im)
            for j in range(bs):
                print(pr[j,:])
                plt.imshow(im[j,:,:])
                if n_type == "f_scale" or n_type == "e_scale":
                    plt.axvline(pr[j,0],color="red")
                    plt.axhline(pr[j,1],color="red")
                if n_type == "label":
                    label_str=""
                    if pr[j,0]> 0.8:
                        label_str+="F "
                    if pr[j,1]> 0.8:
                        label_str+="E "
                    plt.title(label_str)
                plt.show()
            
# three networks to solve all problems
# 1) determine the presence of F and E traces
teach_network(n_type="label",bs=64, n_epochs=10, N=20,plot=False)
# 2) scale f-region trace h'f and fof2
teach_network(n_type="f_scale",bs=32, n_epochs=20,N=20,plot=False)
# 3) scale e-region trace h'e and fe
teach_network(n_type="e_scale",bs=32, n_epochs=20,N=20,plot=False)
