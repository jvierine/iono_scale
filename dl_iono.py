import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import h5py
import numpy as n
import matplotlib.pyplot as plt

h=h5py.File("all.h5","r")
imgs=tf.convert_to_tensor(n.copy(h["data"].value))
imgs=tf.reshape(imgs,[-1,200,498,1])
print(imgs.shape)
labels=tf.convert_to_tensor(n.copy(h["labels"].value))
labels=tf.reshape(labels,[-1,4])
h.close()
if False:
    for i in range(10):
        plt.imshow(imgs[i,:,:])
        plt.axhline(labels[i,0])
        plt.axvline(labels[i,1])
        plt.title(labels[i])
        plt.show()
    data=n.array(data,dtype=n.float32)/255.0
print(imgs.shape)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 498,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(4, activation='relu') # hmf, muf, ph pf
])

model.compile(loss='mean_absolute_error',
              optimizer=tf.keras.optimizers.Adam(0.001))
model.summary()

for i in range(10):
    history = model.fit(imgs,
                        labels,
                        validation_split=0.2,
                        epochs=100)

