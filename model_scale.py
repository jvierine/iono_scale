import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import h5py
import numpy as n
import matplotlib.pyplot as plt
import glob
import imageio

model=tf.keras.models.load_model("model/scaler")

#
# these are test images never shown to the training algorithm
#
fl=glob.glob("test/*.png")

for f in fl:
    im=n.array(imageio.imread(f),dtype=n.float32)
    img=tf.convert_to_tensor(im)
    img=tf.reshape(img,[-1,200,498,1])
    pr=model.predict(img)
    plt.imshow(im)
    plt.axhline(pr[0][0],color="red")
    plt.axvline(pr[0][1],color="red")
    plt.show()
    print(pr)


