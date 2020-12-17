import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import h5py
import numpy as n
import matplotlib.pyplot as plt
import glob
import imageio
import re
import os
import scipy.interpolate as sint
import sgo_data as igd

label_model=tf.keras.models.load_model("sgo_model/label")
f2_model=tf.keras.models.load_model("sgo_model/f2")
f1_model=tf.keras.models.load_model("sgo_model/f1")
es_model=tf.keras.models.load_model("sgo_model/es")
e_model=tf.keras.models.load_model("sgo_model/e")

d=igd.sgo_normalized_data(prob=False,batch_size=32,fr0=0.0,fr1=1.0,output_type="all")

labels=["Es","E","F1","F2"]

for i in range(len(d)):
    
    imgs,scls=d[i]
    
    pr=label_model.predict(imgs)
    f2_pr=f2_model.predict(imgs)
    f1_pr=f1_model.predict(imgs)
    es_pr=es_model.predict(imgs)
    e_pr=e_model.predict(imgs)
    
    for j in range(32):
        gidx=n.where(pr[j,:]>0.9)[0]
        label_str=""
        for gi in gidx:
            label_str+=labels[gi]+" "
        plt.figure(figsize=(14,9))
        plt.imshow(imgs[j,:,:],extent=(d.xlim[0],d.xlim[1],d.ylim[0],d.ylim[1]),aspect="auto")
        plt.colorbar()
        if pr[j,2] > 0.8 or pr[j,3] > 0.8:
 #           plt.axhline(f2_pr[j,0]*100.0,color="red")
            plt.axvline(f2_pr[j,0],color="red")
 #       if pr[j,0] > 0.9:
#            plt.axvline(es_pr[j,0],color="white")
#            plt.axvline(es_pr[j,1],color="white")
        if pr[j,1] > 0.8:
#            plt.axhline(e_pr[j,0]*100.0,color="blue")
            plt.axvline(e_pr[j,0],color="blue")
        if pr[j,2] > 0.8:
  #          plt.axhline(f1_pr[j,0]*100.0,color="green") # h'f
            plt.axvline(f1_pr[j,0],color="green")       # fof1
#            plt.axvline(f1_pr[j,2],color="green")       # fof2
            
        print("prediction")
        print(f2_pr[j,:])
        print("true")
        print(scls[j,:])
        plt.title(label_str)
        plt.show()

