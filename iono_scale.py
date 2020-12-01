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

label_model=tf.keras.models.load_model("model/label")
f_model=tf.keras.models.load_model("model/f_scale")
e_model=tf.keras.models.load_model("model/e_scale")

#
# these are test images never shown to the training algorithm
#
fl=glob.glob("sod_ski/*.png")
fl.sort()
n.random.shuffle(fl)
show_im=False


fof2s=[]
fes=[]
hmfs=[]
hes=[]
t=[]
n_imgs_per_batch=64
n_batches=int(len(fl)/n_imgs_per_batch)
imgs=n.zeros([n_imgs_per_batch,200,498],dtype=n.float32)

hlut=h5py.File("sod_ski/lut.h5","r")
img_rgs=n.copy(hlut[("img_rgs")])
img_freqs=n.copy(hlut[("img_freqs")])

ridx=n.arange(len(img_rgs))
ridx[0]=-100000
ridx[len(img_rgs)-1]=100000
fidx=n.arange(len(img_freqs))
fidx[0]=-100000
fidx[len(img_freqs)-1]=100000
rint=sint.interp1d(ridx,img_rgs)
fint=sint.interp1d(fidx,img_freqs)
plt.plot(img_rgs)
plt.show()

plt.plot(img_freqs)
plt.show()

hlut.close()
fi=0
for bi in range(n_batches):
    for i in range(n_imgs_per_batch):
        t0=re.search("iono-(.*).png",fl[fi]).group(1)
        t.append(float(t0))
        im=n.array(imageio.imread(fl[fi]),dtype=n.float32)/255.0
#        print(imgs.shape)
 #       print(im.shape)
        imgs[i,:,:]=im

        fi+=1
    imgst=tf.reshape(imgs,[n_imgs_per_batch,200,498,1])
    label_pr=label_model.predict(imgst)
    f_pr=f_model.predict(imgst)
    e_pr=e_model.predict(imgst)
    
    for i in range(n_imgs_per_batch):
        hmf=0.0
        fof2=0.0
        he=0.0
        fe=0.0
        label_str=""
        if label_pr[i,0]>0.8:
            label_str+="F-region "
        else:
            label_str+="         "
        if label_pr[i,1]>0.8:
            label_str+="E-region "
        else:
            label_str+="         "
        print(label_str)
        #    print("E-region %1.2f F-region %1.2f"%(label_pr[0,1],label_pr[0,0]))
        if label_pr[i,0]>0.8:
            fof2=fint(f_pr[i,0])
            hmf=rint(f_pr[i,1])
            # f-region
            
        if label_pr[i,1]>0.8:
            # e-region
            fe=fint(e_pr[i,0])
            he=rint(e_pr[i,1])
        fes.append(fe)
        hes.append(he)
        fof2s.append(fof2)
        hmfs.append(hmf)    
        print("%1.2f %1.2f %1.2f %1.2f"%(fof2,fe,hmf,he))
        if show_im:
            plt.figure(figsize=(1.5*8,1.5*6))
            plt.imshow(imgs[i,:,:,0])
            if fof2 > 0:
                plt.axvline(fof2,color="red",alpha=0.5)
                plt.axhline(hmf,color="green",alpha=0.5)
            if he > 0.0:
                plt.axvline(fe,color="blue",alpha=0.5)
                plt.axhline(he+5,color="white",alpha=0.5)
                plt.axhline(he-5,color="white",alpha=0.5)
                
            plt.tight_layout()
            plt.show()


plt.plot(t,fof2s,".")
plt.plot(t,fes,".")
plt.show()

plt.plot(t,hmfs,".")
plt.plot(t,hes,".")
plt.show()
