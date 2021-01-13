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
import ionogram_data as igd
tf.config.set_visible_devices([],'GPU')

def img_lut():
    """ 
    get funcions to convert pixel to frequency 
    - height and frequency are trained in pixels!
    """
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
    return(rint,fint,img_rgs,img_freqs)



def validate_f(plot=False):
    # pixel to freq and range functions
    rint,fint,img_rgs,img_freqs=img_lut()
    
    # read f-region training data
    vd=igd.get_ionogram_data(dirname="./sod_ski",N=1,shift=False,bs=32,fr0=0.8,fr1=1.0,region="f")
    # read f-region scaling model
    f_model=tf.keras.models.load_model("model3/f_scale")


    f_errs=[]
    h_errs=[]
    for bi in range(len(vd)):
        im,sc=vd[bi]
        f_pr=f_model.predict(im)
        for imi in range(32):

            ferr=(fint(sc[imi,0])-fint(f_pr[imi,0]))*1e3
            f_errs.append(ferr)
            rerr=rint(sc[imi,1])-rint(f_pr[imi,1])
            h_errs.append(rerr)
            if plot:
                plt.imshow(im[imi,:,:],
                           extent=[n.min(img_freqs),n.max(img_freqs),n.min(img_rgs),n.max(img_rgs)],
                           aspect="auto")
                
                plt.axhline(rint(f_pr[imi,1]),color="red",alpha=0.5,label="Predicted")
                plt.axvline(fint(f_pr[imi,0]),color="red",alpha=0.5)
                plt.axhline(rint(sc[imi,1]),color="green",alpha=0.5,label="True")
                plt.axvline(fint(sc[imi,0]),color="green",alpha=0.5)
                plt.legend()
                plt.title("ferr %1.2f kHz herr %1.2f (km)"%(ferr,rerr))
                plt.xlabel("Frequency (MHz)")
                plt.ylabel("Virtual range (km)")                
                plt.colorbar()
                plt.show()

    f_errs=n.array(f_errs)
    h_errs=n.array(h_errs)    
    rmse_f=n.sqrt(n.mean(f_errs**2.0))
    rmse_h=n.sqrt(n.mean(h_errs**2.0))
    rmedse_f=n.sqrt(n.median(f_errs**2.0))
    rmedse_h=n.sqrt(n.median(h_errs**2.0))
    print("RMSE freq %1.2f (kHz) range %1.2f (km) median %1.2f %1.2f"%(rmse_f,rmse_h,rmedse_f,rmedse_h))


def validate_e(plot=False):
    # pixel to freq and range functions
    rint,fint,img_rgs,img_freqs=img_lut()
    
    # read f-region validation data
    vd=igd.get_ionogram_data(dirname="./sod_ski",N=1,shift=False,bs=32,fr0=0.8,fr1=1.0,region="e")
    e_model=tf.keras.models.load_model("model3/e_scale")

    f_errs=[]
    h_errs=[]
    for bi in range(len(vd)):
        im,sc=vd[bi]
        f_pr=e_model.predict(im)
        for imi in range(32):
            ferr=(fint(sc[imi,0])-fint(f_pr[imi,0]))*1e3
            f_errs.append(ferr)
            rerr=rint(sc[imi,1])-rint(f_pr[imi,1])
            h_errs.append(rerr)
            if plot:
                plt.imshow(im[imi,:,:],
                           extent=[n.min(img_freqs),n.max(img_freqs),n.min(img_rgs),n.max(img_rgs)],
                           aspect="auto")
                plt.axhline(rint(f_pr[imi,1]),color="red",alpha=0.5,label="Predicted")
                plt.axvline(fint(f_pr[imi,0]),color="red",alpha=0.5)
                plt.axhline(rint(sc[imi,1]),color="green",alpha=0.5,label="True")
                plt.axvline(fint(sc[imi,0]),color="green",alpha=0.5)
                plt.legend()
                plt.title("ferr %1.2f kHz herr %1.2f (km)"%(ferr,rerr))
                plt.xlabel("Frequency (MHz)")
                plt.ylabel("Virtual range (km)")                
                plt.colorbar()
                plt.show()
                
    f_errs=n.array(f_errs)
    h_errs=n.array(h_errs)    
    rmse_f=n.sqrt(n.mean(f_errs**2.0))
    rmse_h=n.sqrt(n.mean(h_errs**2.0))
    rmedse_f=n.sqrt(n.median(f_errs**2.0))
    rmedse_h=n.sqrt(n.median(h_errs**2.0))
    print("RMSE freq %1.2f (kHz) range %1.2f (km) median %1.2f %1.2f"%(rmse_f,rmse_h,rmedse_f,rmedse_h))


def validate_label(plot=False,p_ok=0.8):
    rint,fint,img_rgs,img_freqs=img_lut()    
    # read f-region validation data (top 20%)
    vd=igd.get_ionogram_data(dirname="./sod_ski",N=1,shift=False,bs=32,fr0=0.8,fr1=1.0,prob=True)
    label_model=tf.keras.models.load_model("model3/label")

    f_ok=[]
    e_ok=[]
    for bi in range(len(vd)):
        im,sc=vd[bi]
        f_pr=label_model.predict(im)
        probs = f_pr>p_ok
        sc_probs = sc>p_ok
        
        for imi in range(32):
            if probs[imi,0] == sc_probs[imi,0]:
                f_ok.append(1)
            else:
                f_ok.append(0)
            if probs[imi,1] == sc_probs[imi,1]:
                e_ok.append(1)
            else:
                e_ok.append(0)
            if plot:
                plt.imshow(im[imi,:,:],
                           extent=[n.min(img_freqs),n.max(img_freqs),n.min(img_rgs),n.max(img_rgs)],
                           aspect="auto")
                
                plt.title("F %1.0f %% E %1.0f %%"%(100.0*f_pr[imi,0],100.0*f_pr[imi,1]))
                plt.xlabel("Frequency (MHz)")
                plt.ylabel("Virtual range (km)")                
                plt.colorbar()
                plt.show()
                
    f_acc=n.sum(f_ok)/float(len(f_ok))
    e_acc=n.sum(e_ok)/float(len(e_ok))
    print("Accuracy F %1.2f E %1.2f"%(f_acc,e_acc))

#validate_e(plot=False)    
#validate_f(plot=False)        
validate_label(plot=False)    





