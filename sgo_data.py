#!/usr/bin/env python3
import tensorflow as tf
import glob
import numpy as n
import scipy.io as sio
import re
import os
import matplotlib.pyplot as plt
import imageio

import sgo_reader as sd
import stuffr
import h5py

class sgo_ionograms:
    def __init__(self,dirname="/scratch/data/juha/sgo",dec=2, prob=True, plot=False):
        self.dirname=dirname
        #fl=glob.glob("%s/*/SO166GRAM*/SO166GRAM*/SO166GRAM*/*.166"%(dirname))
        # full hour
        fl=glob.glob("%s/*/SO166PARA*/SO166PARA*/SO166PARA*/*0000.mat"%(dirname))
        print("%d scaled ionograms found"%(len(fl)))
        
        self.fl=[]
        self.gfl=[]
        self.noise_rg0=450
        self.noise_rg1=525
        
        self.shape=(200,590)
        self.shape_learn=(int(525/dec),int(590/dec))
        self.plot=plot
        self.dec=dec
        self.prob=prob
        
        for f in fl:
            gf=self.ionogram_exists(f)
            if gf:
                self.gfl.append(gf)
                self.fl.append(f)
                
    def __len__(self):
        return(len(self.fl))

    def ionogram_exists(self,fname):
        r=re.search("(.*)/(....)/SO166PARA_(....)/SO166PARA_(......)/SO166PARA_(........)/SO166PARA_(........_..)(..)...mat",fname)
        if r:
            strs=r.groups()
            gram_fname0="%s/%s/SO166GRAM_%s/SO166GRAM_%s/SO166GRAM_%s/SO166GRAM_%s00%02d.166"%(strs[0],strs[1],strs[2],strs[3],strs[4],strs[5],0)
            gram_fname1="%s/%s/SO166GRAM_%s/SO166GRAM_%s/SO166GRAM_%s/SO166GRAM_%s00%02d.166"%(strs[0],strs[1],strs[2],strs[3],strs[4],strs[5],50)
            if os.path.exists(gram_fname0):
                return(gram_fname0)
            if os.path.exists(gram_fname1):
                return(gram_fname1)
            return None
    def __getitem__(self,i):
        ionogram=sd.loadGram(fn=self.gfl[i])
        parfile=sio.loadmat(self.fl[i])
        igd=ionogram["data"]

        for fi in range(igd.shape[1]):
            col=igd[:,fi]
            gidx=n.where(col>0)[0]
            bidx=n.where(col<10)[0]
            col[bidx]=n.nanmedian(col[gidx])
            
        igd[igd<40.0]=40.0
            
        for j in range(igd.shape[1]):

            noise_mean=n.nanmedian(igd[:,j])
            noise_pwr=n.nanstd(igd[:,j])
#            noise_pwr=n.nanmedian(n.abs(igd[:,j]-noise_mean))
            igd[:,j]=(igd[:,j]-noise_mean)/noise_pwr

        rgs=n.linspace(ionogram["ylim"][0],ionogram["ylim"][1],num=igd.shape[0])
        freqs=n.linspace(ionogram["xlim"][0],ionogram["xlim"][1],num=igd.shape[1])

        ridx=n.where((rgs > 0.0) & (rgs < 1000.0))[0]
        igd=igd[ridx,:]
        rgs=rgs[ridx]
        print(len(rgs))
   
        igd[igd>4.0]=4.0
        igd[igd<-3.0]=-3.0


        igd[n.isnan(igd)]=0.0
        igd[n.isinf(igd)]=0.0
        igd=igd-n.median(igd)
        igd[igd<0]=0.0
        igd=igd/n.max(igd)

        igd_d=n.zeros([igd.shape[0],int(igd.shape[1]/self.dec)],dtype=n.float32)
        for j in range(igd.shape[0]):
            igd_d[j,:]=stuffr.decimate(igd[j,:],dec=self.dec)
            
        igd_d2=n.zeros([int(igd_d.shape[0]/self.dec),igd_d.shape[1]],dtype=n.float32)
        for j in range(igd_d.shape[1]):
            igd_d2[:,j]=stuffr.decimate(igd_d[:,j],dec=self.dec)
            
        igd_d=igd_d2
        #print(igd_d.shape)
#        print(self.shape_learn)
#        if igd.shape != self.shape:
 #           print("wrong shape")
  #          exit(0)
            
  #      for pi in range(12):
   #         print("%s = %1.2f"%(parfile["ttl"][0][pi],parfile["V"][0][pi]))
#        print()            
 #       print(parfile["V"][0])
        if self.plot:
#            print(igd_d.shape)
            plt.figure(figsize=(12,8))
            plt.imshow(igd_d,origin="lower",extent=ionogram["xlim"]+[n.min(rgs),n.max(rgs)],aspect="auto",vmin=0,vmax=1)
            
            plt.colorbar()
            plt.axhline(parfile["V"][0][7],color="red")
            plt.axvline(parfile["V"][0][9],color="red")
            plt.axhline(parfile["V"][0][5],color="blue")
            plt.axvline(parfile["V"][0][6],color="blue")
            plt.axhline(parfile["V"][0][1],color="white")
            plt.axvline(parfile["V"][0][2],color="white")
            plt.title(self.fl[i])
            plt.show()

        scaling=n.array(parfile["V"][0][0:12],dtype=n.float32)
        scaling_prob=n.array(n.logical_not(n.isnan(scaling)),dtype=n.float32)
        scaling[n.isnan(scaling)]=0.0
        
        return(igd_d,scaling,scaling_prob,self.fl[i],ionogram["xlim"],[n.min(rgs),n.max(rgs)])

def preprocess(dirname="/scratch/data/juha/sgo/pre/",plot=True):
    
    s=sgo_ionograms(plot=plot)
    #print(len(s))
    idx=n.arange(len(s))
    
    #    n.random.shuffle(idx)
    
    for i in range(len(s)):
        try:
            igr,scl,sclp,pf,xl,yl=s[idx[i]]
            print(pf)
            pref=re.search(".*/(SO166PARA_.*).mat",pf).group(1)
            ofn="%s/%s.h5"%(dirname,pref)
            pofn="%s/%s.png"%(dirname,pref)
            imageio.imwrite(pofn,igr[::-1])
            ho=h5py.File(ofn,"w")
            #ho["iono"]=igr
            ho["scaling"]=scl
            ho["scaling_p"]=sclp
            ho["fname"]=pf
            ho["xlim"]=xl
            ho["ylim"]=yl
            ho.close()
            print(pref)
        except:
            print("Failed %d"%(i))



class sgo_normalized_data(tf.keras.utils.Sequence):
    """
    generate data on the fly with random x shifts
    """
    def __init__(self,
                 dirname="/scratch/data/juha/sgo/pre",
                 batch_size=32,
                 prob=True,
                 output_type="all",
                 fr0=0.0,
                 fr1=1.0):#red)array([7,9],dtype=n.int)):
        
        # parameters
        # fmin, h'Es, foEs Type Es, fbEs h'E foE h'F foF1 foF2 fxI M(3000)F2
        # 0     1     2    3         4    5  6   7   8    9    10  11
        #
        # is there an F-region?
        # if 6 or 8 or 9
        # is there an Es
        # if 1 or 2 or 4
        # is there an E-region
        # if 5 or 6
        #
        fl=glob.glob("%s/*.h5"%(dirname))
        n.random.seed(0)
        n.random.shuffle(fl)
        self.batch_size=batch_size
        self.prob=prob
        self.par_fl=[]
        self.img_fl=[]
#        self.par_ids=par_ids
        if prob:
            self.n_pars=4 # es e and f?
        
        for f in fl:
            prefix=re.search("(.*/.*).h5",f).group(1)
            img_fname="%s.png"%(prefix)
            if os.path.exists(img_fname):

                if output_type=="all":
                    self.par_fl.append(f)
                    self.img_fl.append(img_fname)
                else:
                    hp=h5py.File(f,"r")
                    scaling=n.copy(hp["scaling"].value)
                    hp.close()
                    if output_type=="f2":
                        if scaling[7] > 0 and scaling[9]> 0:
                            # only ones that have foF2 and hf
                            self.par_fl.append(f)
                            self.img_fl.append(img_fname)
                            self.par_ids=n.array([7,9],dtype=n.int)
                            self.n_pars=2
                    if output_type=="es":
                        # only ones that have Es (fes and h'e)
                        if scaling[1] > 0 and scaling[2]> 0:
                            self.par_fl.append(f)
                            self.img_fl.append(img_fname)
                            self.par_ids=n.array([1,2],dtype=n.int)
                            self.n_pars=2
                    if output_type=="e":
                        # only ones that have fe and h'e
                        if scaling[5] > 0 and scaling[6]> 0:
                            self.par_fl.append(f)
                            self.img_fl.append(img_fname)
                            self.par_ids=n.array([5,6],dtype=n.int)
                            self.n_pars=2

        self.n_im=len(self.par_fl)
        print("found %d files"%(self.n_im))
        im=imageio.imread(self.img_fl[0])
        self.im_shape=im.shape
        h=h5py.File(self.par_fl[0],"r")
        self.ylim=h["ylim"].value
        self.xlim=h["xlim"].value
        h.close()
        self.fr0=fr0
        self.fr1=fr1

        self.max_idx=int(self.fr1*self.n_im)
        self.min_idx=int(self.fr0*self.n_im)
        
    def __len__(self):
        return(int((self.fr1-self.fr0)*self.n_im/float(self.batch_size)))
    
    def __getitem__(self,idx):
        i0=self.min_idx + self.batch_size*idx
#        i0=self.batch_size*idx
        
        imgs=n.zeros([self.batch_size,self.im_shape[0],self.im_shape[1]],dtype=n.float32)
        scales=n.zeros([self.batch_size,self.n_pars],dtype=n.float32)
        
        for i in range(self.batch_size):
            fi=(i0+i)%self.n_im
            fim=n.array(imageio.imread(self.img_fl[fi]),dtype=n.float32)/255.0
#            fim=fim-n.median(fim)
 #           fim[fim<0]=0.0
  #          fim=fim/n.nanmax(fim)
   #         fim[n.isnan(fim)]=0.0
            imgs[i,:,:]=fim
            h=h5py.File(self.par_fl[fi],"r")
            
        # parameters
        # fmin, h'Es, foEs Type Es, fbEs h'E foE h'F foF1 foF2 fxI M(3000)F2
        # 0     1     2    3         4    5  6   7   8    9    10  11
        #
        # is there an F2-region?
        # if 6 or 9
        # is there f1
        # if 8
        # is there an Es
        # if 1 or 2 or 4
        # is there an E-region
        # if 5 or 6
        #
            
            if self.prob:
                # 0 = Es? 1 = E 3 = F1 4=F2
                if h["scaling_p"].value[1] > 0 or h["scaling_p"].value[2] > 0 or h["scaling_p"].value[4] > 0:
                    scales[i,0]=1.0
                if h["scaling_p"].value[5] > 0 or h["scaling_p"].value[6] > 0:
                    scales[i,1]=1.0
                if h["scaling_p"].value[8] > 0:
                    scales[i,2]=1.0
                if h["scaling_p"].value[7] > 0 or h["scaling_p"].value[9] > 0:
                    scales[i,3]=1.0
            else:
                scales[i,:]=h["scaling"].value[self.par_ids]
                    
        imgs.shape=(imgs.shape[0],imgs.shape[1],imgs.shape[2],1)
        return(imgs,scales)


            
if __name__ == "__main__":
 #   preprocess(plot=False)
#    exit(0)
    d=sgo_normalized_data(prob=False)
    print(d.im_shape)
    imgs,scales=d[0]
    for i in range(imgs.shape[0]):
#        imgs[i,:,:,0]=imgs[i,:,:,0]-n.median(imgs[i,:,:,0])

        plt.imshow(imgs[i,:,:,0],extent=(d.xlim[0],d.xlim[1],d.ylim[0],d.ylim[1]),aspect="auto",vmin=0)
        plt.colorbar()

        # 
        # parameters
        # fmin, h'Es, foEs Type Es, fbEs h'E foE h'F foF1 foF2 fxI M(3000)F2
        # 0     1     2    3         4    5  6   7   8    9    10  11
        #
        plt.axvline(scales[i,9],color="red")
        plt.axhline(scales[i,7],color="red")        
        plt.axvline(scales[i,4],color="green")
        plt.axvline(scales[i,6],color="blue")
        plt.axvline(scales[i,8],color="white")
#        plt.axvline(scales[i,9],color="black")                
        plt.show()


#(262, 295)
#(174, 295)
