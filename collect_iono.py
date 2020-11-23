import glob
import h5py
import numpy as n

fl=glob.glob("iono*.h5")
data=[]
labels=[]
for f in fl:
    h=h5py.File(f,"r")
    data.append(n.copy(h["img"].value))
    ph=0.0
    pf=0.0
    hmf=n.copy(h["hmf"].value)
    muf=n.copy(h["muf"].value)
    print(hmf)
    print(muf)
    if hmf > 0.0:
        ph=1.0
    if muf > 0.0:
        pf=1.0
    labels.append(n.array([hmf,muf,ph,pf]))
    h.close()
data=n.array(data,dtype=n.uint8)
labels=n.array(labels)

ho=h5py.File("all.h5","w")
ho["data"]=data
ho["labels"]=labels
ho.close()


        
