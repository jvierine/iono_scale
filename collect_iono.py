import glob
import h5py
import numpy as n

fl=glob.glob("iono*.h5")
data=[]
labels=[]
ok_labels=[]
n_ok=0
n_bad=0
for f in fl:
    h=h5py.File(f,"r")

    ph=0.0
    pf=0.0
    hmf=n.copy(h["hmf"].value)
    muf=n.copy(h["muf"].value)

    if hmf > 0.0:
        ph=1.0
    if muf > 0.0:
        pf=1.0

    ok=1
    if hmf>0.0 and muf > 0.0:
        n_ok+=1
    else:
        n_bad+=1
        ok=0
    if n_ok >= n_bad-10 and ok == 0:
        data.append(n.copy(h["img"].value))
        labels.append(n.array([hmf,muf,ph,pf]))
        ok_labels.append(ok)
    else:
        print("ok %d not adding bad example"%(ok))
        
    h.close()
    
data=n.array(data,dtype=n.uint8)
labels=n.array(labels)

ho=h5py.File("all.h5","w")
ho["data"]=data
ho["labels"]=labels
ho.close()


        
