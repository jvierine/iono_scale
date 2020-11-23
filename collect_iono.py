import glob
import h5py
import numpy as n

fl=glob.glob("iono*.h5")
n.random.shuffle(fl)
data=[]
ok_data=[]
scaling=[]
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

    ok=0
    if hmf>0.0 and muf > 0.0:
        ok=1
        scaling.append(n.array([hmf,muf]))
        ok_data.append(n.copy(h["img"].value))        
    else:
        ok=0
        
    if ok==1 or (n_ok >= (n_bad-10)):
        if ok == 1:
            n_ok+=1
        else:
            n_bad+=1
        print("adding %d"%(ok))
        data.append(n.copy(h["img"].value))
        ok_labels.append(ok)
    
    else:
        print("ok %d not adding bad example n_ok %d n_bad %d"%(ok,n_ok,n_bad))
        
    h.close()
    
data=n.array(data,dtype=n.uint8)
ok_data=n.array(ok_data,dtype=n.uint8)
ok_labels=n.array(ok_labels)
scaling=n.array(scaling)

ho=h5py.File("all.h5","w")
ho["data"]=data
ho["ok_data"]=ok_data
ho["scaling"]=scaling
ho["ok"]=ok_labels
ho.close()


        
