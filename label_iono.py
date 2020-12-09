import imageio
import matplotlib.pyplot as plt
import glob
import sys
import h5py
import os

data_dir="./sod_ski"
fl=glob.glob("%s/*.png"%(data_dir))
fl.sort()

review=False

for f in fl:
    if os.path.exists("%s.h5"%(f)):
        print("scaling already exists. skipping")
        if review:
            pfname="%s.h5"%(f)
            print(pfname)
            ho=h5py.File(pfname,"r")
#            im=ho["img"].value
            hmf=ho["hmf"].value
            muf=ho["muf"].value
            mue=ho["mue"].value
            he=ho["he"].value
            plt.figure(figsize=(18,12))
            plt.imshow(im)
            plt.title(pfname)
            plt.axhline(hmf,color="green")
            if he > 0.0:
                plt.axhline(he+10,color="blue")
                plt.axhline(he-10,color="blue")
            plt.axvline(muf,color="red")
            plt.axvline(mue,color="white")
            plt.savefig("%s_review.jpg"%(pfname))
            plt.close()
            plt.clf()
            ho.close()
        continue
    im = imageio.imread(f)
    hmf=0.0
    muf=0.0
    mue=0.0
    he=0.0
    def press(event):
        global hmf,muf, mue, he
        x, y = event.xdata, event.ydata
        print("press %f %f"%(x,y))
        sys.stdout.flush()
        if event.key == '1':
            muf=x
            ax.axvline(muf,color="red")
            fig.canvas.draw()
        if event.key == '2':
            hmf=y
            ax.axhline(hmf,color="green")
            fig.canvas.draw()
        if event.key == '3':
            mue=x
            ax.axvline(mue,color="blue")
            fig.canvas.draw()
        if event.key == '4':
            he=y
            ax.axhline(he,color="white")
            fig.canvas.draw()
        if event.key == '5':
            he=0.0
            mue=0.0
            hmf=0.0
            muf=0.0
            fig.canvas.draw()
            
        if event.key == '9':
            ofname="%s.h5"%(f)
            print("saving %s %f %f %f %f"%(ofname,hmf,muf,he,mue))
            ho=h5py.File(ofname,"w")
#            ho["img"]=im
            ho["hmf"]=hmf
            ho["muf"]=muf
            ho["mue"]=mue
            ho["he"]=he
            ho.close()
            plt.close()
        if event.key == '0':            
            ofname="%s.h5"%(f)
            print("skiping %s %f %f"%(ofname,hmf,muf))
#            ho=h5py.File(ofname,"w")
 #           ho["img"]=im
  #          ho["hmf"]=0.0
   #         ho["muf"]=0.0
    #        ho.close()
            plt.close()
            
    fig, ax = plt.subplots(figsize=(18,12))
    fig.canvas.mpl_connect('key_press_event', press)
    ax.imshow(im)
    plt.title("1) hmf 2) muf 3) fe 4) he 9) save 0) skip")
    plt.show()
