import imageio
import matplotlib.pyplot as plt
import glob
import sys
import h5py

fl=glob.glob("*.png")
fl.sort()

for f in fl:
    im = imageio.imread(f)
    hmf=0.0
    muf=0.0
    def press(event):
        global hmf,muf
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
            ofname="%s.h5"%(f)
            print("saving %s %f %f"%(ofname,hmf,muf))
            ho=h5py.File(ofname,"w")
            ho["img"]=im
            ho["hmf"]=hmf
            ho["muf"]=muf
            ho.close()
            plt.close()
        if event.key == '4':            
            ofname="%s.h5"%(f)
            print("skiping %s %f %f"%(ofname,hmf,muf))
#            ho=h5py.File(ofname,"w")
 #           ho["img"]=im
  #          ho["hmf"]=0.0
   #         ho["muf"]=0.0
    #        ho.close()
            plt.close()
            
    fig, ax = plt.subplots(figsize=(14,10))
    fig.canvas.mpl_connect('key_press_event', press)
    ax.imshow(im)
    plt.show()
