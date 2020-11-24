import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import h5py
import numpy as n
import matplotlib.pyplot as plt


class random_shift_data(tf.keras.utils.Sequence):
    """
    generate data on the fly with random x and y shifts.
    """
    def __init__(self,imgs,scaling,batch_size=32,N=5,x_width=100,y_width=100,noise_std=10.0,shift=True):
        self.imgs=imgs
        self.scaling=scaling
        self.batch_size=batch_size
        self.N=N
        self.x_width=x_width
        self.y_width=y_width
        self.noise_std=noise_std
        self.n_im=imgs.shape[0]
        self.shift=shift
        
    def __len__(self):
        return(int(self.n_im*self.N*self.N/float(self.batch_size)))
    
    def __getitem__(self,idx):
        imi = n.array(n.mod(self.batch_size*idx + n.arange(self.batch_size,dtype=n.int),self.n_im),dtype=n.int)
        img_out=n.zeros([self.batch_size,self.imgs.shape[1],self.imgs.shape[2]],dtype=n.float32)
        scale_out=n.zeros([self.batch_size,2],dtype=n.float32)
        for i in range(self.batch_size):
            im0=n.copy(self.imgs[imi[i],:,:])
            xi=0.0
            yi=0.0
            if self.shift:
                xi=int(n.random.rand(1)*self.x_width-self.x_width/2.0)
                yi=int(n.random.rand(1)*self.y_width-self.y_width/2.0)
                im0=n.roll(im0,xi,axis=0)
                im0=n.roll(im0,yi,axis=1)
                for fii in range(4):
                    fi=int(n.random.rand(1)*(im0.shape[1]-2))
                    im0[:,fi]+=n.abs(n.random.randn(1)*200.0)
                im0[im0>255.0]=255.0
                    
            
            img_out[i,:,:]=im0
            scale_out[i,0]=self.scaling[imi[i],0]+xi
            scale_out[i,1]=self.scaling[imi[i],1]+yi
            
        img_out.shape=(img_out.shape[0],img_out.shape[1],img_out.shape[2],1)
        return(img_out,scale_out)
        
def plot_random_sample(imgs,scalings):
    n_im=imgs.shape[0]
    idx=n.array(n.floor(n.random.rand(10)*n_im),dtype=n.int)
    for i in idx:
        print(i)
        plt.imshow(n.array(imgs[i,:,:],dtype=n.float32),vmin=0,vmax=255)
        plt.colorbar()
        plt.axhline(scalings[i,0],color="red")
        plt.axvline(scalings[i,1],color="green")
        plt.title(i)
        plt.show()
    
            

h=h5py.File("all.h5","r")


imgs0=n.array(n.copy(h["ok_data"].value),dtype=n.float32)
scaling0=n.copy(h["scaling"].value)
h.close()
dataset=random_shift_data(imgs0,scaling0,batch_size=32,N=10)
# no x & y shifts in validation dataset
validation_dataset=random_shift_data(imgs0,scaling0,batch_size=32,N=1,shift=False)

#ms=tf.distribute.MirroredStrategy()

#with ms.scope():
model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (7, 7), strides=(2,4), activation='relu', input_shape=(200, 498,1)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
 #       tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#        tf.keras.layers.Dropout(0.25),
#        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
 #       tf.keras.layers.Dropout(0.25),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
  #      tf.keras.layers.Dropout(0.25),               
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
   #     tf.keras.layers.Dropout(0.25),               
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024,activation="relu"),
        tf.keras.layers.Dense(1024,activation="relu"),
        tf.keras.layers.Dense(2)
    ])

model.compile(loss='mse',
              optimizer=tf.keras.optimizers.Adam(0.0001))
model.summary()
imgs00=n.copy(imgs0)
print(n.max(imgs00))
scalings00=n.copy(scaling0)

imgs0=tf.convert_to_tensor(imgs0)
imgs0=tf.reshape(imgs0,[-1,200,498,1])
scaling0=tf.convert_to_tensor(scaling0)

history = model.fit(dataset,
                    batch_size=32,
                    validation_data=validation_dataset,
                    epochs=10)

model.save("model/scaler")

n_input=scaling0.shape[0]

#imgs0=tf.convert_to_tensor(imgs0)
#imgs0=tf.reshape(imgs0,[-1,200,498,1])
#scaling0=tf.convert_to_tensor(scaling0)
pr=model.predict(imgs0)
for i in range(n_input):
    ph=pr[i,0]
    pf=pr[i,1]
        
    plt.imshow(imgs0[i,:,:,:])
    plt.axhline(ph,color="red")
    plt.axvline(pf,color="red")
    
    plt.axhline(scaling0[i,0],color="green")
    plt.axvline(scaling0[i,1],color="green")

    plt.show()
