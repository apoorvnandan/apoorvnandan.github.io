---
layout: post
comments: true
title:  "Clean Implementation: Original U-Net in Keras (tensorflow 2)"
excerpt: "Using Keras' Functional API to implement original U-Net architecture"
date:   2020-08-30 22:00:00
---
## Introduction
Original paper can be found [here](https://arxiv.org/abs/1505.04597).<br>
U-net predicts a class label for each input pixel.  The architecture is fully convolutional and is shown to perform well with small datasets for image segmentation 
tasks (especially for biomedical images).

I'll follow the exact architecture given in the Figure 1. from the paper (copied below)
<image src="/assets/unet.png" style="width:100%;">

## Implementation in 60 lines of code
```python
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model

def encode(inputs):
    conv1 = layers.Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(inputs)
    conv1 = layers.Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = layers.Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool1)
    conv2 = layers.Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = layers.Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool2)
    conv3 = layers.Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = layers.Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool3)
    conv4 = layers.Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = layers.Conv2D(1024, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool4)
    conv5 = layers.Conv2D(1024, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv5)
    return conv5, conv4, conv3, conv2, conv1

def decode(conv5, conv4, conv3, conv2, conv1):
    up6 = layers.Conv2DTranspose(512, 2, strides=(2, 2))(conv5)
    crop4 = layers.Cropping2D(4)(conv4)
    concat6 = layers.Concatenate(axis=3)([crop4,up6])
    conv6 = layers.Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(concat6)
    conv6 = layers.Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv6)

    up7 = layers.Conv2DTranspose(512, 2, strides=(2, 2))(conv6)
    crop3 = layers.Cropping2D(16)(conv3)
    concat7 = layers.Concatenate(axis=3)([crop3,up7])
    conv7 = layers.Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(concat7)
    conv7 = layers.Conv2D(258, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv7)

    up8 = layers.Conv2DTranspose(512, 2, strides=(2, 2))(conv7)
    crop2 = layers.Cropping2D(40)(conv2)
    concat8 = layers.Concatenate(axis=3)([crop2,up8])
    conv8 = layers.Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(concat8)
    conv8 = layers.Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv8)

    up9 = layers.Conv2DTranspose(512, 2, strides=(2, 2))(conv8)
    crop1 = layers.Cropping2D(88)(conv1)
    concat9 = layers.Concatenate(axis=3)([crop1,up9])
    conv9 = layers.Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(concat9)
    conv9 = layers.Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv9)
    conv10 = layers.Conv2D(2, 1, activation = 'softmax')(conv9)
    return conv10

def create_unet(input_size = (572,572,1)):
    inputs = layers.Input(input_size)
    conv5, conv4, conv3, conv2, conv1 = encode(inputs)
    conv10 = decode(conv5, conv4, conv3, conv2, conv1)
    model = Model(inputs, conv10)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model


model = create_unet()
model.summary()
```

```bash
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 572, 572, 1) 0                                            
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 570, 570, 64) 640         input_1[0][0]                    
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 568, 568, 64) 36928       conv2d[0][0]                     
__________________________________________________________________________________________________
max_pooling2d (MaxPooling2D)    (None, 284, 284, 64) 0           conv2d_1[0][0]                   
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 282, 282, 128 73856       max_pooling2d[0][0]              
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 280, 280, 128 147584      conv2d_2[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 140, 140, 128 0           conv2d_3[0][0]                   
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 138, 138, 256 295168      max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 136, 136, 256 590080      conv2d_4[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 68, 68, 256)  0           conv2d_5[0][0]                   
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 66, 66, 512)  1180160     max_pooling2d_2[0][0]            
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 64, 64, 512)  2359808     conv2d_6[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 32, 32, 512)  0           conv2d_7[0][0]                   
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 30, 30, 1024) 4719616     max_pooling2d_3[0][0]            
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 28, 28, 1024) 9438208     conv2d_8[0][0]                   
__________________________________________________________________________________________________
cropping2d (Cropping2D)         (None, 56, 56, 512)  0           conv2d_7[0][0]                   
__________________________________________________________________________________________________
conv2d_transpose (Conv2DTranspo (None, 56, 56, 512)  2097664     conv2d_9[0][0]                   
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 56, 56, 1024) 0           cropping2d[0][0]                 
                                                                 conv2d_transpose[0][0]           
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 54, 54, 512)  4719104     concatenate[0][0]                
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 52, 52, 512)  2359808     conv2d_10[0][0]                  
__________________________________________________________________________________________________
cropping2d_1 (Cropping2D)       (None, 104, 104, 256 0           conv2d_5[0][0]                   
__________________________________________________________________________________________________
conv2d_transpose_1 (Conv2DTrans (None, 104, 104, 512 1049088     conv2d_11[0][0]                  
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 104, 104, 768 0           cropping2d_1[0][0]               
                                                                 conv2d_transpose_1[0][0]         
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 102, 102, 256 1769728     concatenate_1[0][0]              
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 100, 100, 258 594690      conv2d_12[0][0]                  
__________________________________________________________________________________________________
cropping2d_2 (Cropping2D)       (None, 200, 200, 128 0           conv2d_3[0][0]                   
__________________________________________________________________________________________________
conv2d_transpose_2 (Conv2DTrans (None, 200, 200, 512 528896      conv2d_13[0][0]                  
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, 200, 200, 640 0           cropping2d_2[0][0]               
                                                                 conv2d_transpose_2[0][0]         
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 198, 198, 128 737408      concatenate_2[0][0]              
__________________________________________________________________________________________________
conv2d_15 (Conv2D)              (None, 196, 196, 128 147584      conv2d_14[0][0]                  
__________________________________________________________________________________________________
cropping2d_3 (Cropping2D)       (None, 392, 392, 64) 0           conv2d_1[0][0]                   
__________________________________________________________________________________________________
conv2d_transpose_3 (Conv2DTrans (None, 392, 392, 512 262656      conv2d_15[0][0]                  
__________________________________________________________________________________________________
concatenate_3 (Concatenate)     (None, 392, 392, 576 0           cropping2d_3[0][0]               
                                                                 conv2d_transpose_3[0][0]         
__________________________________________________________________________________________________
conv2d_16 (Conv2D)              (None, 390, 390, 64) 331840      concatenate_3[0][0]              
__________________________________________________________________________________________________
conv2d_17 (Conv2D)              (None, 388, 388, 64) 36928       conv2d_16[0][0]                  
__________________________________________________________________________________________________
conv2d_18 (Conv2D)              (None, 388, 388, 2)  130         conv2d_17[0][0]                  
==================================================================================================
Total params: 33,477,572
Trainable params: 33,477,572
Non-trainable params: 0
__________________________________________________________________________________________________
```
