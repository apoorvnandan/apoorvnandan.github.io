---
layout: post
comments: true
title:  "Model Implementation: Original U-Net in Keras / Tensorflow 2.3"
excerpt: "Using Keras' Functional API to implement U-Net architecture"
date:   2020-08-30 22:00:00
---
## Introduction
Original paper can be found [here](https://arxiv.org/abs/1505.04597).<br>
U-net predicts a class label for each input pixel.  The architecture is fully convolutional and is shown to perform well with small datasets for image segmentation 
tasks (especially for biomedical images).

I'll follow the exact architecture given in the Figure 1. from the paper (copied below)
<image src="/assets/unet.png" style="width:100%;">

## Implementation in 50 lines of code
```python
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model

def encode(inputs):
    conv1 = layers.Conv2D(64, 3, activation = 'relu')(inputs)
    conv1 = layers.Conv2D(64, 3, activation = 'relu')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = layers.Conv2D(128, 3, activation = 'relu')(pool1)
    conv2 = layers.Conv2D(128, 3, activation = 'relu')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = layers.Conv2D(256, 3, activation = 'relu')(pool2)
    conv3 = layers.Conv2D(256, 3, activation = 'relu')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = layers.Conv2D(512, 3, activation = 'relu')(pool3)
    conv4 = layers.Conv2D(512, 3, activation = 'relu')(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = layers.Conv2D(1024, 3, activation = 'relu')(pool4)
    conv5 = layers.Conv2D(1024, 3, activation = 'relu')(conv5)
    return conv5, conv4, conv3, conv2, conv1

def decode(conv5, conv4, conv3, conv2, conv1, num_classes):
    up6 = layers.Conv2DTranspose(512, 2, strides=(2, 2))(conv5)
    crop4 = layers.Cropping2D(4)(conv4)
    concat6 = layers.Concatenate(axis=3)([crop4,up6])
    conv6 = layers.Conv2D(512, 3, activation = 'relu')(concat6)
    conv6 = layers.Conv2D(512, 3, activation = 'relu')(conv6)

    up7 = layers.Conv2DTranspose(256, 2, strides=(2, 2))(conv6)
    crop3 = layers.Cropping2D(16)(conv3)
    concat7 = layers.Concatenate(axis=3)([crop3,up7])
    conv7 = layers.Conv2D(256, 3, activation = 'relu')(concat7)
    conv7 = layers.Conv2D(256, 3, activation = 'relu')(conv7)

    up8 = layers.Conv2DTranspose(128, 2, strides=(2, 2))(conv7)
    crop2 = layers.Cropping2D(40)(conv2)
    concat8 = layers.Concatenate(axis=3)([crop2,up8])
    conv8 = layers.Conv2D(128, 3, activation = 'relu')(concat8)
    conv8 = layers.Conv2D(128, 3, activation = 'relu')(conv8)

    up9 = layers.Conv2DTranspose(64, 2, strides=(2, 2))(conv8)
    crop1 = layers.Cropping2D(88)(conv1)
    concat9 = layers.Concatenate(axis=3)([crop1,up9])
    conv9 = layers.Conv2D(64, 3, activation = 'relu')(concat9)
    conv9 = layers.Conv2D(64, 3, activation = 'relu')(conv9)
    conv10 = layers.Conv2D(num_classes, 1)(conv9)
    conv10 = layers.Softmax(axis=-1)(conv10)
    return conv10

def create_unet(input_size=(572,572,1), num_classes=2):
    inputs = layers.Input(input_size)
    conv5, conv4, conv3, conv2, conv1 = encode(inputs)
    conv10 = decode(conv5, conv4, conv3, conv2, conv1, num_classes)
    model = Model(inputs, conv10)
    model.compile(optimizer = Adam(lr=1e-4), loss='categorical_crossentropy')
    return model

model = create_unet()
model.summary()
```

```text
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_2 (InputLayer)            [(None, 572, 572, 1) 0                                            
__________________________________________________________________________________________________
conv2d_19 (Conv2D)              (None, 570, 570, 64) 640         input_2[0][0]                    
__________________________________________________________________________________________________
conv2d_20 (Conv2D)              (None, 568, 568, 64) 36928       conv2d_19[0][0]                  
__________________________________________________________________________________________________
max_pooling2d_4 (MaxPooling2D)  (None, 284, 284, 64) 0           conv2d_20[0][0]                  
__________________________________________________________________________________________________
conv2d_21 (Conv2D)              (None, 282, 282, 128 73856       max_pooling2d_4[0][0]            
__________________________________________________________________________________________________
conv2d_22 (Conv2D)              (None, 280, 280, 128 147584      conv2d_21[0][0]                  
__________________________________________________________________________________________________
max_pooling2d_5 (MaxPooling2D)  (None, 140, 140, 128 0           conv2d_22[0][0]                  
__________________________________________________________________________________________________
conv2d_23 (Conv2D)              (None, 138, 138, 256 295168      max_pooling2d_5[0][0]            
__________________________________________________________________________________________________
conv2d_24 (Conv2D)              (None, 136, 136, 256 590080      conv2d_23[0][0]                  
__________________________________________________________________________________________________
max_pooling2d_6 (MaxPooling2D)  (None, 68, 68, 256)  0           conv2d_24[0][0]                  
__________________________________________________________________________________________________
conv2d_25 (Conv2D)              (None, 66, 66, 512)  1180160     max_pooling2d_6[0][0]            
__________________________________________________________________________________________________
conv2d_26 (Conv2D)              (None, 64, 64, 512)  2359808     conv2d_25[0][0]                  
__________________________________________________________________________________________________
max_pooling2d_7 (MaxPooling2D)  (None, 32, 32, 512)  0           conv2d_26[0][0]                  
__________________________________________________________________________________________________
conv2d_27 (Conv2D)              (None, 30, 30, 1024) 4719616     max_pooling2d_7[0][0]            
__________________________________________________________________________________________________
conv2d_28 (Conv2D)              (None, 28, 28, 1024) 9438208     conv2d_27[0][0]                  
__________________________________________________________________________________________________
cropping2d_4 (Cropping2D)       (None, 56, 56, 512)  0           conv2d_26[0][0]                  
__________________________________________________________________________________________________
conv2d_transpose_4 (Conv2DTrans (None, 56, 56, 512)  2097664     conv2d_28[0][0]                  
__________________________________________________________________________________________________
concatenate_4 (Concatenate)     (None, 56, 56, 1024) 0           cropping2d_4[0][0]               
                                                                 conv2d_transpose_4[0][0]         
__________________________________________________________________________________________________
conv2d_29 (Conv2D)              (None, 54, 54, 512)  4719104     concatenate_4[0][0]              
__________________________________________________________________________________________________
conv2d_30 (Conv2D)              (None, 52, 52, 512)  2359808     conv2d_29[0][0]                  
__________________________________________________________________________________________________
cropping2d_5 (Cropping2D)       (None, 104, 104, 256 0           conv2d_24[0][0]                  
__________________________________________________________________________________________________
conv2d_transpose_5 (Conv2DTrans (None, 104, 104, 256 524544      conv2d_30[0][0]                  
__________________________________________________________________________________________________
concatenate_5 (Concatenate)     (None, 104, 104, 512 0           cropping2d_5[0][0]               
                                                                 conv2d_transpose_5[0][0]         
__________________________________________________________________________________________________
conv2d_31 (Conv2D)              (None, 102, 102, 256 1179904     concatenate_5[0][0]              
__________________________________________________________________________________________________
conv2d_32 (Conv2D)              (None, 100, 100, 256 590080      conv2d_31[0][0]                  
__________________________________________________________________________________________________
cropping2d_6 (Cropping2D)       (None, 200, 200, 128 0           conv2d_22[0][0]                  
__________________________________________________________________________________________________
conv2d_transpose_6 (Conv2DTrans (None, 200, 200, 128 131200      conv2d_32[0][0]                  
__________________________________________________________________________________________________
concatenate_6 (Concatenate)     (None, 200, 200, 256 0           cropping2d_6[0][0]               
                                                                 conv2d_transpose_6[0][0]         
__________________________________________________________________________________________________
conv2d_33 (Conv2D)              (None, 198, 198, 128 295040      concatenate_6[0][0]              
__________________________________________________________________________________________________
conv2d_34 (Conv2D)              (None, 196, 196, 128 147584      conv2d_33[0][0]                  
__________________________________________________________________________________________________
cropping2d_7 (Cropping2D)       (None, 392, 392, 64) 0           conv2d_20[0][0]                  
__________________________________________________________________________________________________
conv2d_transpose_7 (Conv2DTrans (None, 392, 392, 64) 32832       conv2d_34[0][0]                  
__________________________________________________________________________________________________
concatenate_7 (Concatenate)     (None, 392, 392, 128 0           cropping2d_7[0][0]               
                                                                 conv2d_transpose_7[0][0]         
__________________________________________________________________________________________________
conv2d_35 (Conv2D)              (None, 390, 390, 64) 73792       concatenate_7[0][0]              
__________________________________________________________________________________________________
conv2d_36 (Conv2D)              (None, 388, 388, 64) 36928       conv2d_35[0][0]                  
__________________________________________________________________________________________________
conv2d_37 (Conv2D)              (None, 388, 388, 2)  130         conv2d_36[0][0]                  
__________________________________________________________________________________________________
softmax (Softmax)               (None, 388, 388, 2)  0           conv2d_37[0][0]                  
==================================================================================================
Total params: 31,030,658
Trainable params: 31,030,658
Non-trainable params: 0
__________________________________________________________________________________________________
```
