---
layout: post
comments: true
title:  "Model Implementation: Original U-Net in Keras / Tensorflow 2.3)"
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

def decode(conv5, conv4, conv3, conv2, conv1):
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
    conv10 = layers.Conv2D(2, 1, activation = 'softmax')(conv9)
    return conv10

def create_unet(input_size = (572,572,1)):
    inputs = layers.Input(input_size)
    conv5, conv4, conv3, conv2, conv1 = encode(inputs)
    conv10 = decode(conv5, conv4, conv3, conv2, conv1)
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
input_5 (InputLayer)            [(None, 572, 572, 1) 0                                            
__________________________________________________________________________________________________
conv2d_51 (Conv2D)              (None, 570, 570, 64) 640         input_5[0][0]                    
__________________________________________________________________________________________________
conv2d_52 (Conv2D)              (None, 568, 568, 64) 36928       conv2d_51[0][0]                  
__________________________________________________________________________________________________
max_pooling2d_12 (MaxPooling2D) (None, 284, 284, 64) 0           conv2d_52[0][0]                  
__________________________________________________________________________________________________
conv2d_53 (Conv2D)              (None, 282, 282, 128 73856       max_pooling2d_12[0][0]           
__________________________________________________________________________________________________
conv2d_54 (Conv2D)              (None, 280, 280, 128 147584      conv2d_53[0][0]                  
__________________________________________________________________________________________________
max_pooling2d_13 (MaxPooling2D) (None, 140, 140, 128 0           conv2d_54[0][0]                  
__________________________________________________________________________________________________
conv2d_55 (Conv2D)              (None, 138, 138, 256 295168      max_pooling2d_13[0][0]           
__________________________________________________________________________________________________
conv2d_56 (Conv2D)              (None, 136, 136, 256 590080      conv2d_55[0][0]                  
__________________________________________________________________________________________________
max_pooling2d_14 (MaxPooling2D) (None, 68, 68, 256)  0           conv2d_56[0][0]                  
__________________________________________________________________________________________________
conv2d_57 (Conv2D)              (None, 66, 66, 512)  1180160     max_pooling2d_14[0][0]           
__________________________________________________________________________________________________
conv2d_58 (Conv2D)              (None, 64, 64, 512)  2359808     conv2d_57[0][0]                  
__________________________________________________________________________________________________
max_pooling2d_15 (MaxPooling2D) (None, 32, 32, 512)  0           conv2d_58[0][0]                  
__________________________________________________________________________________________________
conv2d_59 (Conv2D)              (None, 30, 30, 1024) 4719616     max_pooling2d_15[0][0]           
__________________________________________________________________________________________________
conv2d_60 (Conv2D)              (None, 28, 28, 1024) 9438208     conv2d_59[0][0]                  
__________________________________________________________________________________________________
cropping2d_10 (Cropping2D)      (None, 56, 56, 512)  0           conv2d_58[0][0]                  
__________________________________________________________________________________________________
conv2d_transpose_10 (Conv2DTran (None, 56, 56, 512)  2097664     conv2d_60[0][0]                  
__________________________________________________________________________________________________
concatenate_10 (Concatenate)    (None, 56, 56, 1024) 0           cropping2d_10[0][0]              
                                                                 conv2d_transpose_10[0][0]        
__________________________________________________________________________________________________
conv2d_61 (Conv2D)              (None, 54, 54, 512)  4719104     concatenate_10[0][0]             
__________________________________________________________________________________________________
conv2d_62 (Conv2D)              (None, 52, 52, 512)  2359808     conv2d_61[0][0]                  
__________________________________________________________________________________________________
cropping2d_11 (Cropping2D)      (None, 104, 104, 256 0           conv2d_56[0][0]                  
__________________________________________________________________________________________________
conv2d_transpose_11 (Conv2DTran (None, 104, 104, 256 524544      conv2d_62[0][0]                  
__________________________________________________________________________________________________
concatenate_11 (Concatenate)    (None, 104, 104, 512 0           cropping2d_11[0][0]              
                                                                 conv2d_transpose_11[0][0]        
__________________________________________________________________________________________________
conv2d_63 (Conv2D)              (None, 102, 102, 256 1179904     concatenate_11[0][0]             
__________________________________________________________________________________________________
conv2d_64 (Conv2D)              (None, 100, 100, 256 590080      conv2d_63[0][0]                  
__________________________________________________________________________________________________
cropping2d_12 (Cropping2D)      (None, 200, 200, 128 0           conv2d_54[0][0]                  
__________________________________________________________________________________________________
conv2d_transpose_12 (Conv2DTran (None, 200, 200, 128 131200      conv2d_64[0][0]                  
__________________________________________________________________________________________________
concatenate_12 (Concatenate)    (None, 200, 200, 256 0           cropping2d_12[0][0]              
                                                                 conv2d_transpose_12[0][0]        
__________________________________________________________________________________________________
conv2d_65 (Conv2D)              (None, 198, 198, 128 295040      concatenate_12[0][0]             
__________________________________________________________________________________________________
conv2d_66 (Conv2D)              (None, 196, 196, 128 147584      conv2d_65[0][0]                  
__________________________________________________________________________________________________
cropping2d_13 (Cropping2D)      (None, 392, 392, 64) 0           conv2d_52[0][0]                  
__________________________________________________________________________________________________
conv2d_transpose_13 (Conv2DTran (None, 392, 392, 64) 32832       conv2d_66[0][0]                  
__________________________________________________________________________________________________
concatenate_13 (Concatenate)    (None, 392, 392, 128 0           cropping2d_13[0][0]              
                                                                 conv2d_transpose_13[0][0]        
__________________________________________________________________________________________________
conv2d_67 (Conv2D)              (None, 390, 390, 64) 73792       concatenate_13[0][0]             
__________________________________________________________________________________________________
conv2d_68 (Conv2D)              (None, 388, 388, 64) 36928       conv2d_67[0][0]                  
__________________________________________________________________________________________________
conv2d_69 (Conv2D)              (None, 388, 388, 2)  130         conv2d_68[0][0]                  
==================================================================================================
Total params: 31,030,658
Trainable params: 31,030,658
Non-trainable params: 0
__________________________________________________________________________________________________
```
