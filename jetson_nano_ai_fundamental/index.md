# Jetson Nano AI Fundamental


*This is a guide for training and performing inference on Jetson Nano.*

<!--more-->

## Using ImageNet on Jetson

```bash
git clone --recursive https://github.com/dusty-nv/jetson-inference
cd jetson-inference
docker/run.sh
# Click OK to download pre-trained models
```

```bash
#docker
cd build/aarch64/bin
./imagenet "images/object_*.jpg" "images/test/object_%i.jpg"
./imagenet "images/cat_*.jpg" "images/test/cat_%i.jpg"
```

## Training Keras Model and Perform Inference

Download MNIST dataset and copy it to "dataset/mnist".

### Read Input Images

```python
# import the needed libraries
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# config
img_width, img_height = 28,28 #width & height of input image
input_depth = 1 #1: gray image
train_data_dir = 'dataset/mnist/training' #data training path
testing_data_dir = 'dataset/mnist/testing' #data testing path
epochs = 2 #number of training epoch
batch_size = 5 #training batch size

# define image generator for Keras,
# here, we map pixel intensity to 0-1
train_datagen = ImageDataGenerator(rescale=1/255)
test_datagen = ImageDataGenerator(rescale=1/255)

# read image batch by batch
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    color_mode='grayscale',#inpput iameg: gray
    target_size=(img_width,img_height),#input image size
    batch_size=batch_size,#batch size
    class_mode='categorical')#categorical: one-hot encoding format class label
testing_generator = test_datagen.flow_from_directory(
    testing_data_dir,
    color_mode='grayscale',
    target_size=(img_width,img_height),
    batch_size=batch_size,
    class_mode='categorical')
```

### Define the Network

```python
# define number of filters and nodes in the fully connected layer
NUMB_FILTER_L1 = 20
NUMB_FILTER_L2 = 20
NUMB_FILTER_L3 = 20
NUMB_NODE_FC_LAYER = 10

#define input image order shape
if K.image_data_format() == 'channels_first':
    input_shape_val = (input_depth, img_width, img_height)
else:
    input_shape_val = (img_width, img_height, input_depth)

#define the network
model = Sequential()

# Layer 1
model.add(Conv2D(NUMB_FILTER_L1, (5, 5), 
                 input_shape=input_shape_val, 
                 padding='same', name='input_tensor'))
model.add(Activation('relu'))
model.add(MaxPool2D((2, 2)))

# Layer 2
model.add(Conv2D(NUMB_FILTER_L2, (5, 5), padding='same'))
model.add(Activation('relu'))
model.add(MaxPool2D((2, 2)))

# Layer 3
model.add(Conv2D(NUMB_FILTER_L3, (5, 5), padding='same'))
model.add(Activation('relu'))

# flattening the model for fully connected layer
model.add(Flatten())

# fully connected layer
model.add(Dense(NUMB_NODE_FC_LAYER, activation='relu'))

# output layer
model.add(Dense(train_generator.num_classes, 
                activation='softmax', name='output_tensor'))

# Compilile the network
model.compile(loss='categorical_crossentropy',
              optimizer='sgd', metrics=['accuracy'])

# Show the model summary
model.summary()
```

### Train the Network

```python
# Train and test the network
model.fit_generator(
    train_generator,#our training generator
    #number of iteration per epoch = number of data / batch size
    steps_per_epoch=np.floor(train_generator.n/batch_size),
    epochs=epochs,#number of epoch
    validation_data=testing_generator,#our validation generator
    #number of iteration per epoch = number of data / batch size
    validation_steps=np.floor(testing_generator.n / batch_size))
```

### Save the Trained Network

```python
print("Training is done!")
model.save('./model/modelLeNet5.h5')
print("Model is successfully stored!")
```

### Perform Inference

```python
# import the needed libraries
from tensorflow.keras.models import load_model
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

# read the input image using Pillow (you can use another library, e.g., OpenCV)
img1= Image.open("dataset/mnist/testing/0/img_108.jpg")
img2= Image.open("dataset/mnist/testing/1/img_0.jpg")
# convert to ndarray numpy
img1 = np.asarray(img1)
img2 = np.asarray(img2)

# load the trained model
model = load_model('./model/modelLeNet5.h5')

# predict the input image using the loaded model
pred1 = model.predict_classes((img1/255).reshape((1,28,28,1)))
pred2 = model.predict_classes((img2/255).reshape((1,28,28,1)))

# plot the prediction result
plt.figure('img1')
plt.imshow(img1,cmap='gray')
plt.title('pred:'+str(pred1[0]), fontsize=22)

plt.figure('img2')
plt.imshow(img2,cmap='gray')
plt.title('pred:'+str(pred2[0]), fontsize=22)

plt.show()
```

## Reference

1. [NVIDIA - Jetson Inference](https://github.com/dusty-nv/jetson-inference)
2. [Ardian Uman - Tenorflow-TensorRT](https://github.com/ardianumam/Tensorflow-TensorRT)


