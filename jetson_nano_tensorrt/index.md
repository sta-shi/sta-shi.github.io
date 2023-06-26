# Jetson Nano TensorRT


This is a guide for TensorRT optimization. 

<!--more-->

## Overview

{{< image src="/images/Jetson_Nano_TensorRT/Keras_to_TensorRT.jpg" caption="Convert Keras Model to TensorRT" >}}

Tensorflow has two versions: v1 and v2. v2 has a poor compatiblity to v1. On aurthor's unit the code doesn't work (some modules cannot be found), so I reinstalled Tensorflow 1.15.0 and re-trained the model (since Keras can't operate models trained by higher vesion). Also, it's found inconvenient to download wheel from NVIDIA download website in China. I download wheel from other device and install Tensorflow with pip3 wheel tool. 

Furthermore, in original code by Ardian Uman, a cap is set for TensorRT, which leads to system dump. After removing it, I can optimize Tensorflow model with TensorRT and run TensorRT model as well. 

In the process of performing inference, it's found that first several inferences of TensorRT model costs a rather long period that significantly influences the result. To obtain a "seemingly reasonable" average time, I abandoned first ten results of each model inference. Reason for the weird phenomenon remains studying. 

## Convert Keras Model to Tensorflow Model

```python
# import the needed libraries
import tensorflow as tf
tf.keras.backend.set_learning_phase(0) #use this if we have batch norm layer in our network
from tensorflow.keras.models import load_model

# path we wanna save our converted TF-model
MODEL_PATH = "./model/TensorFlow"

# load the Keras model
model = load_model('./model/modelLeNet5.h5')

# save the model to Tensorflow model
saver = tf.train.Saver()
sess = tf.keras.backend.get_session()
save_path = saver.save(sess, MODEL_PATH)

print("Keras model is successfully converted to TF model in "+MODEL_PATH)
```

## Use TensorRT to Optimize Tensorflow Model

### Convert Tensorflow Model to Frozen Model

```python
# import the needed libraries
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.platform import gfile

# has to be use this setting to make a session for TensorRT optimization
with tf.Session() as sess:
    # import the meta graph of the tensorflow model
    saver = tf.train.import_meta_graph("./model/TensorFlow.meta")
    # then, restore the weights to the meta graph
    saver.restore(sess, "./model/TensorFlow")
    
    # specify which tensor output you want to obtain 
    # (correspond to prediction result)
    your_outputs = ["output_tensor/Softmax"]
    
    # convert to frozen model
    frozen_graph = tf.graph_util.convert_variables_to_constants(
        sess, # session
        tf.get_default_graph().as_graph_def(),# graph+weight from the session
        output_node_names=your_outputs)
    #write the TensorRT model to be used later for inference
    with gfile.FastGFile("./model/frozen_model.pb", 'wb') as f:
        f.write(frozen_graph.SerializeToString())
    print("Frozen model is successfully stored!")
```

### Optimize Frozen Model to TensorRT Graph

```python
# convert (optimize) frozen model to TensorRT model
trt_graph = trt.create_inference_graph(
    input_graph_def=frozen_graph,# frozen model
    outputs=your_outputs,
    max_batch_size=2,# specify your max batch size
    max_workspace_size_bytes=2*(10**9),# specify the max workspace
    precision_mode="FP32") # precision, can be "FP32" (32 floating point precision) or "FP16"

#write the TensorRT model to be used later for inference
with gfile.FastGFile("./model/TensorRT_model.pb", 'wb') as f:
    f.write(trt_graph.SerializeToString())
print("TensorRT model is successfully stored!")

all_nodes = len([1 for n in frozen_graph.node])
print("numb. of all_nodes in frozen graph:", all_nodes)
trt_engine_nodes = len([1 for n in trt_graph.node if str(n.op) == 'TRTEngineOp'])
print("numb. of trt_engine_nodes in TensorRT graph:", trt_engine_nodes)
all_nodes = len([1 for n in trt_graph.node])
print("numb. of all_nodes in TensorRT graph:", all_nodes)
```

## Perform Inference with TensorRT Model

```python
# import the needed libraries
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt # must import this although we will not use it explicitly
from tensorflow.python.platform import gfile
from PIL import Image
import numpy as np
import time
from matplotlib import pyplot as plt

# read the testing images (only for example)
img1= Image.open("dataset/mnist/testing/0/test3_0.jpg")
img2= Image.open("dataset/mnist/testing/1/test2_1.jpg")
img1 = np.asarray(img1)
img2 = np.asarray(img2)
input_img = np.concatenate((img1.reshape((1, 28, 28, 1)), img2.reshape((1, 28, 28, 1))), axis=0)

# function to read a ".pb" model 
# (can be used to read frozen model or TensorRT model)
def read_pb_graph(model):
  with gfile.FastGFile(model,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def
  
# perform inference
# original model
FROZEN_MODEL_PATH = './model/frozen_model.pb'

graph = tf.Graph()
with graph.as_default():
    with tf.Session() as sess:
        # read frozen model
        trt_graph = read_pb_graph(FROZEN_MODEL_PATH)

        # obtain the corresponding input-output tensor
        tf.import_graph_def(trt_graph, name='')
        input = sess.graph.get_tensor_by_name('input_tensor_input:0')
        output = sess.graph.get_tensor_by_name('output_tensor/Softmax:0')

        # in this case, it demonstrates to perform inference for 50 times
        total_time = 0; n_time_inference = 100
        out_pred = sess.run(output, feed_dict={input: input_img})
        for i in range(n_time_inference):
            t1 = time.time()
            out_pred = sess.run(output, feed_dict={input: input_img})
            t2 = time.time()
            delta_time = t2 - t1
            if i>9:
               total_time += delta_time
            print("Needed time in inference " + str(i) + ": ", delta_time)
        avg_time_original_model = total_time / (n_time_inference-10)
        print("Average inference time: ", avg_time_original_model)

# TensorRT model
TENSORRT_MODEL_PATH = './model/TensorRT_model.pb'

graph = tf.Graph()
with graph.as_default():
    with tf.Session() as sess:
        # read TensorRT model
        trt_graph = read_pb_graph(TENSORRT_MODEL_PATH)

        # obtain the corresponding input-output tensor
        tf.import_graph_def(trt_graph, name='')
        input = sess.graph.get_tensor_by_name('input_tensor_input:0')
        output = sess.graph.get_tensor_by_name('output_tensor/Softmax:0')

        # in this case, it demonstrates to perform inference for 50 times
        total_time = 0; n_time_inference = 100
        out_pred = sess.run(output, feed_dict={input: input_img})
        for i in range(n_time_inference):
            t1 = time.time()
            out_pred = sess.run(output, feed_dict={input: input_img})
            t2 = time.time()
            delta_time = t2 - t1
            if i>9:
               total_time += delta_time
            print("Needed time in inference " + str(i) + ": ", delta_time)
        avg_time_tensorRT = total_time / (n_time_inference-10)
        print("Average inference time: ", avg_time_tensorRT)
        print("TensorRT improvement compared to the original model:", avg_time_original_model/avg_time_tensorRT)

plt.figure('img 1')
plt.imshow(img1,cmap='gray')
plt.title('pred: ' + str(np.argmax(out_pred[0])),fontsize=22)

plt.figure('img 2')
plt.imshow(img2,cmap='gray')
plt.title('pred: ' + str(np.argmax(out_pred[1])),fontsize=22)
plt.show()
```


