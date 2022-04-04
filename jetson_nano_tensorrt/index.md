# Jetson Nano TensorRT


This is a guide for TensorRT optimization. 

<!--more-->

## Convert Keras Model to Tensorflow Model

```python
# import the needed libraries
import tensorflow as tf
tf.keras.backend.set_learning_phase(0) #use this if we have batch norm layer in our network
from tensorflow.keras.models import load_model

# path we wanna save our converted TF-model
MODEL_PATH = "./model"

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
with tf.Session as sess:
    # import the meta graph of the tensorflow model
    saver = tf.train.import_meta_graph("./model/modelLeNet5.meta")
    # then, restore the weights to the meta graph
    saver.restore(sess, "./model/modelLeNet5")
    
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
```

## Method 2: Convert Keras Model to Tensorflow Frozen Model

```python
from keras import backend as K
# This line must be executed before loading Keras model.
K.set_learning_phase(0)

from keras.models import load_model
model = load_model('./model/keras_model.h5')
print(model.outputs)
# [<tf.Tensor 'dense_2/Softmax:0' shape=(?, 10) dtype=float32>]
print(model.inputs)
# [<tf.Tensor 'conv2d_1_input:0' shape=(?, 28, 28, 1) dtype=float32>]

from keras import backend as K
import tensorflow as tf

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])
                              
# Save to ./model/tf_model.pb
tf.train.write_graph(frozen_graph, "model", "tf_model.pb", as_text=False)
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
img1= Image.open("dataset/mnist/testing/0/img_108.jpg")
img2= Image.open("dataset/mnist/testing/1/img_0.jpg")
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
# variable
TENSORRT_MODEL_PATH = './model/TensorRT_model.pb'

graph = tf.Graph()
with graph.as_default():
    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.50))) as sess:
        # read TensorRT model
        trt_graph = read_pb_graph(TENSORRT_MODEL_PATH)

        # obtain the corresponding input-output tensor
        tf.import_graph_def(trt_graph, name='')
        input = sess.graph.get_tensor_by_name('input_tensor_input:0')
        output = sess.graph.get_tensor_by_name('output_tensor/Softmax:0')

        # in this case, it demonstrates to perform inference for 50 times
        total_time = 0; n_time_inference = 50
        out_pred = sess.run(output, feed_dict={input: input_img})
        for i in range(n_time_inference):
            t1 = time.time()
            out_pred = sess.run(output, feed_dict={input: input_img})
            t2 = time.time()
            delta_time = t2 - t1
            total_time += delta_time
            print("needed time in inference-" + str(i) + ": ", delta_time)
        avg_time_tensorRT = total_time / n_time_inference
        print("average inference time: ", avg_time_tensorRT)
```


