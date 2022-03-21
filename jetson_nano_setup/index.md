# Jetson Nano Setup


*This is a Quick Start for Jetson Nano.*

<!--more-->

*Board used in this article is Jetson Nano 2GB.*

## Setup

### Overview

{{< image src="/images/Jetson_Nano_2GB.png" caption="Jetson Nano 2GB" >}}

| Ports                                |                               |
| :----------------------------------- | :---------------------------- |
| ① microSD card slot for main storage | ⑥ USB 3.0 port (x1)           |
| ② 40-pin expansion header            | ⑦ HDMI output port            |
| ③ Micro-USB port for Device Mode     | ⑧ USB-C for 5V power input    |
| ④ Gigabit Ethernet port              | ⑨ MIPI CSI-2 camera connector |
| ⑤ USB 2.0 ports (x2)                 |                               |


### Write Image to microSD Card

1. Download [Jetson Nano 2GB Developer Kit SD Card Image](https://developer.nvidia.com/jetson-nano-2gb-sd-card-image)

2. Write image to microSD with [Etcher](https://www.balena.io/etcher)

### First Boot

1. Insert the microSD card
2. Set the developer kit on a non-conductive surface
3. Connect monitor, keyboard, mouse and USB-C power supply (5V⎓3A)
4. Boot and setup

## Change Source

### Change Ubuntu Source

```bash
sudo nano /etc/apt/sources.list
```

```bash
# delete all contents
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic-backports main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic-security main restricted universe multiverse
```

### Install and Change pip3 Source

```bash
sudo apt install python3-pip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

## Pytorch

### PyTorch pip wheels

Will install PyTorch v1.10.0 since the image is embedded with JetPack 4.6.

### Installation

1. Install Pytorch

   ```bash
   wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
   sudo apt-get install python3-pip libopenblas-base libopenmpi-dev 
   pip3 install Cython
   pip3 install numpy torch-1.10.0-cp36-cp36m-linux_aarch64.whl
   ```
   
3. Install torchvision

   ```bash
   sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
   git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
   cd torchvision
   export BUILD_VERSION=0.11.1
   python3 setup.py install --user
   cd ../
   ```

### Verification

```bash
python3
```

```python
import torch
print(torch.__version__)
print('CUDA available: ' + str(torch.cuda.is_available()))
print('cuDNN version: ' + str(torch.backends.cudnn.version()))
a = torch.cuda.FloatTensor(2).zero_()
print('Tensor a = ' + str(a))
b = torch.randn(2).cuda()
print('Tensor b = ' + str(b))
c = a + b
print('Tensor c = ' + str(c))
import torchvision
print(torchvision.__version__)
```

## Tensorflow

### Installation

```bash
sudo apt-get update
sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
sudo apt-get install python3-pip
pip3 install -U pip testresources setuptools==49.6.0 
pip3 install -U --no-deps numpy==1.19.4 future==0.18.2 mock==3.0.5 keras_preprocessing==1.1.2 keras_applications==1.0.8 gast==0.4.0 protobuf pybind11 cython pkgconfig
env H5PY_SETUP_REQUIRES=0 pip3 install -U h5py==3.1.0
# if h5py installation failed, try "pip3 install h5py", which will install h5py 2.10.0
pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 tensorflow
```

### Verification

```bash
python3
```

```python
import tensorflow
```

## JupyterLab

### Installation

```bash
pip3 install jupyterlab==3
jupyter lab --generate-config
```

### Setting

```bash
nano /home/jetson/.jupyter/jupyter_lab_config.py
```

```python
# copy to front of the file
c.ServerApp.allow_remote_access = True
c.ExtensionApp.open_browser = False
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.password_required = False
c.ServerApp.port = 8888
c.ServerApp.token = ''
```

### Usage

```bash
jupyter lab 
```

## Reference

1. [Zhihu - Jetson Nano 快速入门](https://www.zhihu.com/column/c_1093559321706819584)
2. [NVIDIA Developer - Getting Started with Jetson Nano 2GB Developer Kit](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-2gb-devkit)

