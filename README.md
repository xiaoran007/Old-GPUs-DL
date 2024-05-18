# Build pytorch form source
## Hardware and OS
* Nvidia Tesla K40c
* Ubuntu 20.04 server

## Dependencies
* gcc 10.4
* Nvidia Driver 470
* cuda 11.4.4
* cudnn 8.7.0

## Setup toolkit
Suppose your ubuntu 20.04 server is newly installed.
### Install gcc-10:
```shell
sudo apt install gcc-10 g++-10
alias gcc="gcc-10"
alias g++="g++-10"
```
### Install Nvidia Driver:
```shell
sudo apt install nvidia-driver-470-server # or you can use other way to install it.
```
### Install cuda tool kit 11.4.4:
```shell
wget https://developer.download.nvidia.com/compute/cuda/11.4.4/local_installers/cuda_11.4.4_470.82.01_linux.run
sudo sh cuda_11.4.4_470.82.01_linux.run
```
Make sure you the installation target do not include the Driver, since the driver already installed by apt.

To verify that your CUDA installation is successful, use following commands:
```shell
cuda-install-samples-11.4.sh ~
cd ~/NVIDIA_CUDA-11.4_Samples
make
./bin/x86_64/linux/release/deviceQuery
```
You can see "Result = PASS".

### Install cudnn 8.7.0: 
download archive from [Nvidia website](https://developer.nvidia.com/rdp/cudnn-archive).
```shell
tar -xvf cudnn-linux-x86_64-8.7.0.84_cuda11-archive.tar.xz
cd cudnn-linux-x86_64-8.7.0.84_cuda11-archive/
sudo cp ./include/cudnn*.h /usr/local/cuda/include
sudo cp -P ./lib/libcudnn* /usr/local/cuda/lib64/
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

To verify that your cudnn is successful, use following commands:
```shell
sudo apt-get install libfreeimage3 libfreeimage-dev
git clone https://github.com/workmirror/cudnn_samples_v8.git
cd cudnn_samples_v8/mnistCUDNN/
sudo make clean
sudo make
./mnistCUDNN
```
You can see "Test passed!".

## Setup requirements
### Install miniconda:
```shell
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh
```
### Set env:
```shell
conda create -n build python=3.9
conda activate build
```
### Get pytorch source code:
```shell
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
git checkout v2.1.0 # or other version you want
git submodule sync
git submodule update --init --recursive
```
### Prepare
```shell
conda install cmake ninja
python -m pip install -r requirements.txt
conda install intel::mkl-static intel::mkl-include
conda install -c pytorch magma-cuda113 # no cuda114, use cuda113 instead
```
### Build
```shell
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export TORCH_CUDA_ARCH_LIST="3.5" # for cuda arch 3.5, 3.0 is not support by nvcc(cuda) 11.4.4
python setup.py develop # start build
```


# Test log
**Test Platform:** python 3.9 + pytorch 2.2.2 + cuda 11.8 + cudnn 8.7.0
* Nvidia
  * RTX 3090 24GB Driver 550
    * Image: 1447740
    * Size: 16976.75 MB
    * Score: 44749
  * RTX 4090 24GB Driver 535
    * Image: 1445570
    * Size: 16951.30 MB
    * Score: 43919 ?
  * RTX 4090 24GB Driver 550
    * Image: 1445220
    * Size: 16947.20 MB
    * Score: 73814
  * RTX 4090 D 24GB Driver 550
    * Image: 1445220
    * Size: 16947.20 MB
    * Score: 65718
  * L20 48GB Driver 550
    * Image: 2903670
    * Size: 34049.54 MB
    * Score: 68063
  * P104-100 4GB Driver 525
    * Image: 240939
    * Size: 2825.34 MB
    * Score: 12812

[//]: # (# Env)

[//]: # (```shell)

[//]: # (conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia)

[//]: # (conda install tqdm)

[//]: # (```)

[//]: # ()
[//]: # (# GPUs)

[//]: # (* Tesla K40c)

[//]: # (    * GK110B chip, 12G GDDR5 memory)

[//]: # (    * cuda: 11.2)

[//]: # (    * cuda capability: 3.5)

[//]: # (    * pytorch support: 1.7.1 with cuda 11.0)