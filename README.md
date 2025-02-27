# Build PyTorch from source
## Hardware and OS
* Nvidia Tesla K40c
* Ubuntu 20.04 server

## Dependencies
* gcc 10.4
* Nvidia Driver 470
* cuda 11.4.4
* cudnn 8.7.0

## Setup toolkit
Suppose your Ubuntu 20.04 server is newly installed.
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
Make sure the installation target does not include the Driver, since the driver is already installed by apt.

To verify that your CUDA installation is successful, use the following commands:
```shell
cuda-install-samples-11.4.sh ~
cd ~/NVIDIA_CUDA-11.4_Samples
make
./bin/x86_64/linux/release/deviceQuery
```
You can see "Result = PASS".

### Install cudnn 8.7.0: 
download the archive from [Nvidia website](https://developer.nvidia.com/rdp/cudnn-archive).
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

To verify that your cudnn is installed successfully, use the following commands:
```shell
sudo apt-get install libfreeimage3 libfreeimage-dev
git clone https://github.com/workmirror/cudnn_samples_v8.git # Nvidia only include samples in deb package, so use this mirror here
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
conda create -n build python=3.9 # or you can use other version >= 3.8
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
export MAX_JOBS=12 # you may set this value higher if you have memory larger than 16GB
export TORCH_CUDA_ARCH_LIST="3.5" # for cuda arch 3.5, 3.0 is not support by nvcc(cuda) 11.4.4
python setup.py develop # start build
python setup.py bdist_wheel # build whell package
```

If you want to build pytorch for other cuda compute capability devices, you should install the correct version of cuda toolkit and cudnn, here is the reference. For example, if you want to build torch support old kepler (cc3.0) device, you should use cuda 10.x (not tested).
![compute_capability](./imgs/compute_capability.png)

# Default benchmark
```shell
python main.py -m -s 512 -e 2 -mt resnet50 -bs 256 -dt FP32 -gpu 0
python main.py -m -s 512 -e 2 -mt resnet50 -bs 256 -dt FP16 -gpu 0
```

# New Performance Test Log

## How to understand the results

This performance test evaluates the performance of the hardware device in a training scenario, and the output is a score. Score reflects the unit time to complete a given training task (ResNet50). Thus a higher score means higher computational performance. Note that the score is affected by both the video memory bandwidth and the PCIe bus bandwidth.

## Test command

FP32: `python main.py -m -s 512 -e 3 -mt resnet50 -bs 256 -dt FP32`

FP16: `python main.py -m -s 512 -e 3 -mt resnet50 -bs 256 -dt FP16`

FP32 cudnn: `python main.py -m -s 512 -e 10 -mt resnet50 -bs 256 -dt FP32 -cudnn`

FP16 cudnn: `python main.py -m -s 512 -e 10 -mt resnet50 -bs 256 -dt FP16 -cudnn`

Note: In general, Multi-GPU test case using batch size 2048 `-bs 2048` for FP16 and batch size 1024 `-bs 1024` for FP32.

## Results

Most of the results are obtained by adjusting the batch size to get the maximum video memory usage.

|                 Device                 |          Platform          | FP32  | FP32BS |  FP16  | FP16BS |                     Note                     |
| :------------------------------------: | :------------------------: | :---: | :----: | :----: | :----: | :------------------------------------------: |
|              Apple M4 GPU              |     macOS<br />15.3.1      | 1723  |  128   |  1591  |  128   |                   10 Cores                   |
|              Apple M1 GPU              |     macOS<br />15.3.1      |  948  |  128   |  843   |  128   |                   8 Cores                    |
|      NVIDIA GeForce RTX 3090 24GB      |    Windows<br />566.14     | 16311 |  256   | 28197  |  256   |                      /                       |
|         NVIDIA RTX A5000 24GB          |     Linux<br />535.183     | 15090 |  512   | 27155  |  1024  |                      /                       |
|    NVIDIA RTX A5000 24GB    2 GPUs     |     Linux<br />535.183     | 26962 |  1024  | 49930  |  3072  |                    NVLink                    |
|      NVIDIA GeForce RTX 3080 20GB      | Linux (Docker)<br />560.35 | 13320 |  256   | 24205  |  256   |      Unofficial Video Memory Expansion       |
| NVIDIA GeForce RTX 3080 20GB    2 GPUs | Linux (Docker)<br />560.35 | 23261 |  1024  | 40250  |  2048  |      Unofficial Video Memory Expansion       |
|         Tesla V100S-PCIE 32GB          | Linux (Docker)<br />550.90 | 11577 |  256   | 27963  |  256   |                      /                       |
|            NVIDIA vGPU-32GB            | Linux (Docker)<br />560.35 | 16050 |  1024  | 28155  |  2048  |      Two NVIDIA GeForce RTX 4080 SUPER       |
|       NVIDIA vGPU-32GB    2 GPUs       | Linux (Docker)<br />560.35 | 30275 |  2048  | 52756  |  4096  |      Two NVIDIA GeForce RTX 4080 SUPER       |
|       NVIDIA vGPU-32GB    4 GPUs       | Linux (Docker)<br />560.35 | 56178 |  4096  | 101268 |  8192  |      Two NVIDIA GeForce RTX 4080 SUPER       |
|      NVIDIA TITAN X (Pascal) 12GB      |    Windows<br />566.14     | 5792  |  256   |  7230  |  256   | FP16 Not Officially Supported By Pascal Arch |
|  Intel(R) Arc(TM) A770 Graphics 16GB   |     Linux<br />i915 xe     | 5121  |  256   |  8049  |  256   |             GradScaler Not Work              |



# Old Performance Test log

**Test Platform:** python 3.9 + pytorch 2.2.2 + cuda 11.8 + cudnn 8.7.0\
The following test log is based on benchmark_cnn_v0.1
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
  * P104-100 8GB Driver 536
    * Image: 511600
    * Size: 5999.22 MB
    * Score: 12428
  * Tesla M40 12GB Driver 470
    * Image: 698500
    * Size: 8190.88 MB
    * Score: 13038
  * RTX 2080Ti 11GB Driver 470
    * Image: 682200
    * Size: 8000 MB
    * Score: 28999
  

[//]: # (# Env)

[//]: # (```shell)

[//]: # (conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia)

[//]: # (conda install tqdm)

[//]: # (```)

[//]: #
[//]: # (# GPUs)

[//]: # (* Tesla K40c)

[//]: # (    * GK110B chip, 12G GDDR5 memory)

[//]: # (    * cuda: 11.2)

[//]: # (    * cuda capability: 3.5)

[//]: # (    * pytorch support: 1.7.1 with cuda 11.0)