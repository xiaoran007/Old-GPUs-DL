# Test log
**Test Platform:** python 3.9 + pytorch 2.2.2 + cuda 11.8 + cudnn 8.7.0
* Nvidia
  * RTX 3090 24GB 
    * Image: 1447740
    * Size: 16976.75 MB
    * Score: 44749
  * L20 48GB
    * Image: 2903670
    * Size: 34049.54 MB
    * Score: 68063
  * P104-100 4GB
    * Image: 240939
    * Size: 2825.34 MB
    * Score: 12812



# Env
```shell
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install tqdm
```

# GPUs
* Tesla K40c
    * GK110B chip, 12G GDDR5 memory
    * cuda: 11.2
    * cuda capability: 3.5
    * pytorch support: 1.7.1 with cuda 11.0