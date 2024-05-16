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