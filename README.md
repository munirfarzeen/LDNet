# LDNet: End-to-End Lane Marking Detection Approach Using a Dynamic Vision Sensor
This is the Pytorch implementation of LDNet a Lane marking detection algorithm on DVS data.
Peper "LDNet: End-to-End Lane Marking Detection Approach Using a Dynamic Vision Sensor", accepted in [2021](https://ieeexplore.ieee.org/document/9518365).

## Installation

To run the **demo example** you **need only** Pytorch, Numpy, and dropblock dependecies.

**Main Dependencies:**
- Pytorch 1.6.0
- Torchvision 0.7.0
- pyyaml 3.13
- [DropBlock](https://github.com/miguelvr/dropblock)

Inorder to use this code you must install Anaconda and then apply the following steps:
+ Create the environment from the environment.yml file:

```
conda env create -f environment.yml
```
+ Activate LDNet environment

```
source activate LDNet
```

+ Install DropBlock

```
pip install dropblock

cd LDNet/

pip install -r requirements.txt
```

## Training

#### Dataset
Before start training, download DET dataset from [here](https://spritea.github.io/DET/) 

#### Training paramteres
[training.yaml](https://github.com/likui01/LDNet/blob/master/config/training.yaml) contains parameters needed for training as:
+ DATASET, path to  dataset folder. The folder (DET) must follow this pattern

    ``` 
    /path/to/DET
                /train		
                /val 	  
                /test 
   ```
  #### Start training:
In order to train the network with a specific backbone network and get and replicate the paper result you must train network.

+ To train on DET data set and run the training via 

``` python train.py  ```          



## Citation 
```
@ARTICLE{9518365,
  author={Munir, Farzeen and Azam, Shoaib and Jeon, Moongu and Lee, Byung-Geun and Pedrycz, Witold},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={LDNet: End-to-End Lane Marking Detection Approach Using a Dynamic Vision Sensor}, 
  year={2021},
  volume={},
  number={},
  pages={1-17},
  doi={10.1109/TITS.2021.3102479}}
```
