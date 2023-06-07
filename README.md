# Towards Bone-Conducted Vibration Speech Enhancement on Head-Mounted Wearables
This is a repo for Mobisys 2022 paper: " <a href="https://dl.acm.org/doi/abs/10.1145/3495243.3560519"> Towards Bone-Conducted Vibration Speech Enhancement on Head-Mounted Wearables </a>".

# Requirements
The program has been tested in the following environment: 
* Python 3.8
* torch 1.12.1
* scikit-image 0.20.0
* pesq 0.0.4
* pystoi 0.3.3

Normally, all the other libraries can be installed by pip.
The repo has been tested on Ubuntu 20.04 with two RTX 3090 GPUs, the repo has been partially tested on Windows 10 with one RTX 3060 GPU (some format need revised).

# VibVoice Overview
<p align="center" >
	<img src="https://github.com/CUHK-AIoT-Sensing/vibvoice/tree/main/overview.pdf" width="700">
</p>

# Project Strcuture
```
.
├── __init__.py
├── bone_conduction_function.py
├── dataset.py          # pytorch dataset
├── display.py          # check the tranfer function
├── evaluation.py
├── feature.py
├── json_generate.py     # get the meta data for loading
├── mask.py
├── model
│   ├── SEANet.py
│   ├── __init__.py
│   ├── base_model.py
│   ├── deep_augmentation.py
│   ├── fullsubnet.py
│   ├── module/             # mainly for baseline 
│   ├── new_vibvoice.py
│   ├── vibvoice.py
│   └── voicefilter.py
├── model_zoo.py            # all the training function
├── synchronization.p #data pre-processing, already done
└── train.py          # both train and evaluation
```
# Data collection
 * Connect BMI160_I2C to Raspberry Pi with the <a href="https://github.com/lefuturiste/BMI160-i2c"> Instructions </a>
 * 3D-printing the bone-conduction headset, the design can be found in ./3dmodel
 * get a USB microphone to raspberry pi (other microphone is also possible, but you the USB one is easy to use)
 * Run the following code for data collection, you may need some debugging with the number of device (the mic device number can be random sometimes, may caused by ssh connection)
    ```
    python datarecord.py --device 0
    ```
## troubleshootings
1. Q: why there are enourmous printing out? 
    
    A: Face contact with the I2C header. On the other hand, after 30-minutes recording, such accident may happen (the header loose).
2. Q: why my sample rate is not as high as 1600 Hz?

    A: Raspberry Pi has a default sampling limit.

# Dataset preparation
## Audio-IMU dataset
* <a href="https://mycuhk-my.sharepoint.com/:u:/g/personal/1155170464_link_cuhk_edu_hk/Ef2s_G61F8BMnU-ksQpuP88B7wgDOu7VhNlYXsQZXAq4Pg?e=Xb8Jhc"> Self-collected dataset, 15 people (around 3 hours, main experiment)</a>
* <a href="https://mycuhk-my.sharepoint.com/:f:/g/personal/1155170464_link_cuhk_edu_hk/EiBk2p45s3RMiao70y4SZE8B8bFNUPtjgyot23ZtaXsC5A?e=AUqEeC"> Self-collected dataset, 8 people (small-scale, only for bone conduction function) </a>

* <a href="https://github.com/elevoctech/ESMB-corpus"> ESMB </a>
## Audio-only dataset
* <a href="https://www.openslr.org/12"> Librispeech </a>
* <a href="https://www.eng.biu.ac.il/~gannot/RIR_DATABASE/"> RIR (optional, and can be other RIR dataset) </a>
* Noise dataset, for simplicity, we can just use development set of LibriSpeech as strong speech noise. Other noise dataset can be used as well, besides, we also use <a href="https://mycuhk-my.sharepoint.com/:f:/g/personal/1155170464_link_cuhk_edu_hk/Ej0rWcuPnXVHt7VI4VRALFwBWrrZ4UlzJys3UZvL5NLvBg?e=eO75Tb"> environmental noises</a> and <a href="https://mycuhk-my.sharepoint.com/:f:/g/personal/1155170464_link_cuhk_edu_hk/El6D8hH2-cxMrpB4u9QgP3ABWckinFqNlKpz2veipZqCvA?e=i5hBnl"> music dataset. </a>

## Instructions
* Download all the datasets and place them in the same folder. Specially, we have `./bone_conduction_function`, `our`, `librispeech-100`, `dev` (librispeech-dev), `background`, `music` and `rir_fullsubnet`.


# Quick Start

* Run the following code for transfer-function extraction, the npz files will be created in ./transfer_function
    ```
    python bone_conduction_function.py  --data_dir dir/to/dataset/bone_conduction_function
    ```
* Run the following code for transfer function visulization, the png files will be created in ./vis
    ```
    python display.py
    ```
* Run the following code for data preprocessing, the json files will be created in ./json. (mode = 0, audio-dataset, mode = 1, audio-imu-dataset)
    ```
    python json_generate.py --data_dir ../../audioproject/dataset/our/ --mode 0
    python json_generate.py --data_dir ../../audioproject/dataset/our/ --mode 1
    ```

* Run the following code for model pretraining
    ```
    python train.py --mode 0
    ```
* Run the following code for model fine-tuning
    ```
    python train.py --mode 1
    ```



# Citation
The code of this project are made available for non-commercial, academic research only. If you would like to use the code of this project, please cite the following paper:
```
```
    