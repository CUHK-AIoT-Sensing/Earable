# Towards Bone-Conducted Vibration Speech Enhancement on Head-Mounted Wearables
This is a repo for Mobisys 2022 paper: " <a href="https://dl.acm.org/doi/abs/10.1145/3495243.3560519"> Towards Bone-Conducted Vibration Speech Enhancement on Head-Mounted Wearables </a>".

# Requirements
The program has been tested in the following environment: 
* Python 3.9.7
* Pytorch 1.8.1
* torchvision 0.9.1
* sklearn 0.24.2
* opencv-python 4.5.5
* numpy 1.20.3

# Cosmo Overview
<p align="center" >
	<img src="https://github.com/xmouyang/Cosmo/blob/main/materials/Overview.png" width="700">
</p>

# Project Strcuture
```
```

# Dataset preparation
## Audio-IMU dataset
* <a href="https://github.com/elevoctech/ESMB-corpus"> Self-collected dataset, collected by </a>
* <a href="https://github.com/elevoctech/ESMB-corpus"> ESMB </a>
## Audio-only dataset
* <a href="https://www.openslr.org/12"> Librispeech </a>
* <a href="https://www.eng.biu.ac.il/~gannot/RIR_DATABASE/"> RIR (optional, and can be other RIR dataset) </a>
* Noise dataset, for simplicity, we can just use development set of LibriSpeech as strong speech noise. Other noise dataset can be used as well, besides, we also use music dataset and background environment sound dataset.


# Quick Start
* Download the `self-collected dataset` dataset (8 users for bone conduction function, 15 users for the main results) and `librispeech-100-clean` to your machine. Specially, `./bone_conduction_function`, `our`, `librispeech-100`, `dev` (librispeech-dev), `rir_fullsubnet`, `background` and `music` should be placed on the same folder.
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
    python train.py
    ```
* Run the following code for model fine-tuning
    ```bash
    cd ./sample-code-UTD/supervised-baselines/
    python3 attnsense_main_ce.py --batch_size 16 --label_rate 5 --learning_rate 0.001
    python3 deepsense_main_ce.py --batch_size 16 --label_rate 5 --learning_rate 0.001
    python3 single_main_ce.py --modality inertial --batch_size 16 --label_rate 5 --learning_rate 0.001
    python3 single_main_ce.py --modality skeleton --batch_size 16 --label_rate 5 --learning_rate 0.001
    ```



# Citation
The code of this project are made available for non-commercial, academic research only. If you would like to use the code of this project, please cite the following paper:
```
@inproceedings{ouyang2022cosmo,
  title={Cosmo: contrastive fusion learning with small data for multimodal human activity recognition},
  author={Ouyang, Xiaomin and Shuai, Xian and Zhou, Jiayu and Shi, Ivy Wang and Xie, Zhiyuan and Xing, Guoliang and Huang, Jianwei},
  booktitle={Proceedings of the 28th Annual International Conference on Mobile Computing And Networking},
  pages={324--337},
  year={2022}
}
```
    