# Efficient Point Cloud Upsampling and Normal Estimation using Deep Learning for Robust Surface Reconstruction

To run the project add root folder of the project to python path.```export PYTHONPATH="ROOTPATH_OF_PROJECT:$PYTHONPATH"``` e.g., ```export PYTHONPATH="/home/user/point-normals-upsampling:$PYTHONPATH"```

## Setup
- Use anaconda for python3.7. Install ```requirements.txt```. Install torch and cuda toolkit ```conda install pytorch torchvision cudatoolkit=10.1 -c pytorch```
- Build pointnet++ module run ```python setup.py build_ext --inplace``` in root folder of project
- Build sampling module run ```python setup.py install``` in sampling folder of project
- Add absolute path of chamfer_distace.cpp and chamfer_distance.cu in chamfer_distance.py

## Training

- For this repo we used [PU-NET](https://raw.githubusercontent.com/yulequan/PU-Net) dataset for training. Download the hdf5 format patches dataset from [GoogleDrive](https://drive.google.com/file/d/1wMtNGvliK_pUTogfzMyrz57iDb_jSQR8/view?usp=sharing)
- For training and evalutation run all commands inside code folder.
- Training: ```python train.py --num_points 1024 --checkpoint_path .. --batch_size 20 --epochs 400 --h5_data_file dataset_path``` e.g., ```python train.py --num_points 1024 --checkpoint_path .. --batch_size 20 --epochs 400 --h5_data_file ../data.h5```
- Evaluation: ```python evaluate.py --test_file filename(.xyz) --num_points num (default=1024) --patch_num_ratio num (default=4) --trained_model checkpoint_path``` e.g., ```python evaluate.py --test_file ../test.xyz --num_points 1024 --patch_num_ratio 4 --trained_model ../checkpoint```
- All the results will be saved in results folder in root directory

## Acknowledgement
- **PointNet++ PyTorch Implementation**: [erikwijmans/Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch)
- **Official PyTorch**: [charlesq34/pointnet2](https://github.com/charlesq34/pointnet2)
- **PyTorch Chamfer Distance**: [chrdiller/pyTorchChamferDistance](https://github.com/chrdiller/pyTorchChamferDistance)
- **Patch-base progressive 3D Point Set Upsampling**: [yifita/3PU_pytorch](https://github.com/yifita/3PU_pytorch)
