# Efficient Point Cloud Upsampling and Normal Estimation using Deep Learning for Robust Surface Reconstruction

To run the project add root folder of the project to python path.

## Setup
- Install ```requirements.txt``` to install pytorch. It can also be installed using anaconda
- Build pointnet++ module run ```python setup.py build_ext --inplace``` in root folder of project
- Add absolute path of chamfer_distace.cpp and chamfer_distance.cu in chamfer_distance.py

## Training

- For this repo we used [PU-NET](https://raw.githubusercontent.com/yulequan/PU-Net) dataset for training. Download the hdf5 format patches dataset from [GoogleDrive](https://drive.google.com/file/d/1wMtNGvliK_pUTogfzMyrz57iDb_jSQR8/view?usp=sharing)
- ```shell
    python train.py --num_points 1024 --checkpoint_path .. --batch_size 20 --epochs 400 --h5_data_file dataset_path
    ```

## Acknowledgement
- **PointNet++ PyTorch Implementation**: [erikwijmans/Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch)
- **Official PyTorch**: [charlesq34/pointnet2](https://github.com/charlesq34/pointnet2)
- **PyTorch Chamfer Distance**: [chrdiller/pyTorchChamferDistance](https://github.com/chrdiller/pyTorchChamferDistance)
