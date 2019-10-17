# Efficient Point Cloud Upsampling and Normal Estimation using Deep Learning for Robust Surface Reconstruction

To run the project add root folder of the project to python path.

##### Training

- For this repo we used [PU-NET](https://raw.githubusercontent.com/yulequan/PU-Net) dataset for training. Download the hdf5 format patches dataset from [GoogleDrive](https://drive.google.com/file/d/1wMtNGvliK_pUTogfzMyrz57iDb_jSQR8/view?usp=sharing)
- ```shell
    python train.py --num_points 1024 --checkpoint_path .. --batch_size 20 --epochs 400 --h5_data_file dataset_path
    ```