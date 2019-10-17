# -*- coding: utf-8 -*-

from subprocess import call
import os
import shutil


ModelNet40_Path = "/home/rajat/Desktop/research-project/work/python2/ModelNet40"


def create_data(datapath, dir_type):
    
    
    if os.path.isdir("../preprocessed_{0}".format(dir_type)): shutil.rmtree("../preprocessed_{0}".format(dir_type))
    os.mkdir("../preprocessed_{0}".format(dir_type))
    
    count_off = 0
    for _dir in os.listdir(datapath):
        if os.path.isdir(os.path.join(datapath, _dir)):
            off_dir = os.path.join(datapath, _dir, dir_type)
            for off_file in os.listdir(off_dir):
                if off_file.endswith(".off"):
                    op_off_file = off_file.split(".")[0]
                    if not os.path.exists(("../preprocessed_{0}/{1}_2500.pcd").format(dir_type, op_off_file)):
                        # convert obj to ply
                        call(["meshlab.meshlabserver", "-i", os.path.join(off_dir, off_file), "-o", ("../preprocessed_{0}/{1}.ply").format(dir_type, op_off_file)])
                        
                        # create ip (25000 sample), target for each step (1, 2, 3)
                        # step 1 tgt = 10000
                        # step 2 tgt = 40000
                        # step 3 tgt = 160000
                        # step 4 tgt = 640000
                        call(["pcl_mesh_sampling", "-n_samples", "2500", "-leaf_size", "0.001", "-no_vis_result", ("../preprocessed_{0}/{1}.ply").format(dir_type, op_off_file), ("../preprocessed_{0}/{1}_2500.pcd").format(dir_type, op_off_file)])
                        call(["pcl_mesh_sampling", "-n_samples", "10000", "-leaf_size", "0.001", "-no_vis_result", ("../preprocessed_{0}/{1}.ply").format(dir_type, op_off_file), ("../preprocessed_{0}/{1}_10000.pcd").format(dir_type, op_off_file)])
                        #call(["pcl_mesh_sampling", "-n_samples", "40000", "-leaf_size", "0.001", "-no_vis_result", ("../preprocessed_{0}/{1}.ply").format(dir_type, op_off_file), ("../preprocessed_{0}/{1}_40000.pcd").format(dir_type, op_off_file)])
                        #call(["pcl_mesh_sampling", "-n_samples", "160000", "-leaf_size", "0.001", "-no_vis_result", ("../preprocessed_{0}/{1}.ply").format(dir_type, op_off_file), ("../preprocessed_{0}/{1}_160000.pcd").format(dir_type, op_off_file)])
                        #call(["pcl_mesh_sampling", "-n_samples", "640000", "-leaf_size", "0.001", "-no_vis_result", ("../preprocessed_{0}/{1}.ply").format(dir_type, op_off_file), ("../preprocessed_{0}/{1}_640000.pcd").format(dir_type, op_off_file)])
                    count_off += 1
                    
                    
    print(count_off)
                    
                    
create_data(ModelNet40_Path, "test")