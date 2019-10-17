# -*- coding: utf-8 -*-

import os
import h5py
import numpy as np



def read_off_file_points(f_name):
    file = open(f_name, "r")
    lines = file.readlines()
    
    lines = np.array([[float(j) for j in i.split(" ")] for i in lines[11:]])
    
    #print(lines[0], lines[-1])
    return lines.transpose()


def valid_point_cloud(_dir, f_prefix):
    
    for step in [2500, 10000, 40000, 160000, 640000]:
        file = open(_dir + f_prefix + "_{0}.pcd".format(step), "r")
        point_count = int(file.readlines()[9].split(" ")[1])
        if point_count != step:
            print(f_prefix)
            return False
    
    return True
    
#with h5py.File("../data_2500_test.hdf5", "w") as hf_2500, \
with    h5py.File("../data_10000_test.hdf5", "w") as hf_10000:#, \
        #h5py.File("../data_40000_test.hdf5", "w") as hf_40000, \
        #h5py.File("../data_160000_test.hdf5", "w") as hf_160000:
            
            #s = 2468 # dataset size (test)
            s = 9843 # dataset size (train)
            
            # create data set for each sampled points
            #trn_set_2500 = hf_2500.create_dataset("train_set", [s, 3, 2500], dtype="f")
            #tgt_set_2500 = hf_2500.create_dataset("tgt_set", [s, 3, 10000], dtype="f")
            #trn_set_2500 = hf_2500["train_set"]
            #tgt_set_2500 = hf_2500["tgt_set"]
            trn_set_10000 = hf_10000.create_dataset("train_set", [s, 3, 10000], dtype="f")
            tgt_set_10000 = hf_10000.create_dataset("tgt_set", [s, 3, 40000], dtype="f")
            #trn_set_10000 = hf_10000["train_set"]
            #tgt_set_10000 = hf_10000["tgt_set"]
            #trn_set_40000 = hf_40000.create_dataset("train_set", [s, 3, 40000], dtype="f")
            #tgt_set_40000 = hf_40000.create_dataset("tgt_set", [s, 3, 160000], dtype="f")
            #trn_set_40000 = hf_40000["train_set"]
            #tgt_set_40000 = hf_40000["tgt_set"]
            #trn_set_160000 = hf_160000.create_dataset("train_set", [s, 3, 160000], dtype="f")
            #tgt_set_160000 = hf_160000.create_dataset("tgt_set", [s, 3, 640000], dtype="f")
            
            
            c = 0
            
            temp = []
            
            for pcd_files in os.listdir("../preprocessed_train"):
                f = pcd_files.split("_")
                #print(f)
                f_prefix = "_".join(f[0:-1])
                if f_prefix not in temp:
                    if valid_point_cloud("../preprocessed_train/", f_prefix):
                        #print("Adding dataset for {0}".format(f_prefix))
                        #trn_set_2500[c] = read_off_file_points("../preprocessed_test/" + "_".join([f_prefix, "2500.pcd"]))
                        #tgt_set_2500[c] = read_off_file_points("../preprocessed_test/" + "_".join([f_prefix, "10000.pcd"]))
                        tgt_set_10000[c] = read_off_file_points("../preprocessed_train/" + "_".join([f_prefix, "40000.pcd"]))
                        #tgt_set_40000[c] = read_off_file_points("../preprocessed_test/" + "_".join([f_prefix, "160000.pcd"]))
                        #tgt_set_160000[c] = read_off_file_points("../preprocessed_test/" + "_".join([f_prefix, "640000.pcd"]))
                        c += 1
                    temp.append(f_prefix)
            
            print(c)