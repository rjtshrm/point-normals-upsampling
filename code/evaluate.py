# -*- coding: utf-8 -*-

import torch
import argparse
import numpy as np
from model import PointCloudNet
from code.utils import fp_sampling, knn_patch 
import os

def get_best_epoch(f_pointer):
    f_pointer.seek(0, 0) # begining of file
    read_states = f_pointer.readlines()
    read_min_loss = min(read_states, key=lambda k: float(k.split(", ")[1].split(" ")[1][0:-1]))
    best_epoch = int(read_min_loss.split(", ")[0].split(" ")[1])
    return best_epoch, read_min_loss


parser = argparse.ArgumentParser()
parser.add_argument('--num_points', default=1024, type=int, 
                    help='Number of points per patch')
parser.add_argument('--patch_num_ratio', default=4, type=int, 
                    help='Number of points per patch')
parser.add_argument('--trained_model', type=str, 
                    help='Trained model directory')
parser.add_argument('--test_file', type=str, 
                    help='XYZ file for testing')
FLAGS = parser.parse_args()


if not os.path.exists("../results"):
    os.mkdir("../results")

NUM_POINTS = FLAGS.num_points
PATCH_NUM_RATIO = FLAGS.patch_num_ratio
TRAINED_MODEL = FLAGS.trained_model
TEST_FILE = FLAGS.test_file
f_name = TEST_FILE.split("/")[-1]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#normaliaze data and extract patches
pc = torch.tensor(np.loadtxt(TEST_FILE)).float().to(device)
num_patches = int(pc.shape[0] / NUM_POINTS * PATCH_NUM_RATIO)
fps_idx = fp_sampling.furthest_point_sample(torch.unsqueeze(pc[:, 0:3], dim=0).contiguous(), num_patches)
patches = torch.tensor(knn_patch.extract_knn_patch(pc[torch.squeeze(fps_idx, dim=0).cpu().numpy(), 0:3].cpu().numpy(), pc.cpu().numpy(), NUM_POINTS)).to(device) 
print(patches.shape)

centroid = torch.mean(patches[:, :, 0:3], dim=1, keepdim=True)
patches[:, :, 0:3] = patches[:, :, 0:3] - centroid
furthest_distance = torch.max(torch.sqrt(torch.sum(patches[:, :, 0:3] ** 2, dim=-1)), dim=1,keepdim=True).values
patches[:, :, 0:3] = patches[:, :, 0:3] / torch.unsqueeze(furthest_distance, dim=-1)


# read best epoch from trained model
trained_model_state = open("{0}/state.txt".format(TRAINED_MODEL), "r")

best_epoch, read_min_loss = get_best_epoch(trained_model_state)
print(best_epoch, read_min_loss)
print("Best epoch (i.e., minimum  loss) for {0}".format(read_min_loss))

#initialize model
net = PointCloudNet(3, 6, True, NUM_POINTS).to(device)

model = torch.load("{0}/epoch_{1}.pt".format(TRAINED_MODEL, best_epoch))
net.load_state_dict(model["model_state_dict"])
net.eval()


up_patches = net(patches)
    
#denormalize and merge patches
up_patches[:, :, 0:3] = up_patches[:, :, 0:3] * torch.unsqueeze(furthest_distance, dim=-1) + centroid
up_points = torch.cat([p for p in up_patches], dim=0)
fps_idx = fp_sampling.furthest_point_sample(torch.unsqueeze(up_points[:, 0:3], dim=0).contiguous(), pc.shape[0] * 4)
up_points = up_points[torch.squeeze(fps_idx, dim=0).cpu().numpy(), :].detach().cpu().numpy()
np.savetxt("temp/{0}".format(f_name), up_points, fmt='%.6f', delimiter=" ", newline="\n")
