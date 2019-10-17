# -*- coding: utf-8 -*-

import torch
import argparse
import numpy as np
from sklearn.neighbors import NearestNeighbors
from model import PointCloudNet
import sampling

def get_best_epoch(f_pointer):
    f_pointer.seek(0, 0) # begining of file
    read_states = f_pointer.readlines()
    read_min_loss = min(read_states, key=lambda k: float(k.split(", ")[1].split(" ")[1][0:-1]))
    best_epoch = int(read_min_loss.split(", ")[0].split(" ")[1])
    return best_epoch, read_min_loss

class FurthestPointSampling(torch.autograd.Function):

    @staticmethod
    def forward(ctx, xyz, npoint):
        r"""
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor where N > npoint
        npoint : int32
            number of features in the sampled set
        Returns
        -------
        torch.LongTensor
            (B, npoint) tensor containing the indices
        """
        B, N, _ = xyz.size()

        idx = torch.empty([B, npoint], dtype=torch.int32, device=xyz.device)
        temp = torch.full([B, N], 1e10, dtype=torch.float32, device=xyz.device)

        sampling.furthest_sampling(
            B, N, npoint, xyz, temp, idx
        )
        ctx.mark_non_differentiable(idx)
        return idx


furthest_point_sample = FurthestPointSampling.apply


def extract_knn_patch(queries, pc, k):
    """
    queries [M, C]
    pc [P, C]
    """
    #print(queries.shape)
    #print(pc.shape)
    knn_search = NearestNeighbors(n_neighbors=k, algorithm='auto')
    knn_search.fit(pc[:, 0:3])
    knn_idx = knn_search.kneighbors(queries, return_distance=False)
    k_patches = np.take(pc, knn_idx, axis=0)  # M, K, C
    return k_patches



parser = argparse.ArgumentParser()
parser.add_argument('--num_points', default=1024, type=int, 
                    help='Number of points per patch')
parser.add_argument('--patch_num_ratio', default=3, type=int, 
                    help='Number of points per patch')
parser.add_argument('--trained_model', type=str, 
                    help='Trained model directory')
parser.add_argument('--test_file', type=str, 
                    help='XYZ file for testing')
FLAGS = parser.parse_args()

NUM_POINTS = FLAGS.num_points
PATCH_NUM_RATIO = FLAGS.patch_num_ratio
TRAINED_MODEL = FLAGS.trained_model
TEST_FILE = FLAGS.test_file


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#normaliaze data and extract patches
pc = torch.tensor(np.loadtxt(TEST_FILE)).float().to(device)
num_patches = int(pc.shape[0] / NUM_POINTS * PATCH_NUM_RATIO)
fps_idx = furthest_point_sample(torch.unsqueeze(pc[:, 0:3], dim=0).contiguous(), num_patches)
patches = torch.tensor(extract_knn_patch(pc[torch.squeeze(fps_idx, dim=0).cpu().numpy(), 0:3].cpu().numpy(), pc.cpu().numpy(), NUM_POINTS)).to(device) 
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
fps_idx = furthest_point_sample(torch.unsqueeze(up_points[:, 0:3], dim=0).contiguous(), pc.shape[0] * 4)
up_points = up_points[torch.squeeze(fps_idx, dim=0).cpu().numpy(), :].detach().cpu().numpy()
np.savetxt("temp/output_{0}".format(TEST_FILE), up_points, fmt='%.6f', delimiter=" ", newline="\n")
