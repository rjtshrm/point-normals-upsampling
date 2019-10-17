import torch
import torch.nn.functional as f
from pyTorchChamferDistance.chamfer_distance import ChamferDistance

def knns_dist(xyz1, xyz2, k, device):
    """
        Parameters
        ----------
        samples: Number of points in xyz1
        xyz1: B * N * 6
        xyz2: B * N * 6
        k: number of points in xyz2 which are least distant to xyz1
        
        Returns
        ----------
        k number of points  in xyz2 which are least distant to xyz1
    """
    samples = xyz1.shape[1]
    xyz1_xyz1 = torch.bmm(xyz1, xyz1.transpose(2, 1))
    xyz2_xyz2 = torch.bmm(xyz2, xyz2.transpose(2, 1))
    xyz1_xyz2 = torch.bmm(xyz1, xyz2.transpose(2, 1))
    diag_ind_x = torch.arange(0, samples).to(device)
    rx = xyz1_xyz1[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(xyz1_xyz2.transpose(2,1))
    ry = xyz2_xyz2[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(xyz1_xyz2)
    pair_wise_loss = rx.transpose(2,1) + ry - 2 * xyz1_xyz2

    top_min_k = torch.topk(pair_wise_loss, k, dim=2, largest=False)

    return top_min_k


def l2_normal_loss(xyz1, xyz2, device):
    """
        Parameters
        ----------
        xyz1: B * N * 6
        xyz2: B * N * 6
        
        Returns
        ----------
        l2 normal loss for points which are closer in points1 & points2
    """
    batch = xyz1.shape[0]
    num_points = xyz1.shape[1]
    channels = xyz1.shape[2]
    
    # get indices of points1, points2 which have minimum distgance
    get_knn = knns_dist(xyz1[:, :, 0:3], xyz2[:, :, 0:3], k=1, device=device)
    
    k_indices = get_knn.indices
    k_values = get_knn.values
    
    k_points = torch.gather(xyz2.unsqueeze(1).expand(-1, xyz1.size(1), -1, -1), 
            2, 
            k_indices.unsqueeze(-1).expand(-1, -1, -1, xyz2.size(-1)))
    
    #dist = torch.mean(torch.sum((points1.view(batch, num_points, 1, channels)[:, :, :, 0:3] - k_points[:, :, :, 0:3]) ** 2, dim=-1))
    normal_loss = torch.mean(torch.sum((xyz1.view(batch, num_points, 1, channels)[:, :, :, 3:6] - k_points[:, :, :, 3:6]) ** 2, dim=-1))
    return normal_loss #, dist

def knn_loss(xyz, k_point, k_normal, device):
    """
        Parameters
        ----------
        points: B * N * 6
        k_point: number of neighbour for point regularization
        k_normal: number of neighbour for normal regularization
        
        Returns
        ----------
        cosine_normal_loss
        normal_neighbor_loss
        point_neighbor_loss
    """
    k = max(k_point, k_normal)
    k = k + 1  # a point also includes itself in knn search
    batch = xyz.shape[0]
    num_points = xyz.shape[1]
    channels = xyz.shape[2]

    get_knn = knns_dist(xyz[:, :, 0:3], xyz[:, :, 0:3], k, device)
    
    k_indices = get_knn.indices
    
    kv = get_knn.values
    kp = torch.gather(xyz.unsqueeze(1).expand(-1, xyz.size(1), -1, -1), 
            2, 
            k_indices.unsqueeze(-1).expand(-1, -1, -1, xyz.size(-1)))
    
    #print(kp.shape)
    #print(kv.shape)
    #print(kp[:, :, 0, :].view(batch, num_points, 1, channels)[:, :, :, 0:3].shape)
    p_dist = kp[:, :, 0, :].view(batch, num_points, 1, channels)[:, :, :, 0:3] - kp[:, :, 0:k_point+1, 0:3]
    # remove first column of each row as it is the same point from where min distance is calculated
    p_dist = p_dist[:, :, 1:, :]
    point_neighbor_loss = torch.mean(torch.sum(p_dist ** 2, dim=-1))
    #print(p_dist)
    #print(point_neighbor_loss)

    n_dist = kp[:, :, 0, :].view(batch, num_points, 1, channels)[:, :, :, 3:6] - kp[:, :, 0:k_normal+1, 3:6]
    # remove first column of each row as it is the same point from where min distance is calculated
    n_dist = n_dist[:, :, 1:, :]
    #print(n_dist)
    normal_neighbor_loss = torch.mean(torch.sum(n_dist ** 2, dim=-1))

    #print(normal_neighbor_loss)

    dot_product = f.normalize(p_dist, p=2, dim=-1) * f.normalize(kp[:, :, 0, :].view(batch, num_points, 1, channels)[:, :, :, 3:6], p=2, dim=-1)
    #print(dot_product)
    cosine_normal_loss = torch.mean(torch.abs(torch.sum(dot_product, dim=-1)))
    #print(normal_loss)
    return cosine_normal_loss, normal_neighbor_loss, point_neighbor_loss


if __name__ == "__main__":

    a = torch.tensor(
            [
                [[1, 2, 3, 0.1, 0.2, 0.3], [3, 2, 1, 0.3, 0.2, 0.1], [2, 4, 2, 0.2, 0.4, 0.2], [5, 1, 2, 0.5, 0.1, 0.2]],
                [[5, 3, 2, 0.5, 0.3, 0.2], [0, 1, 0, 0.0, 0.1, 0.0], [3, 0, 6, 0.3, 0.0, 0.6], [3, 2, 1, 0.3, 0.2, 0.1]]
            ],
            dtype=torch.float
            )
    b = torch.tensor(
            [
                [[1.1, 2.1, 3.1, 0.1, 0.2, 0.3], [3, 2, 1, 0.3, 0.2, 0.1], [2, 4, 2, 0.2, 0.4, 0.2], [15, 11, 12, 0.5, 0.1, 0.2]],
                [[5.1, 3.1, 2.1, 0.5, 0.3, 0.2], [10, 11, 10, 0.0, 0.1, 0.0], [3, 0, 6, 0.3, 0.0, 0.6], [3, 2, 1, 0.3, 0.2, 0.1]]
            ],
            dtype=torch.float
            )
    nl = knn_loss(a, 2, 2, "cuda:0")
    ecdn, ecdv = l2_normal_loss(a, b, device="cuda:0")
    print(nl)
    #print(ecdn, ecdv)
    #print(a, b)
