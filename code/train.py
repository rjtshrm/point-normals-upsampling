# -*- coding: utf-8 -*-

import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
print(sys.path)
from dataset import PointCLoudDataset
from code.loss import knn_loss, l2_normal_loss, ChamferDistance
from code.model import PointCloudNet
from code.utils.helper_function import get_current_state
from code.utils import data_provider




parser = argparse.ArgumentParser()
parser.add_argument('--num_points', default=1024, type=int, 
                    help='Number of points per patch')
parser.add_argument('--checkpoint_path', default="..", type=str, 
                    help='Folder path to save checkpoint after each epoch')
parser.add_argument('--batch_size', default=20, type=int, 
                    help='Batch Size')
parser.add_argument('--epochs', default=500, type=int, 
                    help='Number of epochs')
parser.add_argument('--lr', default=5e-4, type=float, 
                    help='Learning Rate')
parser.add_argument('--weight_decay', default=1e-5, type=float, 
                    help='Weight Decay')
parser.add_argument('--add_noise', default=True, type=bool, 
                    help='Add Gaussian Noise')
parser.add_argument('--h5_data_file', type=str, 
                    help='Training h5 file path')
FLAGS = parser.parse_args()


NUM_POINTS = FLAGS.num_points
CHECKPOINT_PATH = FLAGS.checkpoint_path
BATCH_SIZE = FLAGS.batch_size
EPOCHS = FLAGS.epochs
LR = FLAGS.lr
WD = FLAGS.weight_decay
ADD_NOISE = FLAGS.add_noise
H5_FILE = FLAGS.h5_data_file

print(FLAGS)




# use gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# create directory to save checkpoints
checkpoint = os.path.join(CHECKPOINT_PATH, "checkpoint")
if not os.path.exists(checkpoint):
    os.mkdir(checkpoint)



# initialize chamfer loss function
chamfer_dist = ChamferDistance()

# initialize upsampling network
net = PointCloudNet(3, 6, True, NUM_POINTS).to(device)

#define optimizer
optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=WD)


# read and get last state of check point if already training started
state = open(os.path.join(checkpoint, "state.txt"), "a+")
start_epoch = get_current_state(state)

if start_epoch == -1:
    start_epoch = 0
elif start_epoch == EPOCHS:
    print("Final epoch {0} already Trained.".format(start_epoch))
else:
    # get last state
    model = torch.load("{0}/epoch_{1}.pt".format(checkpoint, start_epoch))
    net.load_state_dict(model["model_state_dict"])
    optimizer.load_state_dict(model["optimizer_state_dict"])
    
    print("Loaded Checkpoint ::: last  trained epoch epoch_{0} with loss {1}".format(start_epoch, model["loss"]))
    

print("TRAININNG STARTED")
print("Starting from epoch {0}".format(start_epoch + 1))


pcDataset = PointCLoudDataset(H5_FILE, transform=True)
trainloader = DataLoader(pcDataset, batch_size=BATCH_SIZE, shuffle=True)

trainloader_len = len(trainloader) 
print(trainloader_len)

# start training
for epoch in range(start_epoch, EPOCHS):
    
    running_loss = 0.0
    for i, data in enumerate(trainloader):

        inputs, targets = data_provider.data_augmentation(data[0].numpy(), data[1].numpy())
        if ADD_NOISE:
            inputs = data_provider.add_noise(inputs)
        inputs, targets = torch.from_numpy(inputs).float().to(device), torch.from_numpy(targets).float().to(device)
        
        # make gradient zeros
        optimizer.zero_grad()
        
        # forward + backward + optimize
        predicted_outputs = net(inputs)
        dist1, dist2 = chamfer_dist(targets[:, :, 0:3], predicted_outputs[:, :, 0:3])
        n_loss = l2_normal_loss(targets, predicted_outputs, device)
        cosine_normal_loss, normal_neighbor_loss, point_neighbor_loss = knn_loss(predicted_outputs, 5, 5, device)
        loss = (torch.mean(dist1)) + (torch.mean(dist2)) + (0.1 * point_neighbor_loss) + (0.05 * n_loss) + (0.0001 * cosine_normal_loss) + (0.0001 * normal_neighbor_loss)

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if i % 50 == 49:
            print("EPOCH {0}, BATCH {1}, LOSS {2}".format(epoch + 1, i + 1, loss.item()))
            
    print("EPOCH {0}, LOSS {1}".format(epoch + 1, running_loss / trainloader_len))
    
    torch.save({
            'epoch': epoch + 1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss / trainloader_len
            }, "{0}/epoch_{1}.pt".format(checkpoint, epoch + 1))
    
    state.write("EPOCH {0}, TRAINING_LOSS {1}\n".format(epoch + 1, running_loss / trainloader_len))

