import argparse
import glob
# import warnings
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.ATTSR import ATTSR
from tqdm import tqdm

from utils.dataset import build_dataloader


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def get_args():
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument('--data_root', type=str, default='./data/Austria', help='Root directory of the dataset')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save model')
    return parser.parse_args()


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Slope(nn.Module):
    def __init__(self):
        super(Slope, self).__init__()
        weight1 = np.zeros(shape=(3, 3), dtype=np.float32)
        weight2 = np.zeros(shape=(3, 3), dtype=np.float32)
        weight1[0][0] = -1
        weight1[0][1] = 0
        weight1[0][2] = 1
        weight1[1][0] = -2
        weight1[1][1] = 0
        weight1[1][2] = 2
        weight1[2][0] = -1
        weight1[2][1] = 0
        weight1[2][2] = 1

        weight2[0][0] = -1
        weight2[0][1] = -2
        weight2[0][2] = -1
        weight2[1][0] = 0
        weight2[1][1] = 0
        weight2[1][2] = 0
        weight2[2][0] = 1
        weight2[2][1] = 2
        weight2[2][2] = 1
        weight1 = np.reshape(weight1, (1, 1, 3, 3))
        weight2 = np.reshape(weight2, (1, 1, 3, 3))
        weight1 = weight1 / (8 * 10)
        weight2 = weight2 / (8 * 10)
        self.weight1 = nn.Parameter(torch.tensor(weight1))  # 自定义的权值
        self.weight2 = nn.Parameter(torch.tensor(weight2))
        self.bias = nn.Parameter(torch.zeros(1))  # 自定义的偏置
    def forward(self, x):
        dx = F.conv2d(x, self.weight1, self.bias, stride=1, padding=1)
        dy = F.conv2d(x, self.weight2, self.bias, stride=1, padding=1)
        ij_slope = torch.sqrt(torch.pow(dx, 2) + torch.pow(dy, 2))
        ij_slope = torch.arctan(ij_slope) * 180 / math.pi
        return ij_slope

compute_gradient_map = Slope().cuda()

class HybirdLoss(torch.nn.Module):
    def __init__(self):
        super(HybirdLoss, self).__init__()
        self.MSE = nn.MSELoss().to(DEVICE)
        self.L1 = nn.L1Loss()

    def forward(self, pred, mask):
        gradient_target = compute_gradient_map(mask)
        gradient_output = compute_gradient_map(pred)
        MSEloss = self.MSE(pred , mask)
        L1loss = self.L1(pred,mask)

        Lterrain = self.MSE(gradient_output, gradient_target)
        loss = MSEloss + L1loss+ 0.001*Lterrain

        return loss

def load_model(DEVICE):
    model = ATTSR(args, scale_factor=3)
    model.to(DEVICE)
    return model


def save_losses(losses_history, path):
    with open(path, 'w') as f:
        for epoch, loss in enumerate(losses_history):
            f.write(f"Epoch {epoch + 1}: Loss = {loss:.4f}\n")


def train(num_epochs, optimizer, scheduler, loss_fn, train_loader, model, save_path, loss_save_path):
    best_loss = float('inf')
    best_loss_epoch = 0
    losses_history = []

    for epoch in range(num_epochs):
        model.train()
        losses = []
        with tqdm(train_loader) as iterator:
            for j, (x, y, base_max, base_min) in enumerate(iterator):
                optimizer.zero_grad()
                x, y = x.to(DEVICE), y.to(DEVICE)

                lr = F.interpolate(y, scale_factor=1 / 3, mode='nearest',
                                   align_corners=None)
                lr = lr.to(DEVICE)

                output1, output2, output3 = model(lr, x)

                for i in range(output1.size(0)):
                    output1[i] = ((output1[i] + 1) / 2) * (base_max[i] - base_min[i] + 10) + base_min[i]
                    output2[i] = ((output2[i] + 1) / 2) * (base_max[i] - base_min[i] + 10) + base_min[i]
                    output3[i] = ((output3[i] + 1) / 2) * (base_max[i] - base_min[i] + 10) + base_min[i]
                    y[i] = ((y[i] + 1) / 2) * (base_max[i] - base_min[i] + 10) + base_min[i]

                loss1 = loss_fn(output1, y)
                loss2 = loss_fn(output2, y)
                loss3 = loss_fn(output3, y)
                loss = (loss1+loss2+loss3)/3

                loss.backward()
                optimizer.step()
                losses.append(loss.item())

        scheduler.step()
        train_loss = np.array(losses).mean()
        losses_history.append(train_loss)

        if train_loss < best_loss:
            best_loss = train_loss
            best_loss_epoch = epoch
            torch.save(model.state_dict(), save_path)
            tqdm.write(f'\rEpoch {epoch + 1}: Train Loss = {train_loss:.4f}, model saved')
        else:
            tqdm.write(f'\rEpoch {epoch + 1}: Train Loss = {train_loss:.4f}')

        save_losses(losses_history, loss_save_path)

        if (epoch - best_loss_epoch) >= 25:
            break


if __name__ == '__main__':
    random_seed = 5
    num_epochs = 100
    batch_size = 8
    lr = 1e-4
    setup_seed(random_seed)
    args = get_args()

    train_img_path = os.path.join(args.data_root, 'img')
    train_dem_path = os.path.join(args.data_root, 'dem')

    x_data = sorted(glob.glob(os.path.join(train_img_path, '*.tif')))
    y_data = sorted(glob.glob(os.path.join(train_dem_path, '*.tif')))

    dataset = [x_data, y_data]
    train_loader, __ = build_dataloader(dataset, dataset, int(batch_size))

    model_save_path = os.path.join(args.save_dir, 'ATTSR.pth')
    loss_save_path = os.path.join(args.save_dir, 'ATTSR.txt')

    model = load_model(DEVICE)

    optimizer = torch.optim.AdamW([dict(params=model.parameters(), lr=lr)])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=90, gamma=0.1)

    loss = HybirdLoss().cuda()

    train(num_epochs, optimizer, scheduler, loss, train_loader, model, model_save_path, loss_save_path)