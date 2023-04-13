import torchvision.transforms as T
from torch import nn
import torch
import torch.optim as optim
import json
import numpy as np
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import ensemble_wbf as ensemble
from mmdet.core.bbox.assigners.approx_max_iou_assigner import MaxIoUAssigner

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WeightCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.denses = nn.Sequential(
            nn.Linear(3840*2160, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 5),
            nn.Softmax(dim=1)
        )

    def forward(self, z):
        output = self.denses(z)
        return output

def prepare_data():
    path2data="./data/mva2023_sod4bird_train/images"
    path2json="./data/mva2023_sod4bird_train/annotations/split_train_coco.json"
    coco_train = dset.CocoDetection(root = path2data, annFile = path2json)
    train_loader = DataLoader(
        dataset=coco_train, batch_size=16, shuffle=True, num_workers=4)
    return train_loader
    
def loss_func(input, target):
    def xywh2xyxy(xywh):
        bbox = [xywh[0], xywh[1], (xywh[0] + xywh[2]), (xywh[1] + xywh[3])]
        return bbox
    loss = 0
    for img in zip(input, target):
        img[1] = [xywh2xyxy(i) for i in img[1]]
        assign_result =  MaxIoUAssigner(img[1], img[0])
        gt_results = img[1][assign_result.gt_inds]
        pre_results = img[0]
        loss += nn.SmoothL1Loss(pre_results, gt_results)
    return loss


def model_train():
    model = WeightCNN().to(device)
    op = optim.Adam(model.parameters(), lr = 0.0008, weight_decay= 0.00003)
    train_loader = prepare_data()
    fig_train_loss_y = []
    for epoch in range(1, 10):

        # training
        model.train()
        correct = 0

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            batch_size = len(imgs)
            bboxes_list = []
            ids_list = []
            img_list = []
            for i in range(len(imgs)):
                bboxes = []
                ids = []
                img = imgs[i]
                labels_ = labels[i]
                for label in labels_:
                    bboxes.append([ label['bbox'][0],
                                    label['bbox'][1],
                                    label['bbox'][0] + label['bbox'][2],
                                    label['bbox'][1] + label['bbox'][3] 
                    ])
                    ids.append(label['category_id'])
                bboxes_list.append(bboxes)
                ids_list.append(ids)
                img_list.append(img)


            img_list, bboxes_list = img_list.to(device), bboxes_list.to(device)
            op.zero_grad()
            output = model(img_list)

            results = ensemble("config.txt", "wbf.json", weights=output)
            bboxes_predict = [ [] for i in range(batch_size)]
            
            for data in results:
                bboxes_predict[int(data['image_id'])].append(results['bbox'])

            # TODO: choose your own loss function
            loss = loss_func(bboxes_list, bboxes_predict)
            
            loss.backward()
            op.step()
            if batch_idx % 50 == 0:
                fig_train_loss_y.append(loss.item())
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                print('Train set:  Accuracy: {}/{} ({:.0f}%)\n'.format(
                        correct, len(train_loader.dataset),
                        100. * correct / len(train_loader.dataset)))