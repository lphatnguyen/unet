import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Unet import UNet
from carvana_dataset import CarvanaDataset
import numpy as np
import matplotlib.pyplot as plt

img_csv_file = './data/train_masks.csv'
train_img_dir = './data/train'
train_mask_dir = './data/train_masks_png'
dataset = CarvanaDataset(img_csv_file, train_img_dir, train_mask_dir)
trainLoader = DataLoader(dataset, shuffle = True, batch_size = 4)
net = UNet().cuda()

loss_fn = torch.nn.MultiLabelSoftMarginLoss()
opt = torch.optim.SGD(net.parameters(),lr = 0.000001,momentum=0.5)
lossValue = []
opt.zero_grad()
    
for epoch in range(7):
    runningLoss = 0.0
    for i, datum in enumerate(trainLoader):
        img, label = datum
        inputImg, lbl = Variable(img.cuda()), Variable(label.cuda())
        imgOut = net(inputImg)
        imgOut = imgOut.squeeze(0)
        imgOut = torch.nn.functional.sigmoid(imgOut)
        loss = loss_fn(imgOut, lbl)
        loss.backward()
        opt.step()
        runningLoss += loss.data[0]
        if i%500 == 499:
            print("Loss: ", runningLoss/500)
            lossValue.append(runningLoss/500)
            runningLoss = 0

plt.figure()
plt.plot(lossValue)
torch.save(net,'trainedModel.pt')

