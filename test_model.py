import torch
import PIL.Image as Image
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

path="./data/test/"
filename=os.listdir(path)
i=0

for string in filename:
    filename[i]=path+filename[i]
    img = Image.open(filename[i])
    
    img = img.resize((256,256), Image.BILINEAR)
    tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    image = tensor(img)
    net = torch.load('trainedModel2.pt')
    inputImg = torch.autograd.Variable(image.unsqueeze(0).cuda())
    outputImg = net(inputImg)
    outputImg = torch.nn.functional.sigmoid(outputImg)
    out = outputImg.squeeze(0).squeeze(0)
    out = out.data.cpu().numpy()
    out = out.astype(np.uint8)
    out=out*255
    i=i+1
    cv2.imwrite("./output/"+string,out)


plt.figure()
plt.imshow(out)
plt.figure()
plt.imshow(img)
