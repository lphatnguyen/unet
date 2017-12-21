import torch
import PIL.Image as Image
import torchvision
import numpy as np
import matplotlib.pyplot as plt

img = Image.open('./data/test/0bdd2e625f8a_12.jpg')

img = img.resize((64,64), Image.BILINEAR)
tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
image = tensor(img)
net = torch.load('trainedModel.pt')
inputImg = torch.autograd.Variable(image.unsqueeze(0).cuda())
outputImg = net(inputImg)
outputImg = torch.nn.functional.sigmoid(outputImg)
out = outputImg.squeeze(0).squeeze(0)
out = out.data.cpu().numpy()

out = out.astype(np.uint8)
plt.figure()
plt.imshow(out)
plt.figure()
plt.imshow(img)
