import torch
from modules import AlexNet
from torch import distributed
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

device = torch.device("cuda")

model = AlexNet.AlexNet()
# ckpt = torch.load("../results/office/single_office_AlexNet/amazon.ckpt")
# ckpt = torch.load("../results/7.ckpt")

# ckpt = torch.load("../results/amazon")
# model.load_state_dict(ckpt['model'])

model.load_state_dict(torch.load("../results/12.ckpt")['model'])

# model = torch.load("../results/ckpt")
model = model.to(device)

label_dict = {'back_pack': 0, 'bike': 1, 'calculator': 2, 'headphones': 3, 'keyboard': 4, 'laptop_computer': 5,
              'monitor': 6, 'mouse': 7, 'mug': 8, 'projector': 9}
img = Image.open("./1.jpg")

if len(img.split()) != 3:
    image = transforms.Grayscale(num_output_channels=3)(img)

transform_test = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.ToTensor()
])

des = transform_test(img)

des = torch.unsqueeze(des, dim=0)
ori = torch.cat((des, des), dim=0)
# for i in range(30):
#     ori = torch.cat((ori, des), dim=0)

ori = ori.to(device)

print(ori.shape)
outputs = model(ori)

print(outputs)
print(outputs.data.max(1))
print(outputs.data.max(1)[1])
