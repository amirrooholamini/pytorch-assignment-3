import argparse

import time
start = time.time()
import torch
import torchvision
import cv2 as cv
import numpy as np
import torchvision.transforms as transforms
from torch import nn
parser = argparse.ArgumentParser(description='inference model params')
parser.add_argument('--weights', type=str, default='weights_config_1.pth')
args = parser.parse_args()

inference_transform = transforms.Compose([
     transforms.ToPILImage(mode=None),
     transforms.Resize((224,224)),
     transforms.ToTensor(),
     transforms.Normalize((0),(1)),
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torchvision.models.resnet50(pretrained=False)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 1)
model.load_state_dict(torch.load(args.weight_file, map_location=torch.device(device)))
model.train(False)
model.eval()

model.train(False)
model.eval()

img = cv.imread(args.image)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = cv.resize(img,(args.image_size,args.image_size))
tensor = inference_transform(img).unsqueeze(0).to(device)
prediction = model(tensor).cpu().detach().numpy()
print(np.argmax(prediction, axis=1))
end = time.time()
print(f'time: {end-start} seconds')