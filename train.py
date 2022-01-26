from prepare_dataset import getData
from dataset import Dataset,splitData

from torch.utils.data import DataLoader
import torch
import torchvision
from torch import nn
from tqdm import tqdm
import wandb


import argparse
parser = argparse.ArgumentParser(description='train model params')
parser.add_argument('--dataset_dir', type=str, default='archive/crop_part1')
args = parser.parse_args()

width = height = 224
batch_size = 16
dataset_dir = args.dataset_dir

print('preparing data ...')
X,Y = getData(dataset_dir,width,height)
print('data loaded')

dataset = Dataset(X, Y)
train_dataset = splitData(dataset, 0.2, True)
dataloader = DataLoader(train_dataset, batch_size=batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sweep_config = {
  "method": "random",
  "metric": {
    "name": "loss",
    "goal": "minimize" 
  },
    
  "parameters": {
      "learning_rate":{
          "min": 0.0001,
          "max": 0.01
      },
      "epochs": {
          'values': [10, 20]
      },   
  }
    
}

sweep_id = wandb.sweep(sweep_config, project="transfer learning and fine tuning")


# TRAIN 

counter = 0
def train():
  global counter
  counter += 1
  print('training ...')
  config_defaults = {
        'epochs': 5,
        'learning_rate': 0.001
    }
  wandb.init(config=config_defaults)
  config=wandb.config

  model = torchvision.models.resnet50(pretrained=True)
  in_features = model.fc.in_features
  model.fc = nn.Linear(in_features, 1)

  counter = 0
  for child in model.children():
    counter += 1
    if(counter < 7):
      for params in child.parameters():
        params.requires_grad = False

  model.to(device)
  wandb.watch(model)

  optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
  loss_function = nn.L1Loss()

  model.train()

  for epoch in range(config.epochs):
    train_loss = 0.0
    train_acc = 0.0
    for images, labels in tqdm(dataloader):

      images, labels = images.to(device), labels.to(device)

      optimizer.zero_grad()
      images = images.float()
      preds = model(images)

      loss = loss_function(preds,  labels.float())
      loss.backward()

      optimizer.step()

      train_loss += loss

    total_loss = train_loss / len(dataloader)
    print(f"Epoch: {epoch+1}, Loss: {total_loss}")

    wandb.log({'epochs':  epoch + 1,'loss': total_loss,})

  torch.save(model.state_dict(),'weights_config_' + counter + ".pth")


wandb.agent(sweep_id, train)