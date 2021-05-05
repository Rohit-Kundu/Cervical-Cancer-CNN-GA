import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision import datasets,models,transforms
from torch import nn
import torch.nn.functional as F
from torch import optim
import os

data_path=".../.../"     # Provide the Dataset path here (whole dataset).
#5-fold Cross Validation will be applied during O-bHSA and so the features are to be extracted from ALL the images


# Model for Feature Extraction
new_model=models.googlenet(pretrained=True) #For Resnet18, change to models.resnet18(pretrained=True)

#redefine the model for feature extraction
class Net(nn.Module):
  def __init__(self,num_classes):
    super(Net, self).__init__()
    
    self.cnn_layer= torch.nn.Sequential(*(list(new_model.children())[:-1]))
    self.fc=torch.nn.Sequential(nn.Linear(1024,num_classes,bias=True), #For ResNet18 change 1024 to 518
                                nn.LogSoftmax(dim=1))
    
  def forward(self,x):
    out=self.cnn_layer(x)
    out = out.view(out.size(0),-1)
    x1=out
    out=self.fc(out)
    return x1,out

num_classes= len(os.listdir(data_path))# Number of classes
model=Net(num_classes)

print(model)

data_transforms=transforms.Compose([
                                    transforms.Resize((224,224)),
                                     transforms.ToTensor()
                                     ]) #No need to use data augmentation since only the pretrained features are to be extracted

dataset=datasets.ImageFolder(data_path,transform=data_transforms)
data_loader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)

model.eval()
features=[]
classes=[]
with torch.no_grad():
  for inputs,labels in data_loader:
    x1,output=model.forward(inputs)
    features.append((x1.detach().numpy().tolist())[0])   
    classes.append(labels.numpy().tolist()[0])

df=pd.DataFrame(features)
df2=pd.DataFrame(classes)
print(df.shape,df2.shape)

df.to_csv("/content/drive/My Drive/trial/googlenet_sipakmed.csv")     #Sipakmed features file
df2.to_csv("/content/drive/My Drive/trial/sipakmed_labels.csv")       #Sipakmed labels file
# Note that the saved csv files have NO HEADERS
