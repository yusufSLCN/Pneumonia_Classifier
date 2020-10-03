from xrayNet import XrayNet
from imgDataset import ImgDataset
import os
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2

#get train data paths
path = r'C:\Users\user\Desktop\computer vision\x-rey project\xray dataset\chest_xray\train'
trainData_path = [[], []]
for classFolder in os.listdir(path):
    fpath = os.path.join(path, classFolder)
    for img_name in os.listdir(fpath):
        img_path = os.path.join(fpath, img_name)
        if classFolder == 'NORMAL':
            trainData_path[0].append(img_path)
        else:
            trainData_path[1].append(img_path)

#upsample negative samples
np.random.seed = 42
newSampleInd = np.random.randint(0, len(trainData_path[0]), len(trainData_path[1]) - len(trainData_path[0]), dtype=int)
upsamples = np.array(trainData_path[0])[newSampleInd]
trainData_path[0] = np.hstack((trainData_path[0], upsamples))
# print(len(trainData_path[0]))

labels = np.hstack((np.zeros(len(trainData_path[0])), np.ones(len(trainData_path[1]))))
trainData = np.hstack((trainData_path[0], trainData_path[1]))

# show an example xray image
img = cv2.imread(trainData[0], 0)
plt.imshow(img, cmap='gray')
plt.title(f'{labels[0]} labeled sample')
plt.pause(2)
plt.show(block=False)
plt.close()

# get valid data paths
validData_path = [[], []]
valPath = r'C:\Users\user\Desktop\computer vision\x-rey project\xray dataset\chest_xray\val'
for classFolder in os.listdir(valPath):
    fpath = os.path.join(valPath, classFolder)
    for img_name in os.listdir(fpath):
        img_path = os.path.join(fpath, img_name)
        if classFolder == 'NORMAL':
            validData_path[0].append(img_path)
        else:
            validData_path[1].append(img_path)

val_labels = np.hstack((np.zeros(len(validData_path[0])), np.ones(len(validData_path[1]))))
validData_path = np.hstack((np.array(validData_path[0]), np.array(validData_path[1])))

validDataset = ImgDataset(validData_path, val_labels, dim=(480, 360))
trainDataset = ImgDataset(trainData, labels, dim=(480, 360))

# set dataloaders
batchSize = 16
trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
valLoader = DataLoader(validDataset, batch_size=8, shuffle=False)

# set the model
model = XrayNet()
criterion = nn.CrossEntropyLoss().to(model.device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# start training
epochNo = 5
batchNo = round(len(trainDataset) / batchSize)
lossLog = []
accLog = []
valAccLog = []

for epoch in range(epochNo):
    epochLoss = 0
    epochAcc = []
    model.train()
    for i, (data, label) in enumerate(trainLoader):

        optimizer.zero_grad()
        data = data.to(model.device)
        target = label.long().to(model.device)

        output = model(data)
        loss = criterion(output, target)
        epochLoss += loss.item()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if not i % 10:
                y_true = label.to('cpu').numpy()
                y_pred = output.to('cpu').numpy()
                y_pred = (y_pred[:, 1] > 0.5) * 1
                print(f"Epoch {epoch + 1}, Batch {i} / {batchNo} loss: {loss.item():.3f} ", end=' ')
                epochAcc.append(np.mean(y_true == y_pred))
                print(f'average acc: {np.mean(epochAcc):.3f}')
    accLog.append(np.mean(epochAcc))
    lossLog.append(epochLoss/batchNo)

    #evaluate on validation data
    valAcc = []
    model.eval()
    with torch.no_grad():
        for i, (data, label) in enumerate(valLoader):
            data = data.to(model.device)
            out = model(data).to('cpu').numpy()
            label = label.numpy()
            out = (out[:, 1] > 0.5)*1
            valAcc.append(np.mean(out == label))

        print(f'Epoch {epoch + 1}, Val acc: {np.mean(valAcc)} \n')
    valAccLog.append(np.mean(valAcc))

#save the model
torch.save({'state_dict': model.state_dict()},
           f'./models/pneumoniaClass_xrayNet_epoch{epochNo}_valAcc{np.mean(valAcc)}_loss{lossLog[-1]:.2f}.pth.tar')

plt.plot(lossLog, label='train_loss')
plt.plot(accLog, label='train_acc')
plt.plot(valAccLog, label='val_acc')
plt.legend()
plt.title('Classifier Training')
plt.xlabel('Epoch')
plt.show()

