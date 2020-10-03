import torch
import os
import numpy as np
from imgDataset import ImgDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score,recall_score,precision_score, accuracy_score
from xrayNet import XrayNet
import seaborn as sns


model = XrayNet()
model_weights = torch.load('./models/pneumoniaClass_xrayNet_epoch5_valAcc0.6875_loss0.35.pth.tar')
model.load_state_dict(model_weights['state_dict'])
model.eval()

# get valid data paths
testData_path = [[], []]
testPath = r'C:\Users\user\Desktop\computer vision\x-rey project\xray dataset\chest_xray\test'
for classFolder in os.listdir(testPath):
    fpath = os.path.join(testPath, classFolder)
    for img_name in os.listdir(fpath):
        img_path = os.path.join(fpath, img_name)
        if classFolder == 'NORMAL':
            testData_path[0].append(img_path)
        else:
            testData_path[1].append(img_path)

test_labels = np.hstack((np.zeros(len(testData_path[0])), np.ones(len(testData_path[1]))))
testData_path = np.hstack((testData_path[0],testData_path[1]))

# form the dataloader
testDataset = ImgDataset(testData_path, test_labels, dim=(480, 360))
testDataLoader = DataLoader(testDataset, batch_size=1, shuffle=False)

# start testing
y_log = []
target_log = []
for i, (imgs, label) in enumerate(testDataLoader):
    with torch.no_grad():
        y_pred = model(imgs.to(model.device)).to('cpu').numpy()
        label = label.numpy()
        y_log.append((y_pred[0][1] > 0.5) * 1)
        target_log.append(label[0])

# plot confusion matrix
conf_matrix = confusion_matrix(target_log, y_log)
accu = accuracy_score(target_log, y_log)
f1 = f1_score(target_log, y_log)
precision = precision_score(target_log, y_log)
recall = recall_score(target_log, y_log)
print(f'accuracy: {accu:.2f}, f1:{f1:.2f}, precision: {precision:.2f}, recall:{recall:.2f}')
print(conf_matrix)

ax = plt.subplot()
sns.heatmap(conf_matrix, annot=True, ax=ax, cmap='Blues', fmt='3d')
# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['normal', 'pneumonia'])
ax.yaxis.set_ticklabels(['normal', 'pneumonia'])
plt.show()
