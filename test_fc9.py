from fc9 import segNet
from fc9_knn import DGCNN

import torch
from tqdm import tqdm
import numpy as np

model = segNet()
model.load_state_dict(torch.load('best_model.pt'))
test_dataloader = torch.load('shifted_test_dataloader.pt')

confusion_matrix = torch.zeros(3, 3)
class_map = {0: "floor", 1: "wall", 2: "ceiling"}

with torch.no_grad():
    model.eval()
    correct_per_class = np.zeros(3)
    total_per_class = np.zeros(3)
    for i, (data, labels) in enumerate(tqdm(test_dataloader)):
        data = data.float()
            
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        
        predicted = predicted.numpy()
        labels = labels.numpy().astype(int)
        
        for i in range(len(predicted)):
            confusion_matrix[labels[i], predicted[i]] += 1
            if predicted[i] == labels[i]:
                correct_per_class[labels[i]] += 1
            total_per_class[labels[i]] += 1

print("Done testing")
    

for i in range(3):
    print("Accuracy of class ", class_map[i], ":", correct_per_class[i]/total_per_class[i])
    
print("most accurate class: ", class_map[np.argmax(correct_per_class/total_per_class)])
print("least accurate class: ", class_map[np.argmin(correct_per_class/total_per_class)])
print(confusion_matrix)

    
