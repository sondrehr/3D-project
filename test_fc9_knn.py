from fc9_knn import DGCNN

import torch
from tqdm import tqdm
import numpy as np

model = DGCNN()
model.load_state_dict(torch.load('best_model_DGCNN.pt'))

test_dataset = torch.load('individual_scenes\\test_dataset.pt')
test_labels = torch.load('individual_scenes\\test_labelset.pt')

confusion_matrix = torch.zeros(3, 3)
class_map = {0: "floor", 1: "wall", 2: "ceiling"}


with torch.no_grad():  
    model.eval()
    total_per_scene = 0
    correct_per_scene = 0
    
    total_per_class = np.zeros(3)
    correct_per_class = np.zeros(3)
    for i in range(len(test_dataset)):
        
        # divide the data into batches of ~40000
        length = len(test_dataset[i])
        batch = 1
        while True:
            if length//batch < 40000:
                break
            batch += 1
            
        # test the model on each batch
        total = 0
        correct = 0    
        for k in range(batch):
            print(len(test_dataset[i]))
            print("from: ", (k)*len(test_dataset[i])//batch , " to: ", (k+1)*len(test_dataset[i])//batch)
            
            labels = test_labels[i][k*len(test_dataset[i])//batch:(k+1)*len(test_dataset[i])//batch]
            data = test_dataset[i][k*len(test_dataset[i])//batch:(k+1)*len(test_dataset[i])//batch]
            
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            predicted = predicted.numpy()
            
            total += len(labels)
            correct += np.sum(predicted == labels)
            
            for j in range(len(predicted)):
                confusion_matrix[labels[j], predicted[j]] += 1
                if predicted[j] == labels[j]:
                    correct_per_class[labels[j]] += 1
                total_per_class[labels[j]] += 1
            
        total_per_scene += total
        correct_per_scene += correct
            
        print('Validation accuracy for scene {}: {}'.format(i, correct / total))
    val_acc = correct_per_scene / total_per_scene
    print('Validation accuracy: {}'.format(val_acc))

print("Done testing")
    

for i in range(3):
    print("Accuracy of class ", class_map[i], ":", correct_per_class[i]/total_per_class[i])
    
print("most accurate class: ", class_map[np.argmax(correct_per_class/total_per_class)])
print("least accurate class: ", class_map[np.argmin(correct_per_class/total_per_class)])
print(confusion_matrix)

    
