import torch
import torch.nn as nn
import numpy as np
from fc9 import segNet
from tqdm import tqdm
#import cv2
import matplotlib.pyplot as plt

#testing the model 
test_dataloader = torch.load('test_dataloader9.pt')
model = segNet()
model.load_state_dict(torch.load('best_model.pt'))
if torch.cuda.is_available():
    model = model.cuda()

#class_map = torch.load('class_map.pt')
confusion_matrix = np.zeros((9, 9))

# testing the model 
model.eval()  # set the model to evaluation mode
total_per_class = np.zeros(9)
correct_per_class = np.zeros(9)

# Initialize the confusion matrix
confusion_matrix = np.zeros((9, 9))

# Iterate over the test data
for i, (data, labels) in enumerate(tqdm(test_dataloader)):
    data = data.float()
    if torch.cuda.is_available():
        data = data.cuda()
        labels = labels.cuda()
               
    output = model(data)

    # Get the predictions
    _, predictions = torch.max(output, 1)

    predictions = predictions.cpu().numpy().astype(int)
    labels = labels.cpu().numpy().astype(int)

    # Update the confusion matrix
    for i in range(len(predictions)):
        confusion_matrix[labels[i], predictions[i]] += 1
        if predictions[i] == labels[i]:
            correct_per_class[labels[i]] += 1
        total_per_class[labels[i]] += 1
print("how many of each class: ", total_per_class)
for i in range(9):
    print("Class", i, ":", correct_per_class[i], "/", total_per_class[i])
print("Accuracy: ", np.sum(correct_per_class)/np.sum(total_per_class))
print("Done testing model")   

#make a list with these values: arch":0, "column":1, "moldings":2, "floor":3, "door_window":4, "wall":5, "stairs":6, "vault":7, "roof":8

classlist = ["arch", "column", "moldings", "floor", "door_window", "wall", "stairs", "vault", "roof"]

np.set_printoptions(precision=3)
for i in range(9):
    print("Accuracy of class", classlist[i], ":", correct_per_class[i]/total_per_class[i])    
    print("most accurate class: ", classlist[np.argmax(correct_per_class/total_per_class)])
    print("least accurate class: ", classlist[np.argmin(correct_per_class/total_per_class)])


x = np.arange(9)
# Plot the confusion matrix
#normalize the confusion matrix
confusion_matrix = confusion_matrix/np.sum(confusion_matrix, axis=1)
print("Plotting confusion matrix...")
print(confusion_matrix)
plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.xticks(x, classlist, rotation=45)
plt.yticks(x, classlist)
plt.show()



