import torch
from tqdm import tqdm
from torch import nn
from torch_geometric import nn as gnn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from utilities import *
    
class DGCNN(nn.Module):
    def __init__(self):
        super(DGCNN, self).__init__()
        
        self.fc9_64 = nn.Linear(9, 64)
        self.fc64_64 = nn.Linear(64, 64)
        self.fc96_1024 = nn.Linear(96, 1024)
        
        self.classifier = nn.Sequential(
            nn.Linear(1216, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )
        
    def forward(self, X):
        # Block 1
        X = knn(X, 20)
        X = self.fc9_64(X)
        X1 = torch.max(X, 1)[0]
        print(X1.shape)
        print("Block 1 done")
        
        # Block 2
        X2 = knn(X1.detach(), 20)
        X2 = self.fc64_64(X2)
        X2 = torch.max(X2, 1)[0]
        print(X2.shape)
        print("Block 2 done")
        
        # Block 3
        X3 = knn(X2.detach(), 20)
        X3 = self.fc64_64(X3)
        X3 = torch.max(X3, 1)[0]
        print(X3.shape)
        print("Block 3 done")

        X = torch.cat((X1, X2, X3), 1)
        X = nn.MaxPool1d(2)(X)
        X = self.fc96_1024(X)
        X = torch.cat((X1, X2, X3, X), 1)
        print(X.shape)
        print("DGCNN done")
        
        X = self.classifier(X)
        return X
    
    
if __name__ == '__main__':
    LR = 1e-3
    EPOCHS = 1
    
    # Load the datasets and labels
    train_dataset = torch.load('individual_scenes\\train_dataset.pt')
    val_dataset = torch.load('individual_scenes\\val_dataset.pt')
    test_dataset = torch.load('individual_scenes\\test_dataset.pt')
    
    train_labels = torch.load('individual_scenes\\train_labelset.pt')
    val_labels = torch.load('individual_scenes\\val_labelset.pt')
    test_labels = torch.load('individual_scenes\\test_labelset.pt')
    
    # Create the model
    model = DGCNN()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    writer = SummaryWriter()
    
    ## Train the model
    print('Training the model...')
    
    best_acc = 0
    early_stop = False
    for i in range(EPOCHS):
        print('Epoch: ', str(i))
        for j, (train_data) in enumerate(tqdm(train_dataset)):
            model.train()
            
            # divide the data into batches of ~40000
            length = len(train_data)
            batch = 1
            while True:
                if length//batch < 40000:
                    break
                batch += 1
            
            # train the model on each batch
            for k in range(batch):
                print("total len: ", len(train_data))
                print("from: ", (k)*len(train_data)//batch , " to: ", (k+1)*len(train_data)//batch)
            
                labels = train_labels[j][k*len(train_data)//batch:(k+1)*len(train_data)//batch]
                data = train_data[k*len(train_data)//batch:(k+1)*len(train_data)//batch]
                
                print(data.shape)
                print(labels.shape)
                optimizer.zero_grad()
                output = model(data)
                
                # one hot encoding
                labels = torch.tensor(labels, dtype=torch.long)
                labels = F.one_hot(labels, num_classes=3)
                labels = labels.float()
                
                loss = loss_fn(output, labels)
                print("loss: ", loss)
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            total_per_scene = 0
            correct_per_scene = 0
            for i in range(len(val_dataset)):
                
                # divide the data into batches of ~40000
                length = len(val_dataset[i])
                batch = 1
                while True:
                    if length//batch < 40000:
                        break
                    batch += 1
                    
                # test the model on each batch
                total = 0
                correct = 0    
                for k in range(batch):
                    print(len(val_dataset[i]))
                    print("from: ", (k)*len(val_dataset[i])//batch , " to: ", (k+1)*len(val_dataset[i])//batch)
                    
                    labels = val_labels[i][k*len(val_dataset[i])//batch:(k+1)*len(val_dataset[i])//batch]
                    data = val_dataset[i][k*len(val_dataset[i])//batch:(k+1)*len(val_dataset[i])//batch]
                    
                    output = model(data)
                    _, predicted = torch.max(output.data, 1)
                    predicted = predicted.numpy()
                    
                    total += len(labels)
                    correct += np.sum(predicted == labels)
                    
                total_per_scene += total
                correct_per_scene += correct
                    
                print('Validation accuracy for scene {}: {}'.format(i, correct / total))
            val_acc = correct_per_scene / total_per_scene
            print('Validation accuracy: {}'.format(val_acc))
            
            writer.add_scalar('Loss/train', loss, i * len(train_dataset) + j)
            writer.add_scalar('Accuracy/val', val_acc, i * len(train_dataset) + j)
                
        if early_stop: break
    
    print('Training completed')
    
    torch.save(model.state_dict(), 'best_model.pt')
    
    ###########################################################################
    print('Testing the model...')
    
    model.load_state_dict(torch.load('best_model.pt'))
    
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for i, (data) in enumerate(tqdm(test_dataset)):
            labels = test_labels[i]
 
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
                            
        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    print("Done testing")
    