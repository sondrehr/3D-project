import torch
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class segNet(nn.Module):
    def __init__(self):
        super(segNet, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(9, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 3),
        )

    def forward(self, X):
        X = self.fc(X)  
        return X
    
if __name__ == '__main__':
    LR = 1e-3
    WEIGHT_DECAY = 1e-5
    EPOCHS = 1
    
    # Load the dataloaders and change labels
    train_dataloader = torch.load('train_dataloader.pt')
    val_dataloader = torch.load('val_dataloader.pt')
    test_dataloader = torch.load('test_dataloader.pt')
    
    # Create the model
    model = segNet()        
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    writer = SummaryWriter()
    
    ## Train the model
    print('Training the model...')
    
    best_acc = 0
    early_stop = False
    for i in range(EPOCHS):
        print('Epoch: ', str(i))
        for j, (data, labels) in tqdm(enumerate(train_dataloader)):
            model.train()
            data = data.float()
                        
            optimizer.zero_grad()
            output = model(data)

            # one hot encoding
            labels = torch.tensor(labels, dtype=torch.long)
            labels = F.one_hot(labels, num_classes=3)
            labels = labels.float()
            
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            
            if j % 200 == 0:
            
                ## Validate the model
                model.eval()
                total = 0
                correct = 0
                for (data, labels) in tqdm(val_dataloader):
                    data = data.float()
                        
                    output = model(data)
                    _, predicted = torch.max(output.data, 1)
                    
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item() 
                val_acc = correct / total
                print('\nValidation accuracy: {}'.format(val_acc))
                
                # Save the best model
                if val_acc > best_acc + 0.006:
                    best_acc = val_acc
                    index = i * len(train_dataloader) + j
                    torch.save(model.state_dict(), 'best_model.pt')
                    print('Model saved')
                   
                # early stopping 
                if i * len(train_dataloader) + j > index + 1000:
                    early_stop = True
                    print('Early stopping')
                    break
                
                writer.add_scalar('Loss/train', loss, i * len(train_dataloader) + j)
                writer.add_scalar('Accuracy/val', val_acc, i * len(train_dataloader) + j)
                
        if early_stop: break
    
    print('Training completed')
    
    ###########################################################################
    print('Testing the model...')
    
    model.load_state_dict(torch.load('best_model.pt'))
    
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for i, (data, labels) in enumerate(tqdm(test_dataloader)):
            data = data.float()
               
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
                            
        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    print("Done testing")
    