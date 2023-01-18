import numpy as np
import torch
import os
from torch.utils.data import random_split, DataLoader, TensorDataset
import glob
from tqdm import tqdm



def parse_arCH_dataset_to_numpy(PATH):
    # open folders
    folders = glob.glob(os.path.join(PATH, "*"))
    for folder in folders:
        print("processing: {}".format(os.path.basename(folder)))
        
        # open files in folder
        files = glob.glob(os.path.join(folder, "*"))
        for file in files:
            print("processing: {}".format(os.path.basename(file)))
            
            # process files
            with open(file, "r") as f:
                lines_processed = []
                lines = f.readlines()
                for line in tqdm(lines):
                    line_processed = np.array(line.split()).astype(np.double)
                    lines_processed.append(line_processed)
                
                # convert to numpy array
                lines_processed = np.array(lines_processed)
                print(lines_processed.shape)
            
            # save as .pt file    
            name = os.path.basename(file).split(".")[0]
            torch.save(lines_processed, folder + "\\" + name + ".pt")



def remove_class_9(PATH):
    # open folders
    folders = glob.glob(os.path.join(PATH, "*"))
    for folder in folders:
        print("processing: {}".format(os.path.basename(folder)))

        # open files in folder    
        files = glob.glob(os.path.join(folder, "*"))
        for file in files:
            
            # remove points with label 9
            if file.split(".")[-1] == "pt":
                print("processing: {}".format(os.path.basename(file)))
                data = torch.load(file)
                
                print("before: ",data.shape)
                data = np.delete(data, np.where(data[:, 6] == 9), axis=0)
                print("after: ",data.shape)
                
                name = os.path.basename(file).split(".")[0]
                torch.save(data, folder + "\\" + name + "_no9.pt")


def reduce_dataset(PATH):
    # open folders
    folders = glob.glob(os.path.join(PATH, "*"))
    for folder in folders:
        print("processing: {}".format(os.path.basename(folder)))

        # open files in folder    
        files = glob.glob(os.path.join(folder, "*"))
        for file in files:
            
            # process files
            if file[-6:] == "no9.pt":
                print("processing: {}".format(os.path.basename(file)))
                data = torch.load(file)
                
                print("Before: {}".format(len(data)))
                
                data = data[np.random.permutation(len(data))]
                data = data[:int(len(data)/10)]
                
                print("After: {}".format(len(data)))
                
                name = os.path.basename(file).split(".")[0]
                torch.save(data, folder + "\\" + name + "_reduced.pt")


def map_labels(PATH, classes):
    # open folders
    folders = glob.glob(os.path.join(PATH, "*"))
    for folder in folders:
        print("processing: {}".format(os.path.basename(folder)))

        # open files in folder    
        files = glob.glob(os.path.join(folder, "*"))
        for file in files:
            
            # process files
            if file[-10:] == "reduced.pt":
                print("processing: {}".format(os.path.basename(file)))
                data = torch.load(file)
                
                print("Before: ")
                print("0: ", len(np.where(data[:, 6] == 0)[0]), " - ", len(np.where(data[:, 5] == 0)[0]))
                print("1: ", len(np.where(data[:, 6] == 1)[0]), " - ", len(np.where(data[:, 5] == 1)[0]))
                print("2: ", len(np.where(data[:, 6] == 2)[0]), " - ", len(np.where(data[:, 5] == 2)[0]))
                print("3: ", len(np.where(data[:, 6] == 3)[0]), " - ", len(np.where(data[:, 5] == 3)[0]))
                print("4: ", len(np.where(data[:, 6] == 4)[0]), " - ", len(np.where(data[:, 5] == 4)[0]))
                print("5: ", len(np.where(data[:, 6] == 5)[0]), " - ", len(np.where(data[:, 5] == 5)[0]))
                print("6: ", len(np.where(data[:, 6] == 6)[0]), " - ", len(np.where(data[:, 5] == 6)[0]))
                print("7: ", len(np.where(data[:, 6] == 7)[0]), " - ", len(np.where(data[:, 5] == 7)[0]))
                print("8: ", len(np.where(data[:, 6] == 8)[0]), " - ", len(np.where(data[:, 5] == 8)[0]))
                
                # change all values in column 7 to some other value
                # Create a boolean mask to select the elements in the 6th column that need to be changed
                mask = np.logical_or.reduce((data[:,6] == 3, data[:,6] == 6,
                                            data[:,6] == 1, data[:,6] == 2,
                                            data[:,6] == 4, data[:,6] == 5,
                                            data[:,6] == 0, data[:,6] == 7,
                                            data[:,6] == 8))

                # Use the boolean mask to change the values
                data[mask,6] = np.where(np.isin(data[mask,6], [3,6]), 0, np.where(np.isin(data[mask,6], [1,2,4,5]), 1, 2))
                
                print("After: ") 
                print("0: ", len(np.where(data[:, 6] == 0)[0]), " - ", len(np.where(data[:, 5] == 0)[0]))
                print("1: ", len(np.where(data[:, 6] == 1)[0]), " - ", len(np.where(data[:, 5] == 1)[0]))
                print("2: ", len(np.where(data[:, 6] == 2)[0]), " - ", len(np.where(data[:, 5] == 2)[0]))
                print("3: ", len(np.where(data[:, 6] == 3)[0]), " - ", len(np.where(data[:, 5] == 3)[0]))
                print("4: ", len(np.where(data[:, 6] == 4)[0]), " - ", len(np.where(data[:, 5] == 4)[0]))
                print("5: ", len(np.where(data[:, 6] == 5)[0]), " - ", len(np.where(data[:, 5] == 5)[0]))
                print("6: ", len(np.where(data[:, 6] == 6)[0]), " - ", len(np.where(data[:, 5] == 6)[0]))
                print("7: ", len(np.where(data[:, 6] == 7)[0]), " - ", len(np.where(data[:, 5] == 7)[0]))
                print("8: ", len(np.where(data[:, 6] == 8)[0]), " - ", len(np.where(data[:, 5] == 8)[0]))
                
                name = os.path.basename(file).split(".")[0]
                torch.save(data, folder + "\\" + name + "_mapped" + str(classes) + ".pt")





def normalize_dataset(PATH):
    # open folders
    folders = glob.glob(os.path.join(PATH, "*"))
    for folder in folders:
        print("processing: {}".format(os.path.basename(folder)))

        # open files in folder    
        files = glob.glob(os.path.join(folder, "*"))
        for file in files:
            # process files
            if file[-10:] == "mapped3.pt":
                print("processing: {}".format(os.path.basename(file)))
                data = torch.load(file)
                     
                # print("max_x: ", np.max(data[:, 0]))
                # print("max_y: ", np.max(data[:, 1]))
                # print("max_z: ", np.max(data[:, 2]))
                # print("min_x: ", np.min(data[:, 0]))
                # print("min_y: ", np.min(data[:, 1]))
                # print("min_z: ", np.min(data[:, 2]))
                
                # move points to positive octant
                data[:, 0] -= np.min(data[:, 0])
                data[:, 1] -= np.min(data[:, 1])
                data[:, 2] -= np.min(data[:, 2])
                
                # print("max_x: ", np.max(data[:, 0]))
                # print("max_y: ", np.max(data[:, 1]))
                # print("max_z: ", np.max(data[:, 2]))
                # print("min_x: ", np.min(data[:, 0]))
                # print("min_y: ", np.min(data[:, 1]))
                # print("min_z: ", np.min(data[:, 2]))
                
                # # normalize points to unit cube
                # max_x = np.max(data[:, 0])
                # max_y = np.max(data[:, 1])
                # max_z = np.max(data[:, 2])
                # max_all = np.max([max_x, max_y, max_z])
                
                # # print("before: ")
                # # print("max_x: ", np.max(data[:, 0]))
                # # print("max_y: ", np.max(data[:, 1]))
                # # print("max_z: ", np.max(data[:, 2]))
                
                # data[:, 0] = data[:, 0] / max_all
                # data[:, 1] = data[:, 1] / max_all
                # data[:, 2] = data[:, 2] / max_all
                
                # print("after: ")
                # print("max_x: ", np.max(data[:, 0]))
                # print("max_y: ", np.max(data[:, 1]))
                # print("max_z: ", np.max(data[:, 2]))
                
                name = os.path.basename(file).split(".")[0]
                torch.save(data, folder + "\\" + name + "_norm.pt")
                
                


def data_to_dataloader(PATH):
    val_ratio = 0.1

    train_dataset = []
    val_dataset = []
    test_dataset = []
    
    train_labelset = []
    val_labelset = []
    test_labelset = []
    
    # open folders
    folders = glob.glob(os.path.join(PATH, "*"))
    for folder in folders:
        print("processing: {}".format(os.path.basename(folder)))

        # open files in folder    
        files = glob.glob(os.path.join(folder, "*"))
        for file in files:
            
            # process files
            if file[-7:] == "norm.pt":
                print("processing: {}".format(os.path.basename(file)))
                data = torch.load(file)
                
                if folder.split("\\")[-1] == "Test":
                    test_label = data[:, 6]
                    test_data = np.delete(data, 6, axis=1)
                    
                    test_dataset.append(test_data)
                    test_labelset.append(test_label)
                    
                if folder.split("\\")[-1] == "Training":                    
                    val_size = int(len(data) * val_ratio)
                    train_size = len(data) - val_size
                    train, val = random_split(data, [train_size, val_size])
                    train = np.array(train)
                    val = np.array(val)

                    train_label = train[:, 6]
                    train_data = np.delete(train, 6, axis=1)
                    
                    train_dataset.append(train_data)
                    train_labelset.append(train_label)
                    
                    val_label = val[:, 6]
                    val_data = np.delete(val, 6, axis=1)
                    
                    val_dataset.append(val_data)
                    val_labelset.append(val_label)    
                          
    ## Concatenate all the data together
    train_dataset = np.concatenate(train_dataset)
    val_dataset = np.concatenate(val_dataset)
    test_dataset = np.concatenate(test_dataset)
    
    train_labelset = np.concatenate(train_labelset)
    val_labelset = np.concatenate(val_labelset)    
    test_labelset = np.concatenate(test_labelset)
    
    ## Shuffle scenes so the model doesn't end up learning one scene perticularly good
    train_permute = np.random.permutation(len(train_dataset))
    val_permute = np.random.permutation(len(val_dataset))
    test_permute = np.random.permutation(len(test_dataset))
    
    train_dataset = train_dataset[train_permute]
    val_dataset = val_dataset[val_permute]
    test_dataset = test_dataset[test_permute]
    
    train_labelset = train_labelset[train_permute]
    val_labelset = val_labelset[val_permute]
    test_labelset = test_labelset[test_permute]
    
    ## Convert to tensors
    train_data = TensorDataset(torch.from_numpy(train_dataset), torch.from_numpy(train_labelset))
    val_data = TensorDataset(torch.from_numpy(val_dataset), torch.from_numpy(val_labelset))
    test_data = TensorDataset(torch.from_numpy(test_dataset), torch.from_numpy(test_labelset))
    
    ## Create dataloaders
    train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=128)
    test_dataloader = DataLoader(test_data, batch_size=128)

    ## Save dataloaders
    torch.save(train_dataloader, "train_dataloader.pt")
    torch.save(val_dataloader, "val_dataloader.pt")
    torch.save(test_dataloader, "test_dataloader.pt")

                    
                            
    
if __name__ == '__main__':
    PATH = "C:\\Users\\sondr\\OneDrive\\Desktop\\dataset_3D"
    CLASSES = 3
    
    #! 1. Necessary functions to run no matter config
    # parse_arCH_dataset_to_numpy(PATH)
    # remove_class_9(PATH)
    # reduce_dataset(PATH)
    
    #! 2. Change CLASSES to change number of classes
    # map_labels(PATH, CLASSES)
    
    #! 3. Add to increase test accuracy
    normalize_dataset(PATH)
    
    #! 4. Prepare data for training
    data_to_dataloader(PATH)
    

    
    
    
    
    

   
