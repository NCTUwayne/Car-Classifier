# Car-Classifier
NCTU VRDL HW1
## Hardware
The following specs were used to create the original solution.
* Ubuntu 16.04 LTS
* Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz
* NVIDIA TitanX
## Dataset
Download extract training data, training labels and testing data.
    kaggle competitions download -c cs-t0828-2020-hw1
After downloading the datas, the data directory shoud be structured as:
    data
      +- train file
      +- test file
      +- trainlabels.csv
## Requirements
* numpy		1.17.0
* pandas		0.25.0
* pillow		6.1.0
* matplotlib	3.1.1
* natsort 		7.0.1
* sklearn		0.0
* torch		1.7.0
* torchvision 	0.8.1
## Dataloader
Uncomment the following lines if using the validation set.
```
# trainset, validset = torch.utils.data.random_split(dataset, [10000, 1185])

# trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=0)

# validloader = torch.utils.data.DataLoader(validset, batch_size=16, shuffle=True, num_workers=0)
```
## Training model function
Uncomment the following lines if using the validation set.
```
# model.eval()
# valid_acc = eval_model(model)
# valid_accuracies.append(valid_acc)
        
# model.train()
# scheduler.step(valid_acc)
```
## Training model
Run the cells below the "Training" markdown. 
Models tested:
* resnet34 (pretrained=True)
* resnet50 (pretrained=True)
* resnet101 (pretrained=True)
* antialiased_cnns.resnet50 (pretrained=True)
## Predict test set
Run the cell below the "Predict test set" markdown, the "res101.csv" will be in the code directory.
