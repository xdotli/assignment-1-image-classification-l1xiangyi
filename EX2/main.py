import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from torchvision.models import vgg16

import os

# This is a quite simple CNN with 3 convolutional layers
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # the first layer of the CNN
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        # the second layer of the CNN
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 7, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            # GRADED FUNCTION: Please define the third layer of the CNN. 
            # Conv2D with 128 5x5 filters and stride of 2
            # ReLU
            # MaxPool2d with 2x2 filters and stride of 2
            ### START SOLUTION HERE ###




            ### END SOLUTION HERE ###
        )
        self.classifier = nn.Sequential(
            # GRADED FUNCTION: Please define the classifier
            # Linear with input size of 128 x width? x height? and output size of 4096
            # ReLU
            # Linear with input size of 4096 and output size of 4096
            # ReLU
            # Linear with input size of 4096 and output size of number of classes
            ### START SOLUTION HERE ###





            ### END SOLUTION HERE ###
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # GRADED FUNCTION: Flatten Layer
        ### START SOLUTION HERE ###

        ### END SOLUTION HERE ###
        x = self.classifier(x)
        return x

def train(train_loader, model, loss_fn, optimizer, device):
    for i, (image, annotation) in enumerate(train_loader):
        # move data to the same device as model
        image = image.to(device)
        annotation = annotation.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward and compute prediction error
        output = model(image)
        loss = loss_fn(output, annotation)
        # backward + optimize
        loss.backward()
        optimizer.step()

        # print statistics
        if i % 20 == 0:    # print every 20 iterates
            print(f'iterate {i + 1}: loss={loss:>7f}')

def val(val_loader, model, device):
    # switch to evaluate mode
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (image, annotation) in enumerate(val_loader):
            # move data to the same device as model
            image = image.to(device)
            annotation = annotation.to(device)

            # network forward
            output = model(image)

            # for compute accuracy
            _, predicted = torch.max(output.data, 1)
            total += annotation.size(0)
            correct += (predicted == annotation).sum().item()

    # GRADED FUNCTION: calculate the accuracy using variables before
    # use variable named 'acc' to store the accuracy
    ### START SOLUTION HERE ###

    ### END SOLUTION HERE ###
    print(f'total val accuracy: {100 * acc:>2f} %')
    return acc


if __name__ == '__main__':
    # define image transform
    transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                    ])
    batch_size = 64

    # loda data
    traindir = os.path.join('flower_dataset', 'train')
    valdir = os.path.join('flower_dataset', 'val')    
    # GRADED FUNCTION: define train_loader and val_loader
    ### START SOLUTION HERE ###





    ### END SOLUTION HERE ###

    # device used to train
    device = torch.device("cuda:0")
    # GRADED FUNCTION: define a SimpleCNN model and move it to the device
    # use variable named 'model' to store this model
    ### START SOLUTION HERE ###


    ### END SOLUTION HERE ###

    # Classification Cross-Entropy loss 
    loss_fn = nn.CrossEntropyLoss()

    # GRADED FUNCTION: Please define the optimizer as SGD with lr=0.05, momentum=0.9, weight_decay=0.0001
    ### START SOLUTION HERE ###

    ### END SOLUTION HERE ###

    # GRADED FUNCTION: Please define the scheduler
    # the learning rate will decay 0.05 every 5 steps
    ### START SOLUTION HERE ###

    ### END SOLUTION HERE ###

    # create model save path
    os.makedirs('work_dir', exist_ok=True)

    max_acc = -float('inf')
    for epoch in range(10):
        print('-' * 30, 'epoch', epoch + 1, '-' * 30)

        # train
        train(train_loader, model, loss_fn, optimizer, device)
        print('lr: {}'.format(optimizer.param_groups[0]['lr']))

        # validation
        acc = val(val_loader, model, device)

        # save best model
        if acc > max_acc:
            pt_path = os.path.join('work_dir', 'best.pt')
            torch.save(model.state_dict(), pt_path)
            print('save model')

        # decay learning rate
        scheduler.step()
    print('Finished Training')
