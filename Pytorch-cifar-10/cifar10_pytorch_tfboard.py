import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

from __future__ import print_function
from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
# from tensorboardX import SummaryWriter  # for pytorch below 1.14
from torch.utils.tensorboard import SummaryWriter # for pytorch above or equal 1.14

class Net(nn.Module):
    def __init__(self, has_batch = False, more_linear = False, has_dropout = False):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.fc1_5 = nn.Linear(512, 512)
        self.batch_norm = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(p=0.3)
        self.has_batch = has_batch
        self.more_linear = more_linear
        self.has_dropout = has_dropout

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        if self.has_dropout:
          x = self.dropout(x)
        if self.has_batch:
            x = self.batch_norm(x)
        if self.more_linear:
            x = F.relu(self.fc1_5(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def eval_net(dataloader):
    correct = 0
    total = 0
    total_loss = 0
    net.eval() # Why would I do this?
    criterion = nn.CrossEntropyLoss(reduction='mean')
    for data in dataloader:
        images, labels = data
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
        loss = criterion(outputs, labels)
        total_loss += loss.item()
    net.train() # Why would I do this?
    return total_loss / total, correct.float() / total

def plot_history(train_loss_set, validate_loss_set, 
                 train_acc_set, validate_acc_set, file_name = ""):
  fig, ax = plt.subplots(2, 1, figsize = (10, 10))
  ax[0].plot(train_loss_set, color = "b", label = "train_loss")
  ax[0].plot(validate_loss_set, color = "r", label = "val_loss")
  ax[0].set_title("Loss History")
  ax[0].set_xlabel("Epochs")
  ax[0].set_ylabel("Loss")
  ax[0].legend()
  plt.savefig(file_name+"loss_histoty.png", dpi=100)
  ax[1].plot(train_acc_set, color = "b", label = "train_acc")
  ax[1].plot(validate_acc_set, color = "r", label = "val_acc")
  ax[1].set_title("Accuracy History")
  ax[1].set_xlabel("Epochs")
  ax[1].set_ylabel("Accuracy")
  ax[1].legend()
  plt.savefig(file_name+"acc_histoty.png", dpi=100)
  plt.show()

def build_model(net, trainloader, testloader, optimizer_candidate = 1, path_name = ""):
    train_loss_set, validate_loss_set, train_acc_set, validate_acc_set = [], [], [], []
    writer = SummaryWriter(log_dir='./log/{}'.format(path_name))
    criterion = nn.CrossEntropyLoss()
    if optimizer_candidate == 1:
      optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    if optimizer_candidate == 2:
      optimizer = optim.RMSprop(net.parameters(), lr=0.001, momentum=0.9)
    if optimizer_candidate == 3:
      optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0)

    print('Start training...')
    for epoch in range(MAX_EPOCH):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            # wrap them in Variable
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 2000 mini-batches
                print('    Step: %5d avg_batch_loss: %.5f' %
                      (i + 1, running_loss / 500))
                running_loss = 0.0
        print('    Finish training this EPOCH, start evaluating...')
        train_loss, train_acc = eval_net(trainloader)
        test_loss, test_acc = eval_net(testloader)
        print('EPOCH: %d train_loss: %.5f train_acc: %.5f test_loss: %.5f test_acc %.5f' %
              (epoch+1, train_loss, train_acc, test_loss, test_acc))
        
        # loss and accuracy log
        train_loss_set.append(train_loss)
        validate_loss_set.append(test_loss)
        train_acc_set.append(train_acc)
        validate_acc_set.append(test_acc)

        writer.add_scalar('loss/train', train_loss, epoch+1)
        writer.add_scalar('loss/test', test_loss, epoch+1)
        writer.add_scalar('acc/train', train_acc, epoch+1)
        writer.add_scalar('acc/test', test_acc, epoch+1)

    writer.close()
    print('Finished Training')
    print('Saving model...')
    torch.save(net.state_dict(), '{}.pth'.format(path_name))
    plot_history(train_loss_set, validate_loss_set, train_acc_set, validate_acc_set, file_name = path_name)

if __name__ == "__main__":
    BATCH_SIZE = 32 #mini_batch size
    MAX_EPOCH = 10 #maximum epoch to train

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #torchvision.transforms.Normalize(mean, std)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    """
    vanilla model
    """
    print('Building vanilla model...')
    torch.cuda.empty_cache()
    net = Net(has_batch = False, more_linear = False, has_dropout = False).cuda()
    net.train() # Why would I do this?
    build_model(net, trainloader, testloader, optimizer_candidate = 1, path_name = "no_batch")

    """
    vanilla model with batch-normalization
    """
    print('Building vanilla model with batch-normalization...')
    torch.cuda.empty_cache()
    net = Net(has_batch = True, more_linear = False, has_dropout = False).cuda()
    net.train()
    build_model(net, trainloader, testloader, optimizer_candidate = 1, path_name = "with_batch")

    """
    model with additional linear layer
    """
    print('Building vanilla model with additional linear layer...')
    torch.cuda.empty_cache()
    net = Net(has_batch = False, more_linear = True, has_dropout = False).cuda()
    net.train()
    build_model(net, trainloader, testloader, optimizer_candidate = 1, path_name = "additional_linear")

    """
    model with pretrained weight
    """
    print('Building vanilla model with additional linear layer by using pre-trained weight...')
    torch.cuda.empty_cache()
    net = Net(has_batch = False, more_linear = True, has_dropout = False).cuda()
    net.train()
    pre_trained_dict = torch.load("additional_linear.pth")
    current_trained_dict = net.state_dict()
    for key in current_trained_dict:
        if key in pre_trained_dict and key != 'fc2.weight' and key != 'fc2.bias':
            current_trained_dict[key] = pre_trained_dict[key]
    net.load_state_dict(current_trained_dict)
    build_model(net, trainloader, testloader, optimizer_candidate = 1, path_name = "with_pretrained")
    """
    vanilla model with adaptive schedule
    """
    # print('Building vanilla model with RMSprop...')
    # torch.cuda.empty_cache()
    # net = Net(has_batch = False, more_linear = False).cuda()
    # net.train()
    # build_model(net, trainloader, testloader, optimizer_candidate = 2, path_name = "rms_pro")

    print('Building vanilla model with Adam...')
    torch.cuda.empty_cache()
    net = Net(has_batch = False, more_linear = False, has_dropout = False).cuda()
    net.train()
    build_model(net, trainloader, testloader, optimizer_candidate = 3, path_name = "adam")

    """
    vanilla model with dropout
    """
    print('Building vanilla model with adam and dropout...')
    torch.cuda.empty_cache()
    net = Net(has_batch = False, more_linear = False, has_dropout = True).cuda()
    net.train()
    build_model(net, trainloader, testloader, optimizer_candidate = 3, path_name = "with_dropout")
