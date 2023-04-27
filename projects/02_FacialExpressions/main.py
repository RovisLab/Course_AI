import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from torch import optim
from tensorflow.keras.preprocessing.image import ImageDataGenerator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 1
learning_rate = 0.001
num_epochs = 10

# Write transform for image
data_transform = transforms.Compose([
    # Resize the images to 64x64
    transforms.Resize(size=(140, 160)),
    # Turn the image into a torch.Tensor
    transforms.ToTensor(),

])
# train_data
train_data = datasets.ImageFolder(root="E:\work\programming\AI robo\Emotion recognition\data\\training", # target folder of images
                                  transform=data_transform, # transforms to perform on data (images)
                                  target_transform=None) # transforms to perform on labels (if necessary)

test_data = datasets.ImageFolder(root="E:\work\programming\AI robo\Emotion recognition\data\\testing",
                                 transform=data_transform)

#print(f"Train data:\n{train_data}\nTest data:\n{test_data}")


dataloader_train = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
dataloader_val = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)



class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        #self.conv2 = nn.Conv2d(32, 64, 3, 1)
        #self.dropout1 = nn.Dropout(0.25)
        #self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(697728, 3)
        #self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        # x = self.conv2(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        # x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# figure = plt.figure(figsize=(10, 8))
# cols, rows = 5, 5
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(train_data), size=(1,)).item()
#     img, label = train_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(label)
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()

model = LeNet()
model = model.to(device)

def predict():
    conv_mat = np.array((0,0,0,0,0,0,0,0,0)).reshape(3,3)
    f, axarr = plt.subplots(3, 5)
    m = 0
    n = 0

    for idx in range(5):
        sample = next(iter(dataloader_val))
        imgs, labels = sample
        imgs, labels = imgs.to(device), labels.to(device)
        pred = model(imgs)
        pred = pred.argmax(dim=1, keepdim=True)


        for i in range(batch_size):
            print("Target: ", labels[i], "; Prediction = ", pred[i])

            #convolution matrix creation
            conv_mat[labels[i].item(),pred[i].item()] += 1

            #testing image grid
            img_transform = imgs[i].permute(1,2,0)
            axarr[m, n].title.set_text("Target: \{}; Prediction : \{}".format(labels[i].item(),pred[i].item()))
            axarr[m, n].title.set_size(10)
            axarr[m, n].imshow(img_transform.cpu().squeeze())
            n += 1
            if n == 5:
                m += 1
                n = 0

    plt.show()

    print(conv_mat)

loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
model.train()
for epoch in range(num_epochs):
    train_loss = 0
    for i, (images, labels) in enumerate(dataloader_train):
        images, labels = images.to(device), labels.to(device)
        

        outputs = model(images)
        loss = loss_func(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("loss = ", loss.item())
        train_loss += loss.item()

        # f, axarr = plt.subplots(1, 3)
        # for j in range(batch_size):
        #
        #     # testing image grid
        #     img_transform = images[j].permute(1, 2, 0)
        #
        #     match labels[j]:
        #         case 0:
        #             expression = "angry"
        #         case 1:
        #             expression = "happy"
        #         case 2:
        #             expression = "neutral"
        #
        #
        #     axarr[j].title.set_text("Expression: \{}".format(expression))
        #     axarr[j].title.set_size(10)
        #     axarr[j].imshow(img_transform.cpu().squeeze())
        #
        # plt.show()
        print("Labels = ",labels)

        # if i % 10 == 0:
        #     print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'.format(epoch, num_epochs, i, len(dataloader_train), train_loss))
        #     train_loss = 0



torch.save(model.state_dict(), 'LeNet.pt')

state_dict = torch.load('LeNet.pt')
model.load_state_dict(state_dict)
model.eval()
#predict()

