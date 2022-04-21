from __future__ import print_function, division
import os, sys
from os.path import dirname
project_dir = dirname(dirname(dirname(os.path.abspath(__file__))))

from torch.optim import lr_scheduler
import time
import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# prepare
pretrained_size = 224  # at least 224 ,512
pretrained_means = [0.485, 0.456, 0.406]
pretrained_stds = [0.229, 0.224, 0.225]
pan_transform = transforms.Compose([
                                    transforms.Resize((pretrained_size, pretrained_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=pretrained_means, std=pretrained_stds)
                                    ])
def train():

    data_dir = r'workplace\classification_for_vision\image_data'
    pan_dataset = datasets.ImageFolder(data_dir, transform=pan_transform)

    train_set, test_set = torch.utils.data.random_split(pan_dataset, [1500, 100])

    pan_dataloader_train = torch.utils.data.DataLoader(train_set, batch_size=10, shuffle=True)
    pan_dataloader_test = torch.utils.data.DataLoader(test_set, batch_size=10, shuffle=True)

    dataiter = next(iter(pan_dataloader_test))
    images, labels = dataiter
    # plt.imshow(np.transpose(images[1].numpy(), (1, 2, 0)))
    #plt.imshow(images[3].permute(1, 2, 0))

    # create classification
    #model = models.resnet50(pretrained=True)  # 500 sec
    #model = models.vgg16(pretrained=True)  # 900
    #model = models.resnet18(pretrained=True)  # 200 sec (good decrease)
    model = models.vgg11(pretrained=True)    # 300 sec
    # IN_FEATURES = model.classifier[-1].in_features
    # final_fc = nn.Linear(IN_FEATURES, 4)
    # model.classifier[-1] = final_fc

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)

    for param in model.parameters():
        param.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    for param in model.parameters():
        print(param.requires_grad)

    for parameter in model.classifier[-1].parameters():
        parameter.requires_grad = True


    criterion = nn.CrossEntropyLoss()
    loss_fn = criterion
    #optimizer_ft = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    optimizer_ft = optim.SGD(model.classifier[-1].parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    exp_lr_scheduler.step()
    model.to(device)


    def make_train_step(model, loss_fn, optimizer):
        def train_step(x, y):
            model.train()
            yhat = model(x)
            loss = loss_fn(yhat, y)
            loss.backward()
            optimizer.step()

            optimizer.zero_grad()
            return loss.item()
        return train_step


    train_step = make_train_step(model, loss_fn, optimizer_ft)
    n_epochs = 1000
    training_losses = []
    validation_losses = []
    min_valid_loss = np.inf

    for epoch in range(n_epochs):
        start = time.time()
        batch_losses = []
        for x_batch, y_batch in pan_dataloader_train:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            loss = train_step(x_batch, y_batch)
            batch_losses.append(loss)
        training_loss = np.mean(batch_losses)
        training_losses.append(training_loss)

        with torch.no_grad():
            val_losses = []
            for x_val, y_val in pan_dataloader_test:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                model.eval()
                yhat = model(x_val)
                val_loss = loss_fn(yhat, y_val).item()
                val_losses.append(val_loss)
            validation_loss = np.mean(val_losses)
            validation_losses.append(validation_loss)

        if min_valid_loss > validation_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.3f}--->{validation_loss:.3f}) \t Saving The Model')
            min_valid_loss = validation_loss

            # Saving State Dict
            torch.save(model.state_dict(), r'workplace\classification_for_vision\saved_model_vgg11.pth')

        print(f"[{epoch+1}] Training loss: {training_loss:.3f}\t Validation loss: {validation_loss:.3f}")
        print(time.time() - start)



def predict_rotation(pil_image):
    img = pil_image
    PATH = project_dir + r'/workplace/classification_for_vision/saved_model_resnet50.pth'
    #PATH = project_dir + r'\workplace\classification_for_vision\saved_model_resnet18.pth'
    #PATH = project_dir + r'\workplace\classification_for_vision\saved_model_vgg11.pth'
    #print(project_dir+r'\workplace\classification_for_vision\saved_model_test_3_vgg11.pth')
    model = models.resnet50(num_classes=4)
    #model = models.vgg11(num_classes=4)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    # img = img.rotate(90)
    # img = img.rotate(90)
    # img = img.rotate(90)
    # img = img.rotate(90)

    test_image = pan_transform(img).unsqueeze(0)
    #img.show()

    test = model(test_image)
    #print(test)
    predicted_class_no = test.argmax().item()


    if predicted_class_no == 1: img = img.rotate(-90)
    if predicted_class_no == 0: img = img.rotate(-180)
    if predicted_class_no == 2: img = img.rotate(-270)
    if predicted_class_no == 3: img = img.rotate(0)


    return img
'''
dataiter = next(iter(pan_dataloader_test))
images, labels = dataiter
test = model(images)
res = []
for x in test:
    res.append(x.argmax().item())
for x, y in zip(res, labels.tolist()):
    print(x-y)


dataiter = next(iter(pan_dataloader_test))
images, labels = dataiter

test = model(test_image)


plt.imshow(np.transpose(images[2].numpy(), (1, 2, 0)))
print(labels[2])
#plt.imshow(images[3].permute(1, 2, 0))


rotated 180  class_no = 0
rotated left class_no = 1
rotated right classi_no =2
standart class_no = 3
'''





