import torch
from tqdm import tqdm

def train(model, dataloader, optimizer, criterion1, train_data, device):
    print("Training")
    model.train()
    counter = 0
    train_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        counter += 1

        image, label1, label2 = data["image"].to(device), data["label1"].to(device), data["label2"].to(device)
        optimizer.zero_grad()
        outputs = model(image)


        label1_hat = outputs["label1"]
        label2_hat = outputs["label2"]

        loss1 = criterion1(label1_hat, label1.squeeze().long())
        loss2 = criterion1(label2_hat, label2.squeeze().long())

        loss = loss1+loss2

        loss.backward()
        train_loss = train_loss + ((1 / (i + 1)) * (loss.data - train_loss))
        optimizer.step()

    return train_loss

def validate(model, dataloader, criterion1, val_data, device):
    print("Validating")
    model.eval()
    counter = 0
    val_running_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            counter += 1

            image, label1, label2 = data["image"].to(device), data["label1"].to(device), data["label2"].to(device)
            outputs = model(image)

            label1_hat = outputs["label1"]
            label2_hat = outputs["label2"]

            loss1 = criterion1(label1_hat, label1.squeeze().long())
            loss2 = criterion1(label2_hat, label2.squeeze().long())

            loss = loss1+loss2

            val_running_loss = val_running_loss + ((1 / (i + 1)) * (loss.data - val_running_loss))

    return val_running_loss