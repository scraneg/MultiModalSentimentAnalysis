import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
from model import MultiModalModel
from tqdm import tqdm
# for not error when load dataloader
from process import MultiModalDataset, collate_fn


def train_one_epoch(epoch, model, train_dataloader, val_dataloader, criterion, optimizer, device):
    # train
    model.train()
    print('start training for epoch {}'.format(epoch))
    for batch_index, (imgs, texts, atten_masks, token_type_ids, labels) in tqdm(enumerate(train_dataloader)):
        imgs = imgs.to(device)
        texts = texts.to(device)
        atten_masks = atten_masks.to(device)
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(imgs, texts, atten_masks, token_type_ids)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        if batch_index % 10 == 0:
            print('epoch {}, batch_index {}, loss {}'.format(
                epoch, batch_index, loss.item()))

    # val
    model.eval()
    print('start val for epoch {}'.format(epoch))
    correct = 0
    with torch.no_grad():
        for batch_index, (imgs, texts, atten_masks, token_type_ids, labels) in tqdm(enumerate(val_dataloader)):
            imgs = imgs.to(device)
            texts = texts.to(device)
            atten_masks = atten_masks.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)
            output = model(imgs, texts, atten_masks, token_type_ids)
            pred = torch.argmax(output, dim=1)
            correct += torch.sum(pred == labels).item()
        
        acc = correct / len(val_dataloader.dataset)
        print('epoch {}, val acc {}'.format(epoch, acc))

    return acc



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--epochs', type=int, default=10)  # epochs for training
    args.add_argument('--lr', type=float, default=1e-4)  # learning rate

    args = args.parse_args()

    # load dataloader (saved in ./dataset/dataloader)
    train_dataloader = torch.load('./dataset/dataloader/train_dataloader.pth')
    val_dataloader = torch.load('./dataset/dataloader/val_dataloader.pth')

    # load model
    model = MultiModalModel(n_classes=3)
    # train
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(device))
    model.to(device)

    acc_list = []
    for epoch in range(1, args.epochs + 1):
        acc = train_one_epoch(epoch, model, train_dataloader, val_dataloader, criterion, optimizer, device)
        torch.save(model.state_dict(), './models/model_{}.pth'.format(epoch))
        if len(acc_list) == 0 or acc > max(acc_list):
            torch.save(model.state_dict(), './models/model_best.pth')
        acc_list.append(acc)

    print('best acc: {}'.format(max(acc_list)))

    
        
    plt.plot(range(1, args.epochs + 1), acc_list)
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.savefig('./acc.png')
    plt.show()



