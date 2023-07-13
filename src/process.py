# data processing

'''
dataset folder: ./dataset

dataset structure:
    dataset
        |--- train.txt (guid,tag)
        |--- test_without_label.txt (guid)
        |--- data
                |--- 1.jpg
                |--- 1.txt
                |--- 2.jpg
                |--- 2.txt
                |--- ...
'''


import os
import pandas as pd
import torch
import argparse
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import BertTokenizer
import chardet

# dataset


class MultiModalDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        guid = self.df.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, str(guid) + '.jpg')
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        text_path = os.path.join(self.img_dir, str(guid) + '.txt')
        with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read().strip()
        label = self.df.iloc[idx, 1]
        return img, text, label


def collate_fn(batch):
    imgs, texts, labels = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    # use BertTokenizer to tokenize the text
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    texts, atten_masks, token_type_ids = tokenizer(
        list(texts), padding=True, truncation=True, return_tensors='pt').values()
    atten_masks = atten_masks
    token_type_ids = token_type_ids
    tmp_labels = []
    for label in labels:
        if label == 'positive':
            tmp_labels.append(0)
        elif label == 'neutral':
            tmp_labels.append(1)
        else:
            tmp_labels.append(2)

    labels = torch.tensor(tmp_labels)
    return imgs, texts, atten_masks, token_type_ids, labels


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # batch size for dataloader
    args.add_argument('--batch_size', type=int, default=64)
    args.add_argument('--num_workers', type=int,
                      default=4)  # workers for dataloader

    args = args.parse_args()

    # load train.txt
    train_df = pd.read_csv('./dataset/train.txt', sep=',')
    test_df = pd.read_csv('./dataset/test_without_label.txt', sep=',')

    # split train_df into train_df and val_df

    train_df, val_df = train_test_split(
        train_df, test_size=0.1, random_state=2023, stratify=train_df['tag'])
    print('train_df shape: ', train_df.shape)
    print('val_df shape: ', val_df.shape)
    print('test_df shape (label all none): ', test_df.shape)

    # image transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    # train dataloader
    train_dataset = MultiModalDataset(
        train_df, './dataset/data', transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)

    # val dataloader
    val_dataset = MultiModalDataset(
        val_df, './dataset/data', transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    # test dataloader
    test_dataset = MultiModalDataset(
        test_df, './dataset/data', transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    # save dataloader
    torch.save(train_dataloader, './dataset/dataloader/train_dataloader.pth')
    torch.save(val_dataloader, './dataset/dataloader/val_dataloader.pth')
    torch.save(test_dataloader, './dataset/dataloader/test_dataloader.pth')
