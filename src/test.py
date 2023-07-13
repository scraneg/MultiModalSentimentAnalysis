import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from model import MultiModalModel
from tqdm import tqdm
# for not error when load dataloader
from process import MultiModalDataset, collate_fn



if __name__ == '__main__':
    test_dataloader = torch.load('./dataset/dataloader/test_dataloader.pth')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(device))
    model = MultiModalModel(n_classes=3)
    model.load_state_dict(torch.load('./models/model_best.pth'))
    model.to(device)
    model.eval()
    print('start test')
    correct = 0
    pred_list = []
    with torch.no_grad():
        for batch_index, (imgs, texts, atten_masks, token_type_ids, labels) in tqdm(enumerate(test_dataloader)):
            imgs = imgs.to(device)
            texts = texts.to(device)
            atten_masks = atten_masks.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device) # no labels in test set, just ignore it
            output = model(imgs, texts, atten_masks, token_type_ids)
            pred = torch.argmax(output, dim=1)
            pred_list += pred.tolist()


    print('test done')

    # load test data
    test_df = pd.read_csv('./dataset/test_without_label.txt', sep=',')
    # save pred_list
    for i in range(len(test_df)):
        if int(pred_list[i]) == 0:
            test_df.loc[i, 'tag'] = 'positive'
        elif int(pred_list[i]) == 1:
            test_df.loc[i, 'tag'] = 'neutral'
        else:
            test_df.loc[i, 'tag'] = 'negative'


    test_df.to_csv('./dataset/test_with_pred_label.txt', index=False)        
    
    
        