# 自定义dataset
 
import torch
from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd
import config
# 1. 定义dataset类

class InputMethodDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_json(data_path, lines=True, orient='records').to_dict(orient='records')
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        input_tensor = torch.tensor(self.data[index]['input'], dtype=torch.long) 
        target_tensor = torch.tensor(self.data[index]['target'], dtype=torch.long)
        return input_tensor, target_tensor

# 2. 提供一个获取dataloader的函数
def get_dataloader(train=True):
    path = config.PROCESSED_DATA_DIR / ('train.jsonl' if train else 'test.jsonl')
    dataset = InputMethodDataset(path)
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

if __name__ == '__main__':
    # 测试dataset和dataloader
    train_dataloader = get_dataloader(train=True)
    test_dataloader = get_dataloader(train=False)
    print(len(train_dataloader))
    print(len(test_dataloader))

    for input_tensor, target_tensor in train_dataloader:
        print(input_tensor.shape)
        print(target_tensor.shape)
        break

    torch.cuda.is_available