import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random

__all__ = ['MMDataLoader']

class MMDataset(Dataset): 
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.train_mode = args['base']['train_mode'] #train_model代表使用回归/分类
        self.datasetName = args['dataset']['datasetName'] 
        self.dataPath = args['dataset']['dataPath']
               
        DATA_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei,
        }
        DATA_MAP[self.datasetName]()

    def __init_mosi(self):
        with open(self.dataPath, 'rb') as f: #读取数据
            data = pickle.load(f)

        self.data = data

        # 加载对应训练模式下（train）的模态数据，并转为 float32
        self.text = data[self.mode]['text'].astype(np.float32)
        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.audio = data[self.mode]['audio'].astype(np.float32)

        # 替换 inf 和 nan 为 0
        self.text = np.nan_to_num(self.text, nan=0.0, posinf=0.0, neginf=0.0)
        self.audio = np.nan_to_num(self.audio, nan=0.0, posinf=0.0, neginf=0.0)
        self.vision = np.nan_to_num(self.vision, nan=0.0, posinf=0.0, neginf=0.0)

        # 提取ID 和标签
        self.ids = data[self.mode]['id']
        self.labels = {
            'M': data[self.mode][self.train_mode + '_labels'].astype(np.float32)
        }



    def __init_mosei(self):
        return self.__init_mosi()

    def __len__(self):
        return len(self.labels['M'])
       
        
    def __getitem__(self, index): #拼装数据
        text = torch.tensor(self.text[index], dtype=torch.float32)
        audio = torch.tensor(self.audio[index], dtype=torch.float32)
        vision = torch.tensor(self.vision[index], dtype=torch.float32)
        
        sample = {
            'text': text,
            'audio': audio,
            'vision': vision,
            'index': index,
            'id': self.ids[index],
            'labels': {k: torch.tensor(np.array(v[index]).reshape(-1), dtype=torch.float32) for k, v in self.labels.items()}
        }
        return sample
        


def MMDataLoader(args): #加载data构建 DataLoader 用于训练
    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }

    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args['base']['batch_size'],
                       num_workers=args['base']['num_workers'],
                       shuffle=True if ds == 'train' else False)
        for ds in datasets.keys()
    }

    return dataLoader

def MMDataEvaluationLoader(args): #加载data构建 DataLoader 用于评估
    dataset = MMDataset(args, mode='test')
    dataLoader = DataLoader(dataset,
                            batch_size=args['base']['batch_size'],
                            num_workers=args['base']['num_workers'],
                            shuffle=False)
    return dataLoader
