import json

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from dataloader_utils import read_samples, samples2features
import os
import numpy as np


class FeatureDataset(Dataset):
    '''dataset of input features
    '''

    def __init__(self, features):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]


class CustomDataLoader:
    def __init__(self, params):
        self.params = params

        self.corpus_type = params.corpus_type
        self.data_cache = params.data_cache

        self.train_bs = params.train_bs
        self.val_bs = params.val_bs
        self.test_bs = params.test_bs

        self.max_seq_len = params.max_seq_len
        self.tokenizer = params.tokenizer

    @staticmethod
    def collate_fn_train(features):
        '''
        convert target features to tensors for training
        :param features: List[InputFeatures]
        :return: List[Tensors]
        '''
        input_ids = torch.from_numpy(np.array([f.input_ids for f in features], dtype=np.long))
        attention_mask = torch.from_numpy(np.array([f.attention_mask for f in features], dtype=np.long))
        head_start_matrix = torch.from_numpy(np.array([f.head_start_matrix for f in features], dtype=np.long))
        head_end_matrix = torch.from_numpy(np.array([f.head_end_matrix for f in features], dtype=np.long))
        tail_start_matrix = torch.from_numpy(np.array([f.tail_start_matrix for f in features], dtype=np.long))
        tail_end_matrix = torch.from_numpy(np.array([f.tail_end_matrix for f in features], dtype=np.long))
        link_matrix = torch.from_numpy(np.array([f.link_matrix for f in features], dtype=np.long))

        return input_ids, attention_mask, head_start_matrix, head_end_matrix, \
            tail_start_matrix, tail_end_matrix, link_matrix

    @staticmethod
    def collate_fn_test(features):
        '''
        convert target features to tensors for testing
        :param features: List[InputFeatures]
        :return: List[Tensors]
        '''
        input_ids = torch.from_numpy(np.array([f.input_ids for f in features], dtype=np.long))
        attention_mask = torch.from_numpy(np.array([f.attention_mask for f in features], dtype=np.long))
        input_tokens = [f.input_tokens for f in features]
        triples = [f.triples for f in features]
        return input_ids, attention_mask, input_tokens, triples

    def get_dataloader(self, mode='train'):
        '''
        construct dataloader
        :param mode: select from ['train', 'val', 'test']
        :return: DataLoader
        '''
        features = self.get_features(mode=mode)
        dataset = FeatureDataset(features)
        print(f'{len(features)} {mode} datas loaded.')
        print('=*=' * 10)
        # construct dataloader
        if mode == 'train':
            datasampler = RandomSampler(dataset)
            dataloader = DataLoader(
                dataset,
                sampler=datasampler,
                batch_size=self.train_bs,
                collate_fn=self.collate_fn_train
            )
        elif mode == 'val':
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(
                dataset,
                sampler=datasampler,
                batch_size=self.val_bs,
                collate_fn=self.collate_fn_test
            )
        elif mode in ('test', 'EPO', 'SEO', 'HTO', 'Normal', '1', '2', '3', '4', '5'):
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(
                dataset,
                sampler=datasampler,
                batch_size=self.test_bs,
                collate_fn=self.collate_fn_test
            )
        else:
            raise ValueError('please notice that the data can only be train/val/test!')

        return dataloader

    def get_features(self, mode):
        '''
        convert InputSamples to InputFeatures
        :param mode: select from ['train', 'val', 'test']
        :return: List[InputFeatures]
        '''
        print('=*=' * 10)
        print(f'Loading {mode} data')

        # get features
        cache_path = f'{self.params.data_basedir}/{self.corpus_type}/{mode}.cache.{self.max_seq_len}'
        if os.path.exists(cache_path) and self.data_cache:
            features = torch.load(cache_path)
        else:
            # get relation to idx
            rel2idx = self.params.rel2idx

            # get samples
            if mode in ('train', 'val', 'test'):
                samples = read_samples(
                    self.params.generate_data_path(f'{mode}_triples', self.corpus_type),
                    rel2idx
                )
            else:
                raise ValueError('the mode can be only train/val/test!')

            features = samples2features(
                samples=samples,
                max_seq_len=self.max_seq_len,
                rel2idx=rel2idx,
                mode=mode
            )

            # save data
            if self.data_cache:
                torch.save(features, cache_path)

        return features


if __name__ == '__main__':
    from config import Params

    params = Params(corpus_type='NYT')
    dataloader = CustomDataLoader(params)
    features = dataloader.get_features(mode='train')
    print(features[9].input_tokens)
    dataloader.get_features(mode='val')
    dataloader.get_features(mode='test')
