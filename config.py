import os
import torch
import json
from transformers import BertTokenizer


class Params:
    basedir = os.path.abspath(os.path.dirname(__file__))
    data_basedir = f'{basedir}/data'
    pretrain_model_basedir = f'{basedir}/pretrain_models'

    dataset_list = ['NYT', 'NYT-star', 'WebNLG', 'WebNLG-star']
    dataset_json_list = ['rel2id', 'train_triples', 'val_triples', 'test_triples']

    bert_config_json_path = f'{pretrain_model_basedir}/config.json'
    bert_vocab_path = f'{pretrain_model_basedir}/vocab.txt'
    max_seq_len = 100

    tokenizer = BertTokenizer(vocab_file=bert_vocab_path, do_lower_case=False)

    # ablation params
    use_link = True

    def __init__(self, experiments_index=1, corpus_type='NYT'):
        self.experiments_index = experiments_index
        self.corpus_type = corpus_type
        self.experiments_dir = f'{Params.basedir}/experiments/ex{self.experiments_index}'
        self.models_dir = f'{Params.basedir}/models/ex{self.experiments_index}'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_gpu = torch.cuda.device_count()
        self.data_cache = False
        self.train_bs = 6 if 'WebNLG' in corpus_type else 64
        self.val_bs = 24
        self.test_bs = 64

        # load rel2idx
        self.rel2idx = json.load(
            open(Params.generate_data_path('rel2id', self.corpus_type),
                 'r', encoding='utf-8-sig'))[-1]
        self.rel_num = len(self.rel2idx.keys())
        self.threshold_ent = 0.5
        self.threshold_link = 0.5

        # early stop strategy
        self.min_epoch_num = 70
        self.patience = 0.00001
        self.patience_num = 20

        # learning rate
        self.fine_tuning_lr = 1e-4
        self.downs_stream_lr = 1e-3
        self.clip_grad = 2.0
        self.dropout = 0.3
        self.weight_decay_rate = 0.01
        self.warmup_prop = 0.1
        self.gradient_accumulation_steps = 2

    def load(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        """save to json file"""
        params = {}
        with open(json_path, 'w') as f:
            for k, v in self.__dict__.items():
                if isinstance(v, (str, int, float, bool)):
                    params[k] = v
            json.dump(params, f, indent=4)

    @staticmethod
    def generate_data_path(dataset_json, dataset):
        '''
        generate json data's path with 'dataset' and 'dataset_json'

        :param dataset_json: from ('rel2id', 'train_triples', 'val_triples', 'test_triples')
        :param dataset: from ('NYT', 'NYT-star', 'WebNLG', 'WebNLG-star')
        :return: json data path str
        '''
        if dataset not in Params.dataset_list or dataset_json not in Params.dataset_json_list:
            raise Exception('There is not such a file.')
        else:
            return f'{Params.data_basedir}/{dataset}/{dataset_json}.json'
