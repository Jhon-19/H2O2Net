import os
import random

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import torch
from torch import nn
from transformers import BertConfig
import logging
from tqdm import trange
import argparse
from optimization import BertAdam
from evaluate import evaluate
from dataloader import CustomDataLoader
from model import BertForRE
import utils
from config import Params
from datetime import datetime
import pandas as pd

# parse args
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2023, help='random seed for initialization')
parser.add_argument('--ex_index', type=str, default=1)
parser.add_argument('--corpus_type', type=str, default='NYT', help='NYT, WebNLG, NYT-star, WebNLG-star')
parser.add_argument('--device_id', type=int, default=0, help='GPU index')
parser.add_argument('--epoch_num', required=True, type=int, help='number of epochs')
parser.add_argument('--multi_gpu', action='store_true', help='ensure multi-gpu training')
parser.add_argument('--restore_file', default=None, help='name of the file containing weights to reload')
parser.add_argument('--threshold_ent', type=float, default=0.5, help='threshold of ent-rel prediction')
parser.add_argument('--threshold_link', type=float, default=0.5, help='threshold of link prediction')

parser.add_argument('--use_link', action='store_true', default=False, help='use link matrix')


def train(model, data_iterator, optimizer, params):
    '''
    train for one epoch
    :param model: the model of triples extraction
    :param data_iterator: train dataset loader
    :param optimizer: fine-tuning optimizer
    :param params: the params that model need
    '''

    # set model to training mode
    model.train()

    loss_avg = utils.RunningAverage()
    loss_avg_head_start = utils.RunningAverage()
    loss_avg_head_end = utils.RunningAverage()
    loss_avg_tail_start = utils.RunningAverage()
    loss_avg_tail_end = utils.RunningAverage()
    loss_avg_link = utils.RunningAverage()

    # use tqdm for progress bar
    t = trange(len(data_iterator), ascii=True)
    for step, _ in enumerate(t):
        # fetch the next training batch
        batch = next(iter(data_iterator))

        batch = tuple(t.to(params.device) if isinstance(t, torch.Tensor) else t for t in batch)

        input_ids, attention_mask, head_start_matrix, head_end_matrix, \
            tail_start_matrix, tail_end_matrix, link_matrix = batch

        # compute model output and loss
        loss, loss_head_start, loss_head_end, loss_tail_start, loss_tail_end, loss_link \
            = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_start_matrix=head_start_matrix,
            head_end_matrix=head_end_matrix,
            tail_start_matrix=tail_start_matrix,
            tail_end_matrix=tail_end_matrix,
            link_matrix=link_matrix,
        )

        if params.n_gpu > 1 and args.multi_gpu:
            loss = loss.mean()
        else:
            loss = loss / params.gradient_accumulation_steps

        # backward propagation
        loss.backward()

        if (step + 1) % params.gradient_accumulation_steps == 0:
            # performs updates using calculated gradients
            optimizer.step()
            model.zero_grad()

        # update the average loss
        loss_avg.update(loss.item() * params.gradient_accumulation_steps)
        loss_avg_head_start.update(loss_head_start.item())
        loss_avg_head_end.update(loss_head_end.item())
        loss_avg_tail_start.update(loss_tail_start.item())
        loss_avg_tail_end.update(loss_tail_end.item())
        loss_avg_link.update(loss_link.item())
        t.set_postfix(
            loss='{:05.3f}'.format(loss_avg()),
            loss_head_start='{:05.3f}'.format(loss_avg_head_start()),
            loss_head_end='{:05.3f}'.format(loss_avg_head_end()),
            loss_tail_start='{:05.3f}'.format(loss_avg_tail_start()),
            loss_tail_end='{:05.3f}'.format(loss_avg_tail_end()),
            loss_link='{:05.3f}'.format(loss_avg_link()),
        )


def train_and_evaluate(model, params, restore_file=None):
    '''
    train the model and evaluate in every epoch
    :param model: the model for triples extraction
    :param params: some params for training and evaluate
    :param restore_file: the file for containing weights to reload
    '''

    # load train data and val data
    dataloader = CustomDataLoader(params)
    train_loader = dataloader.get_dataloader(mode='train')
    val_loader = dataloader.get_dataloader(mode='val')

    # restore weights from restore_file if specified
    if restore_file is not None:
        restore_path = f'{params.models_dir}/{restore_file}.pth.tar'
        logging.info(f'Restoring parameters from {restore_path}')
        # load checkpoint
        model, optimizer = utils.load_checkpoint(restore_path)

    model.to(params.device)

    # parallel model
    if params.n_gpu > 1 and args.multi_gpu:
        model = nn.DataParallel(model)

    # prepare optimizer
    # fine tuning
    param_optimizer = list(model.named_parameters())
    # pretrain model param
    param_pretrain = [(n, p) for n, p in param_optimizer if 'bert' in n]
    # downstream model param
    param_downstream = [(n, p) for n, p in param_optimizer if 'bert' not in n]
    no_decay = ['bias', 'LayerNorm', 'layer_norm']
    optimizer_grouped_parameters = [
        # pretrain model param
        {'params': [p for n, p in param_pretrain if not any(nd in n for nd in no_decay)],
         'weight_decay': params.weight_decay_rate, 'lr': params.fine_tuning_lr
         },
        {'params': [p for n, p in param_pretrain if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': params.fine_tuning_lr
         },
        # downstream model
        {'params': [p for n, p in param_downstream if not any(nd in n for nd in no_decay)],
         'weight_decay': params.weight_decay_rate, 'lr': params.downs_stream_lr
         },
        {'params': [p for n, p in param_downstream if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': params.downs_stream_lr
         }
    ]
    num_train_optimization_steps = len(train_loader) // params.gradient_accumulation_steps * args.epoch_num
    optimizer = BertAdam(optimizer_grouped_parameters, warmup=params.warmup_prop, schedule="warmup_cosine",
                         t_total=num_train_optimization_steps, max_grad_norm=params.clip_grad)

    # patience stage
    best_val_f1 = 0.0
    patience_counter = 0

    # init_time = datetime.now()
    # records = [(0, 0)]
    for epoch in range(1, args.epoch_num + 1):
        logging.info(f'Epoch {epoch}/{args.epoch_num}')

        # train for one epoch on train set
        train(model, train_loader, optimizer, params)

        # evaluate for one epoch on val set
        val_metrics, _, _ = evaluate(model, val_loader, params, mode='val')
        val_f1 = val_metrics['f1']
        improve_f1 = val_f1 - best_val_f1

        # save weights of the network
        model_to_save = model.module if hasattr(model, 'module') else model  # only save the model itself
        optimizer_to_save = optimizer
        utils.save_checkpoint(
            {
                'epoch': epoch + 1,
                'model': model_to_save,
                'optimizer': optimizer_to_save
            },
            is_best=improve_f1 > 0,
            checkpoint=params.models_dir
        )
        params.save(f'{params.experiments_dir}/params.json')

        # stop training based params.patience
        if improve_f1 > 0:
            logging.info('-*- Found new best F1')
            best_val_f1 = val_f1
            if improve_f1 < params.patience:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1

        # early stopping and logging best f1
        if (patience_counter > params.patience_num and epoch > params.min_epoch_num) \
                or epoch == args.epoch_num:
            break


if __name__ == '__main__':
    args = parser.parse_args()
    params = Params(args.ex_index, args.corpus_type)
    params.threshold_ent = args.threshold_ent
    params.threshold_link = args.threshold_link
    Params.use_link = args.use_link

    if args.multi_gpu:
        params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        n_gpu = torch.cuda.device_count()
        params.n_gpu = n_gpu
    else:
        torch.cuda.set_device(args.device_id)
        print('current decice: ', torch.cuda.current_device())
        params.n_gpu = n_gpu = 1

    # set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    params.seed = args.seed
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # set logger
    utils.set_logger(save=True, log_path=f'{params.experiments_dir}/train.log')
    logging.info(f'device: {params.device}')

    logging.info('Load pre-train model weights...')
    bert_config = BertConfig.from_json_file(params.bert_config_json_path)
    model = BertForRE.from_pretrained(
        config=bert_config,
        pretrained_model_name_or_path=params.pretrain_model_basedir,
        params=params
    )
    logging.info('-*- done')

    # train and evaluate the model
    logging.info(f'Staring training for {args.epoch_num} epoch(s)')
    train_and_evaluate(model, params, args.restore_file)
