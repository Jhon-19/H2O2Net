import random
import json
import logging
import argparse
import pandas as pd
from tqdm import tqdm
from dataloader_utils import span2ent, matrices2triple_spans
import torch
import numpy as np
from config import Params
import utils
from dataloader import CustomDataLoader

# parse args
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2023, help='random seed for initialization')
parser.add_argument('--ex_index', type=str, default=1)
parser.add_argument('--corpus_type', type=str, default='NYT', help='NYT, WebNLG, NYT-star, WebNLG-star')
parser.add_argument('--device_id', type=int, default=0, help='GPU index')
parser.add_argument('--restore_file', default='best', help='name of the file containing weights to reload')
parser.add_argument('--mode', type=str, default='test', help='test mode for evaluate in test dataset')
parser.add_argument('--threshold_ent', type=float, default=0.5, help='threshold of ent-rel prediction')
parser.add_argument('--threshold_link', type=float, default=0.5, help='threshold of link prediction')

parser.add_argument('--use_link', action='store_true', default=False, help='use link matrix')


def get_metrics(correct_num, predict_num, gold_num):
    '''
    compute precision, recall and F1
    :param correct_num: num of correct prediction
    :param predict_num: num of all prediction
    :param gold_num: num of gold triples
    :return: metrics_dict <correct_num, predict_num, gold_num, p, r, f1>
    '''
    p = correct_num / predict_num if predict_num > 0 else 0
    r = correct_num / gold_num if gold_num > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {
        'correct_num': correct_num,
        'predict_num': predict_num,
        'gold_num': gold_num,
        'precision': p,
        'recall': r,
        'f1': f1
    }


def spans2triples(input_tokens, triple_spans):
    '''
    convert triple_span to triples
    :param input_tokens: tokenized text
    :param triple_spans: span list of triple
    :return: triple <head_ent, tail_ent, rel_index>
    '''
    triples = []
    for triple_span in triple_spans:
        head_ent_span, tail_ent_span, rel_index = triple_span
        head_ent = span2ent(input_tokens, head_ent_span[0], head_ent_span[1])
        tail_ent = span2ent(input_tokens, tail_ent_span[0], tail_ent_span[1])
        triples.append((head_ent, tail_ent, rel_index))
    return triples


def evaluate(model, data_iterator, params, mode='val'):
    '''
    evaluate for one epoch
    :param model: the model for triples extraction
    :param data_iterator: val dataset loader
    :param params: params that model need
    :param mode: 'val' or 'test'
    '''
    model.eval()
    predictions = []
    ground_truths = []
    correct_num, predict_num, gold_num = 0, 0, 0

    index = 0
    index_list = []

    for batch in tqdm(data_iterator, unit='Batch', ascii=True):
        # to device
        batch = tuple(t.to(params.device) if isinstance(t, torch.Tensor) else t for t in batch)
        input_ids, attention_mask, input_tokens, triples = batch
        bs, seq_len = input_ids.size()

        # inference
        with torch.no_grad():
            head_start_matrix_pred, head_end_matrix_pred, \
            tail_start_matrix_pred, tail_end_matrix_pred, \
            link_matrix_pred = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            for i in range(bs):
                # predict all matrices
                head_start_matrix = np.where(head_start_matrix_pred[i] > params.threshold_ent, 1, 0)  # (seq_len, rel_num)
                head_end_matrix = np.where(head_end_matrix_pred[i] > params.threshold_ent, 1, 0)  # (seq_len, rel_num)
                tail_start_matrix = np.where(tail_start_matrix_pred[i] > params.threshold_ent, 1, 0)  # (seq_len, rel_num)
                tail_end_matrix = np.where(tail_end_matrix_pred[i] > params.threshold_ent, 1, 0)  # (seq_len, rel_num)
                link_matrix = np.where(link_matrix_pred[i] > params.threshold_link, 1, 0)  # (seq_len, rel_num)

                all_matrices = (head_start_matrix, head_end_matrix,
                                tail_start_matrix, tail_end_matrix,
                                link_matrix)
                triple_spans = matrices2triple_spans(all_matrices)

                predict_triples = spans2triples(input_tokens[i], triple_spans)

                gold_triple_spans = triples[i]
                gold_triples = spans2triples(input_tokens[i], gold_triple_spans)

                ground_truths.append(list(set(gold_triples)))
                predictions.append(list(set(predict_triples)))

                # counter
                correct_num += len(set(predict_triples) & set(gold_triples))
                predict_num += len(set(predict_triples))
                gold_num += len(set(gold_triples))
    metrics = get_metrics(correct_num, predict_num, gold_num)

    print(index_list)

    # logging loss, f1 and report
    metrics_str = '; '.join('{}: {:05.3f}'.format(k, v) for k, v in metrics.items())
    logging.info(f'-*- {mode} metrics: {metrics_str}')
    return metrics, predictions, ground_truths


if __name__ == '__main__':
    args = parser.parse_args()
    params = Params(experiments_index=args.ex_index, corpus_type=args.corpus_type)
    params.threshold_ent = args.threshold_ent
    params.threshold_link = args.threshold_link
    Params.use_link = args.use_link

    torch.cuda.set_device(args.device_id)
    print('current device: ', torch.cuda.current_device())
    mode = args.mode

    # set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    params.seed = args.seed

    utils.set_logger()
    dataloader = CustomDataLoader(params)

    # define the model
    logging.info('Loading the model...')
    logging.info(f'Path: {params.models_dir}/{args.restore_file}.pth.tar')
    # reload weights from the saved file
    model, optimizer = utils.load_checkpoint(f'{params.models_dir}/{args.restore_file}.pth.tar')
    model.to(params.device)
    logging.info('-*- done.')

    # load the dataset
    logging.info('Loading the dataset...')
    loader = dataloader.get_dataloader(mode=mode)
    logging.info('-*- done.')

    logging.info('Starting prediction...')
    _, predictions, ground_truths = evaluate(model, loader, params, mode=mode)
    with open(params.generate_data_path(f'{mode}_triples', dataset=args.corpus_type), 'r',
              encoding='utf-8-sig') as f_src:
        src = json.loads(f_src.read())
        df = pd.DataFrame(
            {
                'text': [sample['text'] for sample in src],
                'predictions': predictions,
                'truths': ground_truths
            }
        )
        df.to_csv(f'{params.experiments_dir}/{mode}_result.csv')
    logging.info('-*- done.')