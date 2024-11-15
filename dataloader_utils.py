import numpy as np

from collections import defaultdict
import json
import functools
from config import Params

tokenizer = Params.tokenizer


class InputSample:
    """
    a single set of data samples
    """

    def __init__(self, text, ent_pair_list, rel_list, rel2ents):
        self.text = text
        self.ent_pair_list = ent_pair_list
        self.rel_list = rel_list
        self.rel2ents = rel2ents


class InputFeatures:
    """
    a single set of data features
    """

    def __init__(self,
                 input_tokens,
                 input_ids,
                 attention_mask,
                 triples=None,
                 head_start_matrix=None,
                 head_end_matrix=None,
                 tail_start_matrix=None,
                 tail_end_matrix=None,
                 link_matrix=None
                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.triples = triples
        self.head_start_matrix = head_start_matrix
        self.head_end_matrix = head_end_matrix
        self.tail_start_matrix = tail_start_matrix
        self.tail_end_matrix = tail_end_matrix
        self.link_matrix = link_matrix


def find_first_index(text_tokens, target):
    """
    find the first index in the tokenized sequence
    :param text_tokens: tokenized text
    :param target: tokenized token
    :return: first index of the target token
    """
    target_length = len(target)
    first_index = -1
    for i in range(len(text_tokens)):
        if text_tokens[i:i + target_length] == target:
            first_index = i
            break
    return first_index


def span2ent(text_tokens, start, end):
    """
    select tokens between start index and end index
    :param text_tokens: encoded test tokens
    :param start: start index of entity
    :param end: end index of entity
    :return: ent
    """
    ent_tokens = text_tokens[start:end + 1]
    return ' '.join(ent_tokens).replace(' ##', '')


def label_all_matrices(rel2ents, max_seq_len, rel_num, text_tokens):
    """
    label head_start_matrix, head_end_matrix, tail_start_matrix, tail_start_matrix,
     tail_end_matrix, head_link_matrix, tail_link_matrix
    """
    head_start_matrix = np.zeros((max_seq_len, rel_num), dtype=int)
    head_end_matrix = np.zeros((max_seq_len, rel_num), dtype=int)
    tail_start_matrix = np.zeros((max_seq_len, rel_num), dtype=int)
    tail_end_matrix = np.zeros((max_seq_len, rel_num), dtype=int)
    link_matrix = np.zeros((max_seq_len, max_seq_len), dtype=int)

    for rel_index, ent_pair_list in rel2ents.items():
        for ent_pair in ent_pair_list:
            head_ent, tail_ent = ent_pair
            head_ent_tokens = tokenizer.tokenize(head_ent)
            tail_ent_tokens = tokenizer.tokenize(tail_ent)
            head_start = find_first_index(text_tokens, head_ent_tokens)
            head_end = head_start + len(head_ent_tokens) - 1
            tail_start = find_first_index(text_tokens, tail_ent_tokens)
            tail_end = tail_start + len(tail_ent_tokens) - 1
            if head_start == -1 or tail_start == -1:
                continue

            head_start_matrix[head_start, rel_index] = 1
            head_end_matrix[head_end, rel_index] = 1
            tail_start_matrix[tail_start, rel_index] = 1
            tail_end_matrix[tail_end, rel_index] = 1
            link_matrix[head_start, tail_start] = 1

    return head_start_matrix, head_end_matrix, tail_start_matrix, tail_end_matrix, \
        link_matrix


def find_ent_rel_dict(start_matrix, end_matrix):
    start_indices, start_rel_indices = np.where(start_matrix == 1)
    end_indices, end_rel_indices = np.where(end_matrix == 1)
    ent_rel_dict = defaultdict(list)
    for i, start_rel_index in enumerate(start_rel_indices):
        sub_end_rels = np.where(end_rel_indices == start_rel_index)[0]
        sub_end_indices = end_indices[sub_end_rels]
        ends = np.where(sub_end_indices >= start_indices[i])[0]
        if len(ends) > 0:
            ent_span = (start_indices[i], sub_end_indices[ends[0]])
            ent_rel_dict[start_rel_index].append(ent_span)

    return ent_rel_dict, set(start_rel_indices) & set(end_rel_indices)


def matrices2triple_spans(all_matrices):
    """
        decode triple spans by linking head entity matrices and tail entity matrices
    """
    triple_spans = []
    head_start_matrix, head_end_matrix, \
        tail_start_matrix, tail_end_matrix, \
        link_matrix = all_matrices

    head_ent_rel_dict, head_rel_indices = find_ent_rel_dict(head_start_matrix, head_end_matrix)
    tail_ent_rel_dict, tail_rel_indices = find_ent_rel_dict(tail_start_matrix, tail_end_matrix)
    rel_indices = list(head_rel_indices & tail_rel_indices)

    for rel_index in rel_indices:
        head_ent_spans = head_ent_rel_dict[rel_index]
        tail_ent_spans = tail_ent_rel_dict[rel_index]
        for head_ent_span in head_ent_spans:
            head_ent_start, head_ent_end = head_ent_span
            for tail_ent_span in tail_ent_spans:
                tail_ent_start, tail_ent_end = tail_ent_span

                is_link = link_matrix[head_ent_start, tail_ent_start] == 1
                if is_link or not Params.use_link:
                    triple_spans.append((head_ent_span, tail_ent_span, rel_index))
    return triple_spans


def read_samples(json_data_path, rel2idx):
    """
    load data to InputSamples
    :param json_data_path: the path of json data
    :param rel2idx: rel2idx dict
    :return: list of InputSamples
    """
    samples = []

    # read src data
    with open(json_data_path, 'r', encoding='utf-8-sig') as f:
        data = json.loads(f.read())
        for sample in data:
            text = sample['text']
            rel2ents = defaultdict(list)
            ent_pair_list = []
            rel_list = []

            for triple in sample['triple_list']:
                ent_pair_list.append([triple[0], triple[-1]])
                rel_list.append(rel2idx[triple[1]])
                rel2ents[rel2idx[triple[1]]].append((triple[0], triple[-1]))

            input_sample = InputSample(
                text=text,
                ent_pair_list=ent_pair_list,
                rel_list=rel_list,
                rel2ents=rel2ents
            )
            samples.append(input_sample)
    print('Number of InputSamples: ', len(samples))
    return samples


def convert(sample, max_seq_len, rel2idx, mode):
    """
    convert sample to features
    :param sample: InputSample
    :param max_seq_len: max sequence length
    :param rel2idx: the dict of [relation, index]
    :param mode: train, val, test
    :return: List[InputFeatures] (the list is the triples' count in a sample)
    """
    text_tokens = tokenizer.tokenize(sample.text)
    # cut off
    if len(text_tokens) > max_seq_len:
        text_tokens = text_tokens[:max_seq_len]

    # token to id
    input_ids = tokenizer.convert_tokens_to_ids(text_tokens)
    attention_mask = [1] * len(input_ids)
    # zero-padding up to the sequence length
    if len(input_ids) < max_seq_len:
        pad_len = max_seq_len - len(input_ids)
        input_ids += [0] * pad_len
        attention_mask += [0] * pad_len

    # train data
    if mode == 'train':
        rel_num = len(rel2idx.values())
        head_start_matrix, head_end_matrix, tail_start_matrix, tail_end_matrix, \
            link_matrix = label_all_matrices(sample.rel2ents, max_seq_len, rel_num, text_tokens)
        sub_features = InputFeatures(
            input_tokens=text_tokens,
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_start_matrix=head_start_matrix,
            head_end_matrix=head_end_matrix,
            tail_start_matrix=tail_start_matrix,
            tail_end_matrix=tail_end_matrix,
            link_matrix=link_matrix,
        )
    # val and test data
    else:
        triples = []
        for rel_index, ent_pair in zip(sample.rel_list, sample.ent_pair_list):
            head_tokens = tokenizer.tokenize(ent_pair[0])
            tail_tokens = tokenizer.tokenize(ent_pair[1])
            head_start = find_first_index(text_tokens, head_tokens)
            tail_start = find_first_index(text_tokens, tail_tokens)
            if head_start != -1 and tail_start != -1:
                head_span = (head_start, head_start + len(head_tokens) - 1)
                tail_span = (tail_start, tail_start + len(tail_tokens) - 1)
                triples.append((head_span, tail_span, rel_index))
        sub_features = InputFeatures(
            input_tokens=text_tokens,
            input_ids=input_ids,
            attention_mask=attention_mask,
            triples=triples
        )
    return sub_features


def samples2features(samples, max_seq_len, rel2idx, mode):
    """
    convert all samples to features
    :param samples: InputSamples
    :param max_seq_len: max sequence length
    :param rel2idx: the dict of [relation, index]
    :param mode: train, val, test
    :return: List[InputFeatures](the list is the triples' count in all sample)
    """

    # multi-process
    convert_func = functools.partial(convert,
                                     max_seq_len=max_seq_len,
                                     rel2idx=rel2idx,
                                     mode=mode
                                     )
    features = map(convert_func, samples)

    return list(features)
