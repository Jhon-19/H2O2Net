from transformers import BertPreTrainedModel, BertModel
from torch import nn
import torch
import numpy as np


class SequenceClassifier(nn.Module):
    '''
    dual linear layers for classification
    '''

    def __init__(self, input_dim, output_dim, dropout_rate=0):
        super(SequenceClassifier, self).__init__()
        self.output_dim = output_dim
        self.hidden_linear = nn.Linear(input_dim, int(input_dim / 2))
        self.tag_linear = nn.Linear(int(input_dim / 2), self.output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        features = self.hidden_linear(input_features)
        features = nn.ReLU()(features)
        features = self.dropout(features)
        features_output = self.tag_linear(features)
        return features_output


class BertForRE(BertPreTrainedModel):
    def __init__(self, config, params):
        super().__init__(config)
        self.params = params

        self.max_seq_len = params.max_seq_len
        self.rel_num = params.rel_num
        self.dropout = params.dropout

        self.bert = BertModel(config)

        self.head_start_classifier = nn.Linear(config.hidden_size, self.rel_num)
        self.head_end_classifier = nn.Linear(config.hidden_size, self.rel_num)
        self.tail_start_classifier = nn.Linear(config.hidden_size, self.rel_num)
        self.tail_end_classifier = nn.Linear(config.hidden_size, self.rel_num)

        if params.use_link:
            self.link_classifier = SequenceClassifier(config.hidden_size * 2, 1, params.dropout)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            head_start_matrix=None,
            head_end_matrix=None,
            tail_start_matrix=None,
            tail_end_matrix=None,
            link_matrix=None,
    ):
        '''
        model for relation extraction
        :param link_matrix: (bs, seq_len, seq_len)
        :param input_ids: (bs, seq_len)
        :param attention_mask: (bs, seq_len)
        :param head_start_matrix: (bs, seq_len, rel_num)
        :param head_end_matrix: (bs, seq_len, rel_num)
        :param tail_start_matrix: (bs, seq_len, rel_num)
        :param tail_end_matrix: (bs, seq_len, rel_num)
        :return: loss
        '''
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        seq_output = outputs[0]  # (bs, seq_len, d)
        bs, seq_len, d = seq_output.size()

        # predict all matrices
        # (bs, seq_len, rel_num)
        head_start_matrix_pred = self.head_start_classifier(seq_output)
        # (bs, seq_len, rel_num)
        head_end_matrix_pred = self.head_end_classifier(seq_output)
        # (bs, seq_len, rel_num)
        tail_start_matrix_pred = self.tail_start_classifier(seq_output)
        # (bs, seq_len, rel_num)
        tail_end_matrix_pred = self.tail_end_classifier(seq_output)

        # for every position $i$ in sequence, should concat $j$ to predict.
        head_extend = seq_output.unsqueeze(2).expand(-1, -1, seq_len, -1)  # (bs, seq_len, seq_len, d)
        tail_extend = seq_output.unsqueeze(1).expand(-1, seq_len, -1, -1)  # (bs, seq_len, seq_len, d)
        link_seq = torch.cat([head_extend, tail_extend], 3)  # (bs, seq_len, seq_len, d*2)

        mask = attention_mask.unsqueeze(-1)

        mask_temp1 = attention_mask.unsqueeze(-1)
        mask_temp2 = attention_mask.unsqueeze(1)
        link_mask = mask_temp1 * mask_temp2

        if self.params.use_link:
            link_matrix_pred = self.link_classifier(link_seq).squeeze(-1)  # (bs, seq_len, seq_len)

        # train mode
        if head_end_matrix is not None:
            loss_func = nn.BCEWithLogitsLoss(reduction='none')

            loss_head_start = self.compute_loss(
                head_start_matrix_pred,
                head_start_matrix.float(),
                mask=mask,
                loss_func=loss_func
            )
            loss_head_end = self.compute_loss(
                head_end_matrix_pred,
                head_end_matrix.float(),
                mask=mask,
                loss_func=loss_func
            )
            loss_tail_start = self.compute_loss(
                tail_start_matrix_pred,
                tail_start_matrix.float(),
                mask=mask,
                loss_func=loss_func
            )
            loss_tail_end = self.compute_loss(
                tail_end_matrix_pred,
                tail_end_matrix.float(),
                mask=mask,
                loss_func=loss_func
            )

            loss_link = torch.tensor(0)
            if self.params.use_link:
                loss_link = self.compute_loss(
                    link_matrix_pred,
                    link_matrix.float(),
                    mask=link_mask,
                    loss_func=loss_func
                )

            loss = loss_head_start + loss_head_end + loss_tail_start + loss_tail_end + loss_link
            return loss, loss_head_start, loss_head_end, loss_tail_start, loss_tail_end, \
                loss_link
        # predict mode
        else:
            head_start_matrix_pred = nn.Sigmoid()(head_start_matrix_pred * mask).detach().cpu().numpy()
            head_end_matrix_pred = nn.Sigmoid()(head_end_matrix_pred * mask).detach().cpu().numpy()
            tail_start_matrix_pred = nn.Sigmoid()(tail_start_matrix_pred * mask).detach().cpu().numpy()
            tail_end_matrix_pred = nn.Sigmoid()(tail_end_matrix_pred * mask).detach().cpu().numpy()

            if self.params.use_link:
                link_matrix_pred = nn.Sigmoid()(link_matrix_pred * link_mask).detach().cpu().numpy()
            else:
                link_matrix_pred = np.zeros((bs, seq_len, seq_len))

            return head_start_matrix_pred, head_end_matrix_pred, \
                tail_start_matrix_pred, tail_end_matrix_pred, \
                link_matrix_pred

    @staticmethod
    def compute_loss(matrix_pred, matrix, mask, loss_func):
        loss = (loss_func(matrix_pred, matrix) * mask).sum()
        return loss


if __name__ == '__main__':
    from config import Params
    from transformers import BertConfig

    Params.use_link = False
    params = Params(corpus_type='NYT-star')
    bert_config = BertConfig.from_json_file(params.bert_config_json_path)
    model = BertForRE.from_pretrained(
        config=bert_config,
        pretrained_model_name_or_path=params.pretrain_model_basedir,
        params=params
    )
    model.to(params.device)

