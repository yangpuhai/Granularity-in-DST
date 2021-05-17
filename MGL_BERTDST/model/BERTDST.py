import torch
import torch.nn as nn
from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel

class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))

class MGDST(BertPreTrainedModel):
    def __init__(self, config, n_op, n_slot):
        super(MGDST, self).__init__(config)
        self.hidden_size = config.hidden_size
        self.encoder = Encoder(config)
        self.decoder = Decoder(config, n_op, n_slot)
        self.apply(self.init_weights)
        #self.init_weights()

    def forward(self, input_ids, token_type_ids, attention_mask):
        sequence_output, pooled_output = self.encoder(input_ids=input_ids,
                                                      token_type_ids=token_type_ids,
                                                      attention_mask=attention_mask)
        dec_outputs = self.decoder(sequence_output=sequence_output,
                                   pooled_output=pooled_output)
        state_scores, span_scores = dec_outputs
        return state_scores, span_scores

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.bert = BertModel(config)

    def forward(self, input_ids, token_type_ids, attention_mask):
        bert_outputs = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output, pooled_output = bert_outputs[:2]
        return sequence_output, pooled_output

class Decoder(nn.Module):
    def __init__(self, config, n_op, n_slot):
        super(Decoder, self).__init__()
        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(config.dropout)

        for num in range(n_slot):
            action_cls = nn.Linear(config.hidden_size, n_op)
            self.add_module("action_cls_{}".format(num), action_cls)

            span_cls = nn.Linear(config.hidden_size, 2)
            self.add_module("span_cls_{}".format(num), span_cls)

        self.action_cls = AttrProxy(self, "action_cls_")
        self.span_cls = AttrProxy(self, "span_cls_")
        
        self.n_op = n_op
        self.n_slot = n_slot

    def forward(self, sequence_output, pooled_output):
        sequence_output = sequence_output.unsqueeze(0)
        pooled_output = pooled_output.unsqueeze(0)
        state_scores = None
        span_scores = None
        for i in range(self.n_slot):
            op_score = self.action_cls[i](self.dropout(pooled_output))
            state_scores = op_score if state_scores is None else torch.cat([state_scores, op_score], 0)

            sp_score = self.span_cls[i](self.dropout(sequence_output))
            span_scores = sp_score if span_scores is None else torch.cat([span_scores, sp_score], 0)

        state_scores = state_scores.transpose(0, 1)
        span_scores = span_scores.transpose(-1, -2)
        span_scores = nn.functional.softmax(span_scores, -1)
        
        return state_scores, span_scores.transpose(0, 1)

