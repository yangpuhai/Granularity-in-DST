import numpy as np
import json
from torch.utils.data import Dataset
import torch
import random
import re
from copy import deepcopy
from .fix_dstc import SEMANTIC_DICT

flatten = lambda x: [i for s in x for i in s]
EXPERIMENT_DOMAINS = ["restaurant"]
SLOT=['restaurant-pricerange','restaurant-food','restaurant-area']
domain2id = {d: i for i, d in enumerate(EXPERIMENT_DOMAINS)}

OP = {'none': 0, 'dontcare': 1, 'span': 2}

def slot_value_span(value, diag):
    sequences = diag
    patt = ' ' + value + ' '
    pattern = re.compile(patt)
    m = pattern.finditer(sequences)
    m = [mi for mi in m]
    if m != []:
        line_st = sequences[:m[-1].span()[0]]
        start = len(line_st.split())
        slot_v = [start, start + len(value.split())-1]
    else:
        slot_v = [0, 0]
    return slot_v

def make_span(turn_utter, tokenizer, value):
    diag = tokenizer.tokenize(turn_utter)
    diag = ["[CLS]"] + diag
    diag_text = ' '.join(diag)
    value_text = ' '.join(tokenizer.tokenize(value))
    result = slot_value_span(value_text, diag_text)
    if result == []:
        if value in SEMANTIC_DICT:
            for v in SEMANTIC_DICT[value]:
                v_text = ' '.join(tokenizer.tokenize(v))
                result1 = slot_value_span(v_text, diag_text)
                if result1 != []:
                    result = result1
                    break
    return result

def make_word_idx(turn_utter, tokenizer):
    diag = turn_utter.split()
    diag = ["[CLS]"] + diag
    diag_uttr = diag
    word_idx = []
    diag_len = 0
    for word in diag:
        word_list = tokenizer.tokenize(word)
        list_len = len(word_list)
        word_idx.extend([diag_len] * list_len)
        diag_len += 1
    return word_idx, diag_uttr

def make_turn_label(turn_utter, slot_meta, last_dialog_state, turn_dialog_state, tokenizer, dynamic=False):
    if dynamic:
        gold_state = turn_dialog_state
        turn_dialog_state = {}
        for x in gold_state:
            s = x.split('-')
            k = '-'.join(s[:2])
            turn_dialog_state[k] = s[2]

    word_idx, diag_uttr = make_word_idx(turn_utter, tokenizer)

    op_labels = []
    generate_y = []
    for k in slot_meta:
        v = turn_dialog_state.get(k)
        vv = last_dialog_state.get(k)
        if vv != v:
            if v == 'dontcare' :
                op_labels.append('dontcare')
                generate_y.append([0, 0])
            else:
                op_labels.append('span')
                generate_y.append(make_span(turn_utter, tokenizer, v))
        else:
            op_labels.append('none')
            generate_y.append([0, 0])

    gold_state = [str(k) + '-' + str(v) for k, v in turn_dialog_state.items()]

    if dynamic:
        op2id = OP
        op_labels = [op2id[i] for i in op_labels]

    return op_labels, generate_y, gold_state, word_idx, diag_uttr


def postprocessing(slot_meta, ops, last_dialog_state,
                   generated, input_, gold_gen, word_idx, diag_uttr):
    gid = 0
    for st, op in zip(slot_meta, ops):
        if op == 'dontcare' :
            last_dialog_state[st] = 'dontcare'
        elif op == 'span':
            g = input_[generated[gid][0]:generated[gid][1]+1]
            gen = []
            for gg in g:
                gen.append(gg)
            gen = ' '.join(gen).replace(' ##', '')
            gen = gen.replace(' : ', ':').replace('##', '')
            last_dialog_state[st] = gen
        gid += 1
    return generated, last_dialog_state

def state_equal(pred_dialog_state, gold_dialog_state, slot_meta):
    equal = True
    for slot in slot_meta:
        pred_value = pred_dialog_state.get(slot)
        gold_value = gold_dialog_state.get(slot)
        if pred_value != gold_value:
            equal = False
            for s in SEMANTIC_DICT:
                if pred_value in [s]+SEMANTIC_DICT[s]:
                    for s1 in [s]+SEMANTIC_DICT[s]:
                        if s1 == gold_value:
                            equal = True
                            pred_dialog_state[slot] = s
                            break
    return pred_dialog_state, equal

def make_slot_meta(ontology):
    meta = []
    change = {}
    idx = 0
    max_len = 0
    for i, k in enumerate(ontology.keys()):
        d, s = k.split('-')
        if d not in EXPERIMENT_DOMAINS:
            continue
        if 'price' in s or 'leave' in s or 'arrive' in s:
            s = s.replace(' ', '')
        ss = s.split()
        if len(ss) + 1 > max_len:
            max_len = len(ss) + 1
        meta.append('-'.join([d, s]))
        change[meta[-1]] = ontology[k]
    return sorted(meta), change

def process_state(state, slot_meta):
    result = {}
    for s_dict in state:
        slot = s_dict
        slot = EXPERIMENT_DOMAINS[0] + '-' + slot
        if slot not in slot_meta:
            continue
        value = state[s_dict]
        result[slot] = value
    return result

def create_instance(dialog_history, state_history, size_window, tokenizer, ti, len_turns, dialogue_id,
                    turn_domain, turn_id, turn_dialog_state, slot_meta, max_seq_length):
    if len(state_history) < size_window:
        window_dialog_state = state_history[0]
    else:
        window_dialog_state = state_history[len(state_history) - size_window]

    if (ti + 1) == len_turns:
        is_last_turn = True
    else:
        is_last_turn = False

    turn_utter = " ; ".join(dialog_history[-size_window:])

    op_labels, generate_y, gold_state, word_idx, diag_uttr = make_turn_label(turn_utter, slot_meta, window_dialog_state, turn_dialog_state, tokenizer)

    instance = TrainingInstance(dialogue_id,turn_domain, turn_id, turn_utter, window_dialog_state, turn_dialog_state, op_labels, generate_y, gold_state, max_seq_length, slot_meta, is_last_turn)
    instance.make_instance(tokenizer)
    return instance

def prepare_dataset(data_scale, data_path, tokenizer, slot_meta, size_window, max_seq_length, multi_granularity = False, data_type = ''):
    dials = json.load(open(data_path))
    data = []
    domain_counter = {}

    if data_type == 'train':
        random.seed(42)
        dials = random.sample(dials, int(data_scale*len(dials)))
        random.seed()

    for dial_dict in dials:
        dial_dict["domains"] = ["restaurant"]
        for domain in dial_dict["domains"]:
            if domain not in EXPERIMENT_DOMAINS:
                continue
            if domain not in domain_counter.keys():
                domain_counter[domain] = 0
            domain_counter[domain] += 1
        state_history = []
        dialog_history = []
        last_dialog_state = {}
        for ti, turn in enumerate(dial_dict["turns"]):
            turn_id = turn["turn_id"]
            turn_domain = "restaurant"
            if turn_domain not in EXPERIMENT_DOMAINS:
                continue

            if data_type == 'train':
                turn_uttr = (turn['system_transcript'] + ' [SEP] ' + turn['user_transcript']).strip()
            else:
                turn_uttr = (turn['system_transcript'] + ' [SEP] ' + turn['user_asr']).strip()
            
            dialog_history.append(turn_uttr)
            turn_dialog_state = process_state(turn['state'], slot_meta)
            state_history.append(last_dialog_state)

            len_turns = len(dial_dict['turns'])
            dialogue_id = dial_dict["caller_id"]
            if multi_granularity:
                max_size_window = min(size_window, len(dialog_history))
                for sw in range(1, max_size_window + 1):
                    instance = create_instance(dialog_history, state_history, sw, tokenizer, ti,
                                               len_turns, dialogue_id, turn_domain, turn_id, turn_dialog_state,
                                               slot_meta, max_seq_length)
                    data.append(instance)
            else:
                size_window1 = min(size_window, len(dialog_history))
                instance = create_instance(dialog_history, state_history, size_window1, tokenizer, ti,
                                           len_turns, dialogue_id, turn_domain, turn_id, turn_dialog_state,
                                           slot_meta, max_seq_length)
                data.append(instance)
            last_dialog_state = turn_dialog_state
    return data


class TrainingInstance:
    def __init__(self, ID,
                 turn_domain,
                 turn_id,
                 turn_utter,
                 last_dialog_state,
                 turn_dialog_state,
                 op_labels,
                 generate_y,
                 gold_state,
                 max_seq_length,
                 slot_meta,
                 is_last_turn):
        self.id = ID
        self.turn_domain = turn_domain
        self.turn_id = turn_id
        self.turn_utter = turn_utter
        self.last_dialog_state = last_dialog_state
        self.gold_p_state = last_dialog_state
        self.turn_dialog_state = turn_dialog_state
        self.generate_y = generate_y
        self.op_labels = op_labels
        self.gold_state = gold_state
        self.max_seq_length = max_seq_length
        self.slot_meta = slot_meta
        self.is_last_turn = is_last_turn
        self.op2id = OP

    def make_instance(self, tokenizer, max_seq_length=None, word_dropout=0.):
        if max_seq_length is None:
            max_seq_length = self.max_seq_length
        
        avail_length_1 = max_seq_length - 1
        diag = tokenizer.tokenize(self.turn_utter)

        if len(diag) > avail_length_1:
            avail_length = len(diag) - avail_length_1
            diag = diag[avail_length:]

        drop_mask = [0] + [1] * len(diag)
        diag = ["[CLS]"] + diag
        segment = [1] * len(diag)

        # word dropout
        if word_dropout > 0.:
            drop_mask = np.array(drop_mask)
            word_drop = np.random.binomial(drop_mask.astype('int64'), word_dropout)
            diag = [w if word_drop[i] == 0 else '[UNK]' for i, w in enumerate(diag)]
        input_ = diag
        segment = segment
        self.input_ = input_

        self.segment_id = segment

        input_mask = [1] * len(self.input_)
        self.input_id = tokenizer.convert_tokens_to_ids(self.input_)
        if len(input_mask) < max_seq_length:
            self.input_id = self.input_id + [0] * (max_seq_length-len(input_mask))
            self.segment_id = self.segment_id + [0] * (max_seq_length-len(input_mask))
            input_mask = input_mask + [0] * (max_seq_length-len(input_mask))

        self.input_mask = input_mask
        self.op_ids = [self.op2id[a] for a in self.op_labels]
        self.generate_ids = self.generate_y


class MultiWozDataset(Dataset):
    def __init__(self, data, tokenizer, slot_meta, max_seq_length, rng, word_dropout=0.1):
        self.data = data
        self.len = len(data)
        self.tokenizer = tokenizer
        self.slot_meta = slot_meta
        self.max_seq_length = max_seq_length
        self.word_dropout = word_dropout
        self.rng = rng

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.word_dropout > 0:
            self.data[idx].make_instance(self.tokenizer, word_dropout=self.word_dropout)
        return self.data[idx]

    def collate_fn(self, batch):
        input_ids = torch.tensor([f.input_id for f in batch], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in batch], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_id for f in batch], dtype=torch.long)
        op_ids = torch.tensor([f.op_ids for f in batch], dtype=torch.long)
        gen_ids = [b.generate_ids for b in batch]
        gen_ids = torch.tensor(gen_ids, dtype=torch.long)

        return input_ids, input_mask, segment_ids, op_ids, gen_ids
