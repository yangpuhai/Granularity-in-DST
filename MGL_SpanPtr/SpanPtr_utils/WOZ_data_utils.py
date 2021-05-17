import numpy as np
import json
from torch.utils.data import Dataset
import torch
import random
import re
from copy import deepcopy
from .fix_woz import SEMANTIC_DICT

SLOT=['price range','food', 'area']
OP = {'other': 0, 'none': 1, 'dontcare': 2}

PAD_token = 0
UNK_token = 1
SOS_token = 2
EOS_token = 3
SYS_token = 4
USR_token = 5

class Lang:
    def __init__(self):
        self.word2index = {}
        self.index2word = {PAD_token: "[PAD]", UNK_token: '[UNK]', SOS_token: "[SOS]", EOS_token: "[EOS]", SYS_token: "[SYS]", USR_token: "[USR]"}
        self.n_words = len(self.index2word)  # Count default tokens
        self.word2index = dict([(v, k) for k, v in self.index2word.items()])

    def index_words(self, sent, stype):
        if stype == 'list':
            for word in sent:
                self.index_word(word)
        if stype == 'str':
            for word in sent.split(" "):
                self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

    def convert_tokens_to_ids(self, sent):
        result=[]
        for t in sent:
            if t in self.word2index:
                result.append(self.word2index[t])
            else:
                result.append(UNK_token)
        return result

    def convert_ids_to_tokens(self, sent):
        result = []
        for t in sent:
            result.append(self.index2word[t])
        return result

def slot_value_span(value, diag):
    sequences = diag
    patt = value
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

def make_span(turn_utter, lang, value):
    diag_text = turn_utter
    value_text = value
    result = slot_value_span(value_text, diag_text)
    if result == [0, 0]:
        if value in SEMANTIC_DICT:
            for v in SEMANTIC_DICT[value]:
                result1 = slot_value_span(v, diag_text)
                if result1 != [0, 0]:
                    result = result1
                    break
    return result

def make_turn_label(window_utter, slot_meta, window_dialog_state, turn_dialog_state, lang, dynamic=False):
    op_labels = []
    generate_y = []
    for slot in slot_meta:
        turn_value = turn_dialog_state.get(slot)
        window_value = window_dialog_state.get(slot)
        if turn_value != window_value:
            if turn_value == 'dontcare':
                op_labels.append('dontcare')
                generate_y.append([0, 0])
            elif turn_value is None:
                op_labels.append('none')
                generate_y.append([0, 0])
            else:
                op_labels.append('other')
                generate_y.append(make_span(window_utter, lang, turn_value))
        else:
            op_labels.append('none')
            generate_y.append([0, 0])

    gold_state = [str(k) + '-' + str(v) for k, v in turn_dialog_state.items()]

    return op_labels, generate_y, gold_state


def postprocessing(slot_meta, ops, last_dialog_state, generated, input_, lang, gold_gen={}):
    gid = 0
    for st, op in zip(slot_meta, ops):
        if op == 'dontcare':
            last_dialog_state[st] = 'dontcare'
        elif op == 'other':
            #g = lang.convert_ids_to_tokens(generated[gid])
            g = input_[generated[gid][0]:generated[gid][1]+1]
            gen = []
            for gg in g:
                gen.append(gg)
            gen = ' '.join(gen)
            if '[SYS]' not in gen and '[USR]' not in gen:
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

def create_instance(dialog_history, state_history, size_window, lang, ti, len_turns, dialogue_id, turn_id,
                    turn_dialog_state, slot_meta):
    if len(state_history) < size_window:
        window_dialog_state = state_history[0]
    else:
        window_dialog_state = state_history[len(state_history) - size_window]

    if (ti + 1) == len_turns:
        is_last_turn = True
    else:
        is_last_turn = False

    instance = TrainingInstance(dialogue_id,
                                turn_id, " ; ".join(dialog_history[-size_window:]),
                                window_dialog_state, turn_dialog_state, slot_meta,
                                is_last_turn)
    instance.make_instance(lang)
    return instance

def load_data(train_dials, lang, slot_meta, size_window, MG=False, data_type = ''):
    # load training data
    data = []
    for dial_dict in train_dials:
        dialog_history = []
        state_history = []
        last_dialog_state = {}
        turn_id = 0
        for ti, turn in enumerate(dial_dict["dialogue"]):
            system_uttr = turn['system_transcript'].strip().lower()
            if data_type == 'train':
                user_uttr = turn['transcript'].strip().lower()
            else:
                user_uttr = turn['asr'][0][0].strip().lower()
            if system_uttr == '':
                turn_uttr = '[SYS]' + ' [USR] ' + user_uttr
            else:
                turn_uttr = '[SYS] ' + system_uttr + ' [USR] ' + user_uttr
            dialog_history.append(turn_uttr)
            turn_dialog_state = {}
            for s in turn['belief_state']:
                if s['act'] == 'inform':
                    turn_dialog_state[s['slots'][0][0]] = s['slots'][0][1]

            keys = list(turn_dialog_state.keys())
            for k in keys:
                if turn_dialog_state.get(k) == 'none':
                    turn_dialog_state.pop(k)
            state_history.append(last_dialog_state)

            len_turns = len(dial_dict['dialogue'])
            dialogue_id = dial_dict["dialogue_idx"]
            if MG:
                min_size_window = max(1, size_window + len(dialog_history))
                for sw in range(min_size_window, len(dialog_history)+1):
                    instance = create_instance(dialog_history, state_history, sw, lang, ti,
                                               len_turns, dialogue_id, turn_id, turn_dialog_state, slot_meta)
                    data.append(instance)
            else:
                size_window1 = max(1, size_window + len(dialog_history))
                instance = create_instance(dialog_history, state_history, size_window1, lang, ti,
                                           len_turns, dialogue_id, turn_id, turn_dialog_state, slot_meta)
                data.append(instance)
            last_dialog_state = turn_dialog_state
            turn_id += 1
    return data

def prepare_dataset(train_scale, random_seed, train_data_path, dev_data_path, test_data_path, slot_meta, train_size_window=0, train_MG=False, test_size_window=0, test_MG=False):
    train_dials = json.load(open(train_data_path))
    dev_dials = json.load(open(dev_data_path))
    test_dials = json.load(open(test_data_path))

    random.seed(42)
    train_dials = random.sample(train_dials, int(train_scale*len(train_dials)))
    random.seed()

    lang = Lang()
    for dial_dict in train_dials:
        for turn in dial_dict["dialogue"]:
            system_tokens = turn['system_transcript'].strip().split(' ')
            lang.index_words(system_tokens,'list')
            user_tokens = turn['transcript'].strip().split(' ')
            lang.index_words(user_tokens,'list')
            user_tokens_asr = turn['asr'][0][0].strip().split(' ')
            lang.index_words(user_tokens_asr,'list')
            for s in turn['belief_state']:
                if s['act'] == 'inform':
                    lang.index_words(s['slots'][0][0].split(" "), 'list')
                    lang.index_words(s['slots'][0][1].split(" "), 'list')

    #load training data
    train_data = load_data(train_dials, lang, slot_meta, train_size_window, train_MG, 'train')
    dev_data = load_data(dev_dials, lang, slot_meta, test_size_window, test_MG, 'dev')
    test_data = load_data(test_dials, lang, slot_meta, test_size_window, test_MG, 'test')

    return train_data, dev_data, test_data, lang


class TrainingInstance:
    def __init__(self, ID,
                 turn_id,
                 window_utter,
                 window_dialog_state,
                 turn_dialog_state,
                 slot_meta,
                 is_last_turn):
        self.id = ID
        self.turn_id = turn_id
        self.window_utter = window_utter
        self.window_dialog_state = window_dialog_state
        self.gold_p_state = window_dialog_state
        self.turn_dialog_state = turn_dialog_state
        self.slot_meta = slot_meta
        self.is_last_turn = is_last_turn
        self.op2id = OP

    def make_instance(self, lang, word_dropout=0.):

        #process text
        diag = self.window_utter
        diag= diag.strip().split(" ")
        drop_mask = [1] * len(diag)
        # word dropout
        if word_dropout > 0.:
            drop_mask = np.array(drop_mask)
            word_drop = np.random.binomial(drop_mask.astype('int64'), word_dropout)
            diag = [w if word_drop[i] == 0 or diag[i] == '[SYS]' or diag[i] == '[USR]' else '[UNK]' for i, w in enumerate(diag)]
        input_ = diag
        self.input_ = input_
        self.input_id = lang.convert_tokens_to_ids(self.input_)
        self.input_len = len(self.input_id)

        op_labels, generate_y, gold_state = make_turn_label(self.window_utter, self.slot_meta, self.window_dialog_state, self.turn_dialog_state, lang)
        self.op_labels = op_labels
        self.generate_y = generate_y
        self.gold_state = gold_state
        self.op_ids = [self.op2id[a] for a in self.op_labels]
        self.generate_ids = generate_y

class MultiWozDataset(Dataset):
    def __init__(self, data, lang, word_dropout=0.1):
        self.data = data
        self.len = len(data)
        self.lang = lang
        self.word_dropout = word_dropout

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.word_dropout > 0:
            self.data[idx].make_instance(self.lang, word_dropout=self.word_dropout)
        return self.data[idx]

    def collate_fn(self, batch):
        batch.sort(key=lambda x: x.input_len, reverse=True)
        input_ids = [f.input_id for f in batch]
        input_lens = [f.input_len for f in batch]
        max_input = max(input_lens)
        for idx, v in enumerate(input_ids):
            input_ids[idx] = v + [0] * (max_input - len(v))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_lens = torch.tensor(input_lens, dtype=torch.long)

        op_ids = torch.tensor([f.op_ids for f in batch], dtype=torch.long)
        gen_ids = [b.generate_ids for b in batch]
        gen_ids = torch.tensor(gen_ids, dtype=torch.long)

        return input_ids, op_ids, gen_ids, input_lens, max_input
