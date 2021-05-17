# import os
# os.environ['CUDA_VISIBLE_DEVICES']="5"
from SpanPtr_utils.eval_utils import compute_prf, compute_acc
from model.SpanPtr import MGDST
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import random
import numpy as np
import os
import time
import argparse
import json
from copy import deepcopy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):

    if args.dataset == 'sim-R':
        from SpanPtr_utils.simR_data_utils import prepare_dataset, postprocessing, state_equal, SLOT, OP
    if args.dataset == 'sim-M':
        from SpanPtr_utils.simM_data_utils import prepare_dataset, postprocessing, state_equal, SLOT, OP
    if args.dataset == 'DSTC2':
        from SpanPtr_utils.DSTC2_data_utils import prepare_dataset, postprocessing, state_equal, SLOT, OP
    if args.dataset == 'WOZ2.0':
        from SpanPtr_utils.WOZ_data_utils import prepare_dataset, postprocessing, state_equal, SLOT, OP
    if args.dataset == 'MultiWOZ2.1':
        from SpanPtr_utils.MultiWOZ_data_utils import prepare_dataset, postprocessing, state_equal, OP, make_slot_meta
        ontology = json.load(open(args.ontology_data_path))
        SLOT, ontology = make_slot_meta(ontology)
    
    slot_meta = SLOT
    train_data_raw, dev_data_raw, test_data_raw, lang = \
        prepare_dataset(train_scale = args.train_scale,
                        random_seed = args.random_seed,
                        train_data_path=args.train_data_path,
                        dev_data_path=args.dev_data_path,
                        test_data_path=args.test_data_path,
                        slot_meta=slot_meta,
                        train_size_window=args.train_size_window,
                        train_MG=args.train_MG,
                        test_size_window=args.test_size_window,
                        test_MG=args.test_MG)

    op2id = OP
    model = MGDST(lang, args.hidden_size, 0, len(op2id), slot_meta)
    ckpt = torch.load(args.model_ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt)

    model.eval()
    model.to(device)

    model_evaluation(postprocessing, state_equal, model, test_data_raw, lang, slot_meta, OP, 0, args.test_size_window, args.test_MG)


def model_evaluation_batch(postprocessing, state_equal, model, test_data, lang, slot_meta, OP, epoch, size_window = 0, multi_granularity=False,):
    model.eval()
    op2id = OP
    id2op = {v: k for k, v in op2id.items()}

    slot_turn_acc, joint_acc, slot_F1_pred, slot_F1_count = 0, 0, 0, 0
    final_joint_acc, final_count, final_slot_F1_pred, final_slot_F1_count = 0, 0, 0, 0

    results = {}
    #last_dialog_state = {}
    wall_times = []
    state_history = []

    max_count = len(test_data)
    batch_size = 100
    n = 0
    while n < max_count:
        batch = test_data[n:n+batch_size]
        n = n+batch_size

        batch.sort(key=lambda x: x.input_len, reverse=True)
        input_ids = [f.input_id for f in batch]
        input_lens = [f.input_len for f in batch]
        max_input = max(input_lens)
        for idx, v in enumerate(input_ids):
            input_ids[idx] = v + [0] * (max_input - len(v))
        input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
        input_lens = torch.tensor(input_lens, dtype=torch.long).to(device)

        start = time.time()

        max_value = 2
        with torch.no_grad():
            state_scores, gen_scores = model(input_ids=input_ids,
                                            input_lens=input_lens,
                                            max_value = max_value)
        
        for i in range(state_scores.size(0)):

            s = state_scores[i]
            g = gen_scores[i]
            _, op_ids = s.view(-1, len(op2id)).max(-1)

            if g.size(0) > 0:
                generated = g.max(-1)[1].tolist()
            else:
                generated = []
        
            pred_ops = [id2op[a] for a in op_ids.tolist()]
            gold_gen = {}

            generated, last_dialog_state = postprocessing(slot_meta, pred_ops, deepcopy(batch[i].window_dialog_state),
                                                      generated, batch[i].input_, lang, gold_gen)

            last_dialog_state, equal = state_equal(last_dialog_state, batch[i].turn_dialog_state, slot_meta)

            pred_state = []
            for k, v in last_dialog_state.items():
                pred_state.append('-'.join([k, v]))

            if equal:
                joint_acc += 1

            key = str(batch[i].id) + '_' + str(batch[i].turn_id)
            results[key] = [pred_ops, last_dialog_state, batch[i].op_labels, batch[i].turn_dialog_state]

            # Compute prediction slot accuracy
            temp_acc = compute_acc(set(batch[i].gold_state), set(pred_state), slot_meta)
            slot_turn_acc += temp_acc

            # Compute prediction F1 score
            temp_f1, temp_r, temp_p, count = compute_prf(batch[i].gold_state, pred_state)
            slot_F1_pred += temp_f1
            slot_F1_count += count

            if batch[i].is_last_turn:
                final_count += 1
                if set(pred_state) == set(batch[i].gold_state):
                    final_joint_acc += 1
                final_slot_F1_pred += temp_f1
                final_slot_F1_count += count
        
        end = time.time()
        wall_times.append(end - start)

    joint_acc_score = joint_acc / len(test_data)
    turn_acc_score = slot_turn_acc / len(test_data)
    slot_F1_score = slot_F1_pred / slot_F1_count
    final_joint_acc_score = final_joint_acc / final_count
    final_slot_F1_score = final_slot_F1_pred / final_slot_F1_count
    latency = np.sum(wall_times) * 1000 / len(test_data)

    print("------------------------------")
    print("Epoch %d joint accuracy : " % epoch, joint_acc_score)
    print("Epoch %d slot turn accuracy : " % epoch, turn_acc_score)
    print("Epoch %d slot turn F1: " % epoch, slot_F1_score)
    print("Final Joint Accuracy : ", final_joint_acc_score)
    print("Final slot turn F1 : ", final_slot_F1_score)
    print("Latency Per Prediction : %f ms" % latency)
    print("-----------------------------\n")
    #json.dump(results, open('preds_%d.json' % epoch, 'w'))

    scores = {'epoch': epoch, 'joint_acc': joint_acc_score,
              'slot_acc': turn_acc_score, 'slot_f1': slot_F1_score, 'final_slot_f1': final_slot_F1_score}
    return scores


def model_evaluation(postprocessing, state_equal, model, test_data, lang, slot_meta, OP, epoch, size_window = 0, multi_granularity=False,):
    model.eval()
    op2id = OP
    id2op = {v: k for k, v in op2id.items()}

    slot_turn_acc, joint_acc, slot_F1_pred, slot_F1_count = 0, 0, 0, 0
    final_joint_acc, final_count, final_slot_F1_pred, final_slot_F1_count = 0, 0, 0, 0

    results = {}
    #last_dialog_state = {}
    wall_times = []
    state_history = []
    for di, i in enumerate(test_data):

        if i.turn_id == 0:
            last_dialog_state = {}
            state_history = []

        pre_dialog_state = deepcopy(last_dialog_state)
        state_history.append(pre_dialog_state)

        size_window1 = max(1, size_window + len(state_history))

        if len(state_history) < size_window1:
            window_dialog_state = deepcopy(state_history[0])
        else:
            window_dialog_state = deepcopy(state_history[len(state_history) - size_window1])
        
        i.window_dialog_state = deepcopy(window_dialog_state)
        i.make_instance(lang, word_dropout=0.)
        
        input_ids = torch.LongTensor([i.input_id]).to(device)
        
        start = time.time()
        max_value = 2
        with torch.no_grad():
            s, g = model(input_ids=input_ids, input_lens=None, max_value = max_value)

        _, op_ids = s.view(-1, len(op2id)).max(-1)

        if g.size(1) > 0:
            generated = g.squeeze(0).max(-1)[1].tolist()
        else:
            generated = []
        
        pred_ops = [id2op[a] for a in op_ids.tolist()]
        gold_gen = {}

        generated, last_dialog_state = postprocessing(slot_meta, pred_ops, window_dialog_state,
                                                      generated, i.input_, lang, gold_gen)
        end = time.time()
        wall_times.append(end - start)

        last_dialog_state, equal = state_equal(last_dialog_state, i.turn_dialog_state, slot_meta)

        pred_state = []
        for k, v in last_dialog_state.items():
            pred_state.append('-'.join([k, v]))
        
        if equal:
            joint_acc += 1

        key = str(i.id) + '_' + str(i.turn_id)
        results[key] = [pred_ops, last_dialog_state, i.op_labels, i.turn_dialog_state]

        # Compute prediction slot accuracy
        temp_acc = compute_acc(set(i.gold_state), set(pred_state), slot_meta)
        slot_turn_acc += temp_acc

        # Compute prediction F1 score
        temp_f1, temp_r, temp_p, count = compute_prf(i.gold_state, pred_state)
        slot_F1_pred += temp_f1
        slot_F1_count += count

        if i.is_last_turn:
            final_count += 1
            if set(pred_state) == set(i.gold_state):
                final_joint_acc += 1
            final_slot_F1_pred += temp_f1
            final_slot_F1_count += count

    joint_acc_score = joint_acc / len(test_data)
    turn_acc_score = slot_turn_acc / len(test_data)
    slot_F1_score = slot_F1_pred / slot_F1_count
    final_joint_acc_score = final_joint_acc / final_count
    final_slot_F1_score = final_slot_F1_pred / final_slot_F1_count
    latency = np.mean(wall_times) * 1000

    print("------------------------------")
    print("Epoch %d joint accuracy : " % epoch, joint_acc_score)
    print("Epoch %d slot turn accuracy : " % epoch, turn_acc_score)
    print("Epoch %d slot turn F1: " % epoch, slot_F1_score)
    print("Final Joint Accuracy : ", final_joint_acc_score)
    print("Final slot turn F1 : ", final_slot_F1_score)
    print("Latency Per Prediction : %f ms" % latency)
    print("-----------------------------\n")
    #json.dump(results, open('preds_%d.json' % epoch, 'w'))

    scores = {'epoch': epoch, 'joint_acc': joint_acc_score,
              'slot_acc': turn_acc_score, 'slot_f1': slot_F1_score, 'final_slot_f1': final_slot_F1_score}
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='MultiWOZ2.1', type=str)
    parser.add_argument("--hidden_size", default=400, type=float)
    parser.add_argument("--random_seed", default=42, type=int)

    parser.add_argument("--train_size_window", default=0, type=int) # 0=max_len, -1=max_len-1
    parser.add_argument("--train_MG", default=True) # 0=max_len, -1=max_len-1
    parser.add_argument("--train_scale", default=1.0, type=int)

    parser.add_argument("--test_size_window", default=0, type=int) # 0=max_len, -1=max_len-1
    parser.add_argument("--test_MG", default=False) # 0=max_len, -1=max_len-1

    args = parser.parse_args()
    model_name = 'model_best_gran[%s]_scale[%s]_seed[%s].bin'% (str(args.train_size_window), str(args.train_scale), args.random_seed)
    if args.dataset == 'sim-R':
        data_root = 'data/M2M/sim-R'
        args.train_data_path = os.path.join(data_root, 'train.json')
        args.dev_data_path = os.path.join(data_root, 'dev.json')
        args.test_data_path = os.path.join(data_root, 'test.json')
        args.model_ckpt_path = 'outputs/SpanPtr/sim-R_outputs/' + model_name
    elif args.dataset == 'sim-M':
        data_root = 'data/M2M/sim-M'
        args.train_data_path = os.path.join(data_root, 'train.json')
        args.dev_data_path = os.path.join(data_root, 'dev.json')
        args.test_data_path = os.path.join(data_root, 'test.json')
        args.model_ckpt_path = 'outputs/SpanPtr/sim-M_outputs/' + model_name
    elif args.dataset == 'DSTC2':
        data_root = 'data/DSTC2'
        args.train_data_path = os.path.join(data_root, 'train.json')
        args.dev_data_path = os.path.join(data_root, 'dev.json')
        args.test_data_path = os.path.join(data_root, 'test.json')
        args.model_ckpt_path = 'outputs/SpanPtr/DSTC2_outputs/' + model_name
    elif args.dataset == 'WOZ2.0':
        data_root = 'data/WOZ2.0'
        args.train_data_path = os.path.join(data_root, 'woz_train_en.json')
        args.dev_data_path = os.path.join(data_root, 'woz_validate_en.json')
        args.test_data_path = os.path.join(data_root, 'woz_test_en.json')
        args.model_ckpt_path = 'outputs/SpanPtr/WOZ_outputs/' + model_name
    elif args.dataset == 'MultiWOZ2.1':
        data_root = 'data/mwz2.1'
        args.train_data_path = os.path.join(data_root, 'train_dials.json')
        args.dev_data_path = os.path.join(data_root, 'dev_dials.json')
        args.test_data_path = os.path.join(data_root, 'test_dials.json')
        args.ontology_data_path = os.path.join(data_root, 'ontology.json')
        args.model_ckpt_path = 'outputs/SpanPtr/MultiWOZ_outputs/' + model_name
    else:
        print('select dataset in sim-R, sim-M, DSTC2, WOZ2.0 and MultiWOZ2.1')
        exit()
    main(args)
