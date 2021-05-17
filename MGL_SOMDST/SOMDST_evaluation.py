# import os
# os.environ['CUDA_VISIBLE_DEVICES']="1"
from SOMDST_utils.eval_utils import compute_prf, compute_acc
from pytorch_transformers import BertTokenizer, BertConfig

from model.SOMDST import MGDST
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
        from SOMDST_utils.simR_data_utils import prepare_dataset, make_turn_label, postprocessing, SLOT, domain2id, OP_SET
    if args.dataset == 'sim-M':
        from SOMDST_utils.simM_data_utils import prepare_dataset, make_turn_label, postprocessing, SLOT, domain2id, OP_SET
    if args.dataset == 'DSTC2':
        from SOMDST_utils.DSTC2_data_utils import prepare_dataset, make_turn_label, postprocessing, SLOT, domain2id, OP_SET
    if args.dataset == 'WOZ2.0':
        from SOMDST_utils.WOZ_data_utils import prepare_dataset, make_turn_label, postprocessing, SLOT, domain2id, OP_SET
    if args.dataset == 'MultiWOZ2.1':
        from SOMDST_utils.MultiWOZ_data_utils import prepare_dataset, make_turn_label, postprocessing, domain2id, OP_SET, make_slot_meta
        ontology = json.load(open(args.ontology_data_path))
        SLOT, ontology = make_slot_meta(ontology)
    
    slot_meta = SLOT
    tokenizer = BertTokenizer(args.vocab_path, do_lower_case=True)
    data = prepare_dataset(1.0,
                    args.test_data_path,
                    tokenizer,
                    slot_meta,
                    args.test_size_window,
                    args.n_history,
                    args.max_seq_length,
                    args.op_code,
                    args.test_MG)

    model_config = BertConfig.from_json_file(args.bert_config_path)
    model_config.dropout = 0.1
    op2id = OP_SET[args.op_code]
    model = MGDST(model_config, len(op2id), len(domain2id), op2id['update'])
    ckpt = torch.load(args.model_ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt)

    model.eval()
    model.to(device)
    model_evaluation(make_turn_label, postprocessing, domain2id, OP_SET, model, data, tokenizer, slot_meta, 0, args.op_code, args.test_size_window, args.test_MG)


def model_evaluation(make_turn_label, postprocessing, domain2id, OP_SET, model, test_data, tokenizer, slot_meta, epoch, op_code='4', size_window=1, multi_granularity=False,
                     is_gt_op=False, is_gt_p_state=False, is_gt_gen=False):
    model.eval()
    op2id = OP_SET[op_code]
    id2op = {v: k for k, v in op2id.items()}
    id2domain = {v: k for k, v in domain2id.items()}

    slot_turn_acc, joint_acc, slot_F1_pred, slot_F1_count = 0, 0, 0, 0
    final_joint_acc, final_count, final_slot_F1_pred, final_slot_F1_count = 0, 0, 0, 0
    op_acc, op_F1, op_F1_count = 0, {k: 0 for k in op2id}, {k: 0 for k in op2id}
    all_op_F1_count = {k: 0 for k in op2id}

    tp_dic = {k: 0 for k in op2id}
    fn_dic = {k: 0 for k in op2id}
    fp_dic = {k: 0 for k in op2id}

    results = {}
    last_dialog_state = {}
    wall_times = []
    state_history = []
    for di, i in enumerate(test_data):
        if i.turn_id == 0:
            last_dialog_state = {}
            state_history = []
        state_history.append(last_dialog_state)

        if len(state_history) < size_window:
            window_dialog_state = state_history[0]
        else:
            window_dialog_state = state_history[len(state_history) - size_window]

        if is_gt_p_state is False:
            i.last_dialog_state = deepcopy(window_dialog_state)
            i.make_instance(tokenizer, word_dropout=0.)
        else:  # ground-truth previous dialogue state
            window_dialog_state = deepcopy(i.gold_p_state)
            i.last_dialog_state = deepcopy(window_dialog_state)
            i.make_instance(tokenizer, word_dropout=0.)

        input_ids = torch.LongTensor([i.input_id]).to(device)
        input_mask = torch.FloatTensor([i.input_mask]).to(device)
        segment_ids = torch.LongTensor([i.segment_id]).to(device)
        state_position_ids = torch.LongTensor([i.slot_position]).to(device)

        d_gold_op, _, _ = make_turn_label(slot_meta, window_dialog_state, i.gold_state,
                                          tokenizer, op_code, dynamic=True)
        gold_op_ids = torch.LongTensor([d_gold_op]).to(device)

        start = time.perf_counter()
        MAX_LENGTH = 9
        with torch.no_grad():
            # ground-truth state operation
            gold_op_inputs = gold_op_ids if is_gt_op else None
            d, s, g = model(input_ids=input_ids,
                            token_type_ids=segment_ids,
                            state_positions=state_position_ids,
                            attention_mask=input_mask,
                            max_value=MAX_LENGTH,
                            op_ids=gold_op_inputs)

        _, op_ids = s.view(-1, len(op2id)).max(-1)

        if g.size(1) > 0:
            generated = g.squeeze(0).max(-1)[1].tolist()
        else:
            generated = []

        if is_gt_op:
            pred_ops = [id2op[a] for a in gold_op_ids[0].tolist()]
        else:
            pred_ops = [id2op[a] for a in op_ids.tolist()]
        gold_ops = [id2op[a] for a in d_gold_op]

        if is_gt_gen:
            # ground_truth generation
            gold_gen = {'-'.join(ii.split('-')[:2]): ii.split('-')[-1] for ii in i.gold_state}
        else:
            gold_gen = {}
        generated, last_dialog_state = postprocessing(slot_meta, pred_ops, window_dialog_state,
                                                      generated, tokenizer, op_code, gold_gen)
        end = time.perf_counter()
        wall_times.append(end - start)
        pred_state = []
        for k, v in last_dialog_state.items():
            pred_state.append('-'.join([k, v]))

        if set(pred_state) == set(i.gold_state):
            joint_acc += 1
        key = str(i.id) + '_' + str(i.turn_id)
        results[key] = [pred_state, i.gold_state]

        # Compute prediction slot accuracy
        temp_acc = compute_acc(set(i.gold_state), set(pred_state), slot_meta)
        slot_turn_acc += temp_acc

        # Compute prediction F1 score
        temp_f1, temp_r, temp_p, count = compute_prf(i.gold_state, pred_state)
        slot_F1_pred += temp_f1
        slot_F1_count += count

        # Compute operation accuracy
        temp_acc = sum([1 if p == g else 0 for p, g in zip(pred_ops, gold_ops)]) / len(pred_ops)
        op_acc += temp_acc

        if i.is_last_turn:
            final_count += 1
            if set(pred_state) == set(i.gold_state):
                final_joint_acc += 1
            final_slot_F1_pred += temp_f1
            final_slot_F1_count += count

        # Compute operation F1 score
        for p, g in zip(pred_ops, gold_ops):
            all_op_F1_count[g] += 1
            if p == g:
                tp_dic[g] += 1
                op_F1_count[g] += 1
            else:
                fn_dic[g] += 1
                fp_dic[p] += 1

    joint_acc_score = joint_acc / len(test_data)
    turn_acc_score = slot_turn_acc / len(test_data)
    slot_F1_score = slot_F1_pred / slot_F1_count
    op_acc_score = op_acc / len(test_data)
    final_joint_acc_score = final_joint_acc / final_count
    final_slot_F1_score = final_slot_F1_pred / final_slot_F1_count
    latency = np.mean(wall_times) * 1000
    op_F1_score = {}
    for k in op2id.keys():
        tp = tp_dic[k]
        fn = fn_dic[k]
        fp = fp_dic[k]
        precision = tp / (tp+fp) if (tp+fp) != 0 else 0
        recall = tp / (tp+fn) if (tp+fn) != 0 else 0
        F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
        op_F1_score[k] = F1

    print("------------------------------")
    print('op_code: %s, is_gt_op: %s, is_gt_p_state: %s, is_gt_gen: %s' % \
          (op_code, str(is_gt_op), str(is_gt_p_state), str(is_gt_gen)))
    print("Epoch %d joint accuracy : " % epoch, joint_acc_score)
    print("Epoch %d slot turn accuracy : " % epoch, turn_acc_score)
    print("Epoch %d slot turn F1: " % epoch, slot_F1_score)
    print("Epoch %d op accuracy : " % epoch, op_acc_score)
    print("Epoch %d op F1 : " % epoch, op_F1_score)
    print("Epoch %d op hit count : " % epoch, op_F1_count)
    print("Epoch %d op all count : " % epoch, all_op_F1_count)
    print("Final Joint Accuracy : ", final_joint_acc_score)
    print("Final slot turn F1 : ", final_slot_F1_score)
    print("Latency Per Prediction : %f ms" % latency)
    print("-----------------------------\n")
    # json.dump(results, open('preds_%d.json' % epoch, 'w'), indent=4)

    scores = {'epoch': epoch, 'joint_acc': joint_acc_score,
              'slot_acc': turn_acc_score, 'slot_f1': slot_F1_score,
              'op_acc': op_acc_score, 'op_f1': op_F1_score, 'final_slot_f1': final_slot_F1_score}
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='MultiWOZ2.1', type=str)
    parser.add_argument("--vocab_path", default='bert-base-uncased/vocab.txt', type=str)
    parser.add_argument("--bert_config_path", default='bert-base-uncased/bert_config_base_uncased.json', type=str)
    parser.add_argument("--model_ckpt_path", default='outputs/model_best.bin', type=str)
    parser.add_argument("--test_size_window", default=1, type=int)
    parser.add_argument("--test_MG", default=False)
    parser.add_argument("--n_history", default=1, type=int)
    parser.add_argument("--max_seq_length", default=320, type=int)
    parser.add_argument("--op_code", default="4", type=str)

    parser.add_argument("--random_seed", default=40, type=int)

    parser.add_argument("--train_size_window", default=1, type=int)
    parser.add_argument("--train_MG", default=True)
    parser.add_argument("--train_scale", default=1.0, type=float)

    parser.add_argument("--gt_op", default=False, action='store_true')
    parser.add_argument("--gt_p_state", default=False, action='store_true')
    parser.add_argument("--gt_gen", default=False, action='store_true')
    parser.add_argument("--eval_all", default=False, action='store_true')

    args = parser.parse_args()
    model_name = 'model_best_gran[%s]_scale[%s]_seed[%s].bin'% (str(args.train_size_window), str(args.train_scale), args.random_seed)
    # model_name = 'model_best_TG[%s]_IG[%s]_scale[%s]_seed[%s].bin'% (str(args.train_size_window), str(args.test_size_window), str(args.train_scale), args.random_seed)
    if args.dataset == 'sim-R':
        data_root = 'data/M2M/sim-R'
        args.train_data_path = os.path.join(data_root, 'train.json')
        args.dev_data_path = os.path.join(data_root, 'dev.json')
        args.test_data_path = os.path.join(data_root, 'test.json')
        args.model_ckpt_path = 'outputs/SOMDST/sim-R_outputs/' + model_name
    elif args.dataset == 'sim-M':
        data_root = 'data/M2M/sim-M'
        args.train_data_path = os.path.join(data_root, 'train.json')
        args.dev_data_path = os.path.join(data_root, 'dev.json')
        args.test_data_path = os.path.join(data_root, 'test.json')
        args.model_ckpt_path = 'outputs/SOMDST/sim-M_outputs/' + model_name
    elif args.dataset == 'DSTC2':
        data_root = 'data/DSTC2'
        args.train_data_path = os.path.join(data_root, 'train.json')
        args.dev_data_path = os.path.join(data_root, 'dev.json')
        args.test_data_path = os.path.join(data_root, 'test.json')
        args.model_ckpt_path = 'outputs/SOMDST/DSTC2_outputs/' + model_name
    elif args.dataset == 'WOZ2.0':
        data_root = 'data/WOZ2.0'
        args.train_data_path = os.path.join(data_root, 'woz_train_en.json')
        args.dev_data_path = os.path.join(data_root, 'woz_validate_en.json')
        args.test_data_path = os.path.join(data_root, 'woz_test_en.json')
        args.model_ckpt_path = 'outputs/SOMDST/WOZ_outputs/' + model_name
    elif args.dataset == 'MultiWOZ2.1':
        data_root = 'data/mwz2.1'
        args.train_data_path = os.path.join(data_root, 'train_dials.json')
        args.dev_data_path = os.path.join(data_root, 'dev_dials.json')
        args.test_data_path = os.path.join(data_root, 'test_dials.json')
        args.ontology_data_path = os.path.join(data_root, 'ontology.json')
        args.model_ckpt_path = 'outputs/SOMDST/MultiWOZ_outputs/' + model_name
    else:
        print('select dataset in sim-R, sim-M, DSTC2, WOZ2.0 and MultiWOZ2.1')
        exit()
    main(args)