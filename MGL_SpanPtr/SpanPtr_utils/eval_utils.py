"""
Most codes are from https://github.com/jasonwu0731/trade-dst
"""
import numpy as np

def compute_acc(gold, pred, slot_temp):
    miss_gold = 0
    miss_slot = []
    for g in gold:
        if g not in pred:
            miss_gold += 1
            miss_slot.append(g.rsplit("-", 1)[0])
    wrong_pred = 0
    for p in pred:
        if p not in gold and p.rsplit("-", 1)[0] not in miss_slot:
            wrong_pred += 1
    ACC_TOTAL = len(slot_temp)
    ACC = len(slot_temp) - miss_gold - wrong_pred
    ACC = ACC / float(ACC_TOTAL)
    return ACC


def compute_prf(gold, pred):
    TP, FP, FN = 0, 0, 0
    if len(gold) != 0:
        count = 1
        for g in gold:
            if g in pred:
                TP += 1
            else:
                FN += 1
        for p in pred:
            if p not in gold:
                FP += 1
        precision = TP / float(TP+FP) if (TP+FP) != 0 else 0
        recall = TP / float(TP+FN) if (TP+FN) != 0 else 0
        F1 = 2 * precision * recall / float(precision + recall) if (precision+recall) != 0 else 0
    else:
        if len(pred) == 0:
            precision, recall, F1, count = 1, 1, 1, 1
        else:
            precision, recall, F1, count = 0, 0, 0, 1
    return F1, recall, precision, count

def compute_goal(result, slot_meta):
    goal_correctness = 1.0
    pred_op = []
    gold_op = []
    pred_value = []
    gold_value = []
    for r in result:
        pred_op.append(result[r][0])
        gold_op.append(result[r][2])
        pred_value.append([result[r][1].get(s) for s in slot_meta])
        gold_value.append([result[r][3].get(s) for s in slot_meta])
    # print(pred_op)
    # print(gold_op)
    # print(pred_value)
    # print(gold_value)
    # exit()
    
    for id, slot in enumerate(slot_meta):
      slot_pred_op = [o[id] for o in pred_op]
      slot_gold_op = [o[id] for o in gold_op]
      slot_pred_value = [o[id] for o in pred_value]
      slot_gold_value = [o[id] for o in gold_value]
      tot_cor, cls_cor, value_cor = get_joint_slot_correctness(slot_pred_op, slot_gold_op, slot_pred_value, slot_gold_value)
      print('%s: joint slot acc: %g, class acc: %g, value acc: %g' % (slot, np.mean(tot_cor), np.mean(cls_cor), np.mean(value_cor)))
      goal_correctness *= tot_cor
    acc = np.mean(goal_correctness)
    print('Joint goal acc: %g' % (acc))


def get_joint_slot_correctness(slot_pred_op, slot_gold_op, slot_pred_value, slot_gold_value):
    class_correctness = []
    value_correctness = []
    total_correctness = []
    for joint_pd_class, joint_gt_class, joint_pd_value, joint_gt_value in zip(slot_pred_op, slot_gold_op, slot_pred_value, slot_gold_value):
      total_correct = True
      if joint_gt_class == joint_pd_class:
        class_correctness.append(1.0)
        if joint_gt_class == 'other':
          if joint_pd_value == joint_gt_value:
            value_correctness.append(1.0)
          else:
            value_correctness.append(0.0)
            total_correct = False
      else:
        class_correctness.append(0.0)
        total_correct = False
      if total_correct:
        total_correctness.append(1.0)
      else:
        total_correctness.append(0.0)

    return np.asarray(total_correctness), np.asarray(class_correctness), np.asarray(value_correctness)