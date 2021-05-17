# -*- coding: utf-8 -*-
import copy
import json

def createJson(file_name_list, file_dir, data_dir, dataset):
    data=[]
    name_list = [n.strip() for n in open(file_name_list).readlines()]
    file_names = [[file_dir + "/" + n + "/" + "log.json", file_dir + "/" + n + "/" + "label.json"] for n in name_list]
    for [file_name_log, file_name_label] in file_names:
        data_dialogue={}
        f_log = open(file_name_log)
        f_label = open(file_name_label)
        log_dials = json.load(f_log)
        label_dials = json.load(f_label)
        data_dialogue["caller_id"] = label_dials["caller-id"]
        data_dialogue["domains"] = ["restaurant"]
        # Reading data
        turns=[]
        for log_turn, label_turn in zip(log_dials["turns"], label_dials["turns"]):
            turn={}
            turn["turn_id"] = label_turn["turn-index"]
            turn["turn_domain"] = "restaurant"
            turn["system_transcript"] = log_turn["output"]["transcript"]
            turn["system_acts"] = log_turn["output"]["dialog-acts"]
            turn["user_transcript"] = label_turn["transcription"]
            turn["user_asr"] = log_turn["input"]["live"]["asr-hyps"][0]["asr-hyp"]
            turn["state"] = label_turn["goal-labels"]
            turn["turn_labels"] = {}
            for s in label_turn["semantics"]["json"]:
                if s["act"] == "inform":
                    slot_name = s["slots"][0][0]
                    slot_value = s["slots"][0][1]
                    turn["turn_labels"][slot_name] = slot_value
            turns.append(turn)
        data_dialogue["turns"]=turns
        data.append(data_dialogue)
    save_json = data_dir + "/" + dataset + ".json"
    with open(save_json, 'w') as f:
        json.dump(data, f, indent=4)

def createData():
    file_train_list = 'data/DSTC2/dstc2_traindev/scripts/config/dstc2_train.flist'
    file_dev_list = 'data/DSTC2/dstc2_traindev/scripts/config/dstc2_dev.flist'
    traindev_dir = 'data/DSTC2/dstc2_traindev/data'
    file_test_list = 'data/DSTC2/dstc2_test/scripts/config/dstc2_test.flist'
    test_dir = 'data/DSTC2/dstc2_test/data'
    data_dir='data/DSTC2'
    createJson(file_train_list, traindev_dir, data_dir, "train")
    createJson(file_dev_list, traindev_dir, data_dir, "dev")
    createJson(file_test_list, test_dir, data_dir, "test")

def main():
    print('Create  dialogues. Get yourself a coffee, this might take a while.')
    createData()

if __name__ == "__main__":
    main()



