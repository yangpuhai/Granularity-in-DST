import json
import os

source_files = ["train.json", "dev.json", "test.json"]
target_file = "ontology.json"
ontology = {}

for idx, src in enumerate(source_files):
    fp_src = open(src, "r")
    data = json.load(fp_src)
    for dialogue in data:
        for turn in dialogue["turns"]:
            belief_state = turn["dialogue_state"]
            for slot in belief_state:
                slot_name = slot["slot"]
                slot_value = slot["value"]
                if slot_name not in ontology:
                    ontology[slot_name] = []
                if slot_value not in ontology[slot_name] and slot_value != 'dontcare':
                    ontology[slot_name].append(slot_value)
    fp_src.close()

with open(target_file, "w", encoding='utf-8') as f:
    # json.dump(dict_var, f)
    json.dump(ontology, f, indent=2, sort_keys=True, ensure_ascii=False)