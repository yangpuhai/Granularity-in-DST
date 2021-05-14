import json

source_files = ["train.json", "dev.json", "test.json"]
target_files = ["../train.tsv", "../dev.tsv", "../test.tsv"]

target_slots = ["area", "food", "pricerange"]

fp_ont = open("ontology_dstc2.json", "r")
ontology = json.load(fp_ont)
ontology = ontology["informable"]
for slot in ontology.keys():
    ontology[slot].append("dontcare")
fp_ont.close()

for idx, src in enumerate(source_files):
    trg = target_files[idx]

    fp_src = open(src, "r")
    fp_trg = open(trg, "w")

    data = json.load(fp_src)

    for dialogue_idx, dialogue in enumerate(data):
        # dialogue_idx = dialogue["caller_id"]
        for turn in dialogue["turns"]:
            turn_idx = turn["turn_id"]
            if 'train' in source_files:
                user_utterance = turn["user_transcript"]
            else:
                user_utterance = turn["user_asr"]
            system_response = turn["system_transcript"]
            belief_state = turn["state"]

            # initialize turn label and belief state to "none"
            belief_st = {}
            for ts in target_slots:
                belief_st[ts] = "none"

            # extract slot values in belief state
            for slot in belief_state:
                value = belief_state[slot]
                if slot in belief_st:
                    assert(belief_st[slot] == "none" or belief_st[slot] == value)
                    assert(value in ontology[slot])
                    belief_st[slot] = value

            fp_trg.write(str(dialogue_idx))                 # 0: dialogue index
            fp_trg.write("\t" + str(turn_idx))              # 1: turn index
            fp_trg.write("\t" + str(user_utterance.replace("\t", " ")))        # 2: user utterance
            fp_trg.write("\t" + str(system_response.replace("\t", " ")))       # 3: system response

            for slot in sorted(belief_st.keys()):
                fp_trg.write("\t" + str(belief_st[slot]))   # 4-6: belief state

            fp_trg.write("\n")
            fp_trg.flush()

    fp_src.close()
    fp_trg.close()
