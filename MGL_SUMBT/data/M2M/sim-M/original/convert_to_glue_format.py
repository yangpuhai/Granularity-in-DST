import json

source_files = ["train.json", "dev.json", "test.json"]
target_files = ["../train.tsv", "../dev.tsv", "../test.tsv"]

target_slots = ["num_tickets", "date", "theatre_name", "movie", "time"]

fp_ont = open("ontology.json", "r")
ontology = json.load(fp_ont)
for slot in ontology.keys():
    ontology[slot].append("dontcare")
fp_ont.close()

for idx, src in enumerate(source_files):
    trg = target_files[idx]

    fp_src = open(src, "r")
    fp_trg = open(trg, "w")

    data = json.load(fp_src)

    for dialogue in data:
        dialogue_idx = dialogue["dialogue_id"]
        for ti, turn in enumerate(dialogue["turns"]):
            turn_idx = ti
            user_utterance = turn["user_utterance"]["text"]
            if "system_utterance" not in turn:
                system_response = ""
            else:  
                system_response = turn["system_utterance"]["text"]
            belief_state = turn["dialogue_state"]

            # initialize turn label and belief state to "none"
            belief_st = {}
            for ts in target_slots:
                belief_st[ts] = "none"

            # extract slot values in belief state
            for slots in belief_state:
                slot_name = slots["slot"]
                slot_value = slots["value"]
                if slot_name in belief_st:
                    assert(belief_st[slot_name] == "none" or belief_st[slot_name] == slot_value)
                    assert(slot_value in ontology[slot_name])
                    belief_st[slot_name] = slot_value

            fp_trg.write(str(dialogue_idx))                 # 0: dialogue index
            fp_trg.write("\t" + str(turn_idx))              # 1: turn index
            fp_trg.write("\t" + str(user_utterance.replace("\t", " ")))        # 2: user utterance
            fp_trg.write("\t" + str(system_response.replace("\t", " ")))       # 3: system response

            for slot in sorted(belief_st.keys()):
                fp_trg.write("\t" + str(belief_st[slot]))

            fp_trg.write("\n")
            fp_trg.flush()

    fp_src.close()
    fp_trg.close()
