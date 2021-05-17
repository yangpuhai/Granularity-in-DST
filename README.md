# Granularity-in-DST

This code is the official pytorch implementation of ACL2021 paper: **Comprehensive Study: How the Context Information of Different Granularity Affects Dialogue State Tracking? Puhai Yang, Heyan Huang, Xian-Ling Mao. ACL2021 *(Long paper)***  [[arXiv](https://arxiv.org/abs/2105.03571)]

## Abstract
Dialogue state tracking (DST) plays a key role in task-oriented dialogue systems to monitor the user's goal. In general, there are two strategies to track a dialogue state: predicting it from scratch and updating it from previous state. The scratch-based strategy obtains each slot value by inquiring all the dialogue history, and the previous-based strategy relies on the current turn dialogue to update the previous dialogue state. However, it is hard for the scratch-based strategy to correctly track short-dependency dialogue state because of noise; meanwhile, the previous-based strategy is not very useful for long-dependency dialogue state tracking. Obviously, it plays different roles for the context information of different granularity to track different kinds of dialogue states. Thus, in this paper, we will study and discuss how the context information of different granularity affects dialogue state tracking. First, we explore how greatly different granularities affect dialogue state tracking. Then, we further discuss how to combine multiple granularities for dialogue state tracking. Finally, we apply the findings about context granularity to few-shot learning scenario. Besides, we have publicly released all codes.

## Requirements
* python 3.6
* pytorch >= 1.0

## Baselines

MGL_SpanPtr, MGL_TRADE, MGL_BERTDST, MGL_SOMDST: baselines with granularity, which are reproduced based on the original papers [[SpanPtr](https://www.aclweb.org/anthology/P18-1134.pdf), [TRADE](https://www.aclweb.org/anthology/P19-1078.pdf), [BERTDST](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/1355.pdf), [SOMDST](https://www.aclweb.org/anthology/2020.acl-main.53.pdf)] and the [official pytorch implementation of SOMDST](https://github.com/clovaai/som-dst).

MGL_SUMBT: baseline with granularity, which is reproduced based on the original paper [[SUMBT](https://www.aclweb.org/anthology/P19-1546.pdf)] and the [official pytorch implementation of SUMBT](https://github.com/SKTBrain/SUMBT).

## Datasets

1. Corpus download

    Sim-M and Sim-R: [download](https://github.com/google-research-datasets/simulated-dialogue), 

    WOZ2.0: [download](https://github.com/nmrksic/neural-belief-tracker/tree/master/data/woz)

    DSTC2: [download](https://camdial.org/~mh521/dstc/)

    MultiWOZ2.1: [download](https://github.com/budzianowski/multiwoz/tree/master/data)

2. Data preprocessing

    ```
    python create_data_DSTC2.py
    python create_data_MultiWOZ.py
    ```
    MGL_SpanPtr, MGL_TRADE, MGL_BERTDST, MGL_SOMDST: unzip the dataset.zip file and copy it to the corresponding *MGL_\** folder.

    MGL_SUMBT: The processed data has been included in its *data* folder, and you can reprocess the data by yourself according to the instructions.

## Train
MGL_SpanPtr, MGL_TRADE, MGL_BERTDST, MGL_SOMDST: 

    # For example: 
    bash SOMDST_train_SG.sh  # train SOMDST with single granularity
    bash SOMDST_train.sh  # train SOMDST with Multiple granularities
    
MGL_SUMBT:

    # For example: 
    bash run-multiwoz.sh  # train SUMBT with Multiple granularities on MultiWOZ2.1

## Contact Information
Contact: Puhai Yang (`phyang@bit.edu.cn`), Heyan Huang (`hhy63@bit.edu.cn`), Xian-Ling Mao (`maoxl@bit.edu.cn`)
