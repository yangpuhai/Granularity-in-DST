# Granularity-in-DST

This code is the official pytorch implementation of ACL2021 paper: **Comprehensive Study: How the Context Information of Different Granularity Affects Dialogue State Tracking? Puhai Yang, Heyan Huang, Xian-Ling Mao. ACL2021 *(Long paper)***  [[arXiv](https://arxiv.org/abs/2105.03571)]

## Abstract
Dialogue state tracking (DST) plays a key role in task-oriented dialogue systems to monitor the user's goal. In general, there are two strategies to track a dialogue state: predicting it from scratch and updating it from previous state. The scratch-based strategy obtains each slot value by inquiring all the dialogue history, and the previous-based strategy relies on the current turn dialogue to update the previous dialogue state. However, it is hard for the scratch-based strategy to correctly track short-dependency dialogue state because of noise; meanwhile, the previous-based strategy is not very useful for long-dependency dialogue state tracking. Obviously, it plays different roles for the context information of different granularity to track different kinds of dialogue states. Thus, in this paper, we will study and discuss how the context information of different granularity affects dialogue state tracking. First, we explore how greatly different granularities affect dialogue state tracking. Then, we further discuss how to combine multiple granularities for dialogue state tracking. Finally, we apply the findings about context granularity to few-shot learning scenario. Besides, we have publicly released all codes.

## Requirements
* python 3.6
* pytorch >= 1.0