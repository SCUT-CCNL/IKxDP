This is the code for the paper "IKxDP: Implicit Knowledge Enhanced Explainable Disease Prediction". We primarily utilized two datasets, namely MIMIC III and MIMIC IV, with nearly identical processing methods. Here, we demonstrate how to conduct experiments using MIMIC III.

## Datasets

- **MIMIC III** can be downloaded from: [PhysioNet, MIMIC III ](https://physionet.org/content/mimiciii/1.4/)
- **MIMIC IV** can be accessed at: [PhysioNet, MIMIC IV](https://physionet.org/content/mimiciv/2.2/)

The code of data processing can refer to [RETAIN](https://github.com/mp2893/retain).

## Prerequisites

This code is based on the [KGxDP](https://github.com/SCUT-CCNL/KGxDP ) codebase. To successfully run this code:

1. For fine-tuning the model on the "primary diagnosis" prediction task, execute:
   ```bash
   python pretrain.py
   ```

2. To train and evaluate the model, run:
   ```bash
   python train.py
   ```


## Detailed 

- `train.py`: This script is used for training and validating the models. It utilizes default parameter values but also allows for runtime configuration adjustments.
- `metrics.py`: Contains all the evaluation metrics used in the paper, with references to [Chet](https://github.com/luchang-cs/chet).
- The `modeling/` directory contains the model implementations and the Sequence Graph construction and embedding.
- The `util/` directory includes utility functions, such as the data loading.
-  



#  Acknowledge 

In our work, part of the code is referenced from the following open-source code: 

1. RETAIN: An Interpretable Predictive Model for Healthcare using Reverse Time Attention Mechanism. https://github.com/mp2893/retain 
   
2. Chet: Context-aware Health Event Prediction via Transition Functions on Dynamic Disease Graphs. https://github.com/luchang-cs/chet
   
3. KGxDP：Interpretable Disease Prediction via Path Reasoning over Medical Knowledge Graphs and Admission History. https://github.com/SCUT-CCNL/KGxDP 

Many thanks to the authors and developers!
