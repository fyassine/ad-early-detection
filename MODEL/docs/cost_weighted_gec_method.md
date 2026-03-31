================================================================================
Cost-Weighted GEC for DELCODE Whole-Brain
Method and Implementation Summary
================================================================================


GOAL
────
This document summarizes the cost-weighted objective used for converter 
classification in the GEC pipeline and how it is applied in the whole-brain 
cross-validation training notebook.


TASK DEFINITION
───────────────
The model performs binary graph-level classification:
  • Class 0: non-converter
  • Class 1: converter

For each graph sample i, the model outputs a logit z_i ∈ ℝ and label y_i ∈ {0,1}.


BASE BCEWITHLOGITS LOSS
───────────────────────
The unreduced per-sample binary cross-entropy with positive weighting is:

    ℓ_i = -(α·y_i·log(σ(z_i)) + (1-y_i)·log(1-σ(z_i)))

where σ(·) is the sigmoid function and

    α = pos_weight = N_(-) / N_(+)

N_(+) and N_(-) are the positive (converter) and negative (non-converter) 
sample counts in the current training fold.


CLASS COST WEIGHTS
──────────────────
A second class-balancing term is computed from fold label counts:

    w_0 = N / (2·N_(-))
    w_1 = N / (2·N_(+))

where N = N_(+) + N_(-). In the implementation, these are optionally 
normalized by their mean:

    w̃_c = w_c / ((w_0 + w_1) / 2),    c ∈ {0,1}

Per-sample cost weight is then assigned by class:

    ω_i = { w̃_1,  if y_i = 1
          { w̃_0,  if y_i = 0


WEIGHTED COST AVERAGING
───────────────────────
The final training loss is the weighted average of unreduced BCE losses:

    L_CW = (Σ(i=1 to B) ω_i·ℓ_i) / (Σ(i=1 to B) ω_i + ε)

where B is batch size and ε is a small constant for numerical stability.


HOW IT WAS APPLIED IN THIS PROJECT
───────────────────────────────────
In the cost-weighted whole-brain notebook, each CV fold does the following:

1. Build train/validation subsets from subject-indexed scan graphs.

2. Compute pos_weight from fold labels.

3. Compute class_cost_weights from fold labels.

4. Train with:
   - unreduced BCEWithLogits (with pos_weight), then
   - weighted cost averaging using class_cost_weights.

5. Evaluate fold AUC, sensitivity, specificity, and F1 with threshold 
   selected by Youden's J statistic:

       J = TPR - FPR


INTERPRETATION
──────────────
• If converters are minority in a fold, w̃_1 > w̃_0 and converter errors 
  contribute more.

• If non-converters are minority, w̃_0 > w̃_1.

• Using both pos_weight and class-cost averaging applies balancing at two 
  levels:
  - inside BCE positive term scaling
  - outside BCE as weighted batch averaging


IMPLEMENTATION LOCATIONS
────────────────────────
• Loss definition and weighted averaging: 
  MODEL/model/CostWeightedGEC/train.py

• Weight computation: 
  MODEL/model/CostWeightedGEC/utils.py

• Fold-wise use in training: 
  MODEL/notebooks/COST_WEIGHTED_GEC_DELCODE_WHOLE_BRAIN.ipynb

================================================================================
