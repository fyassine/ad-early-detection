import json
import re

with open('COST_WEIGHTED_GEC_DELCODE_WHOLE_BRAIN.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    
    source = "".join(cell['source'])
    
    # Update imports
    if "from model.CostWeightedGEC.dataset import ClassificationDataset" in source:
        # replace just the first line, leave the rest intact
        source = source.replace(
            "from model.CostWeightedGEC.dataset import ClassificationDataset, CombinedClassificationDataset", 
            "from model.common.dataset import ClassificationDataset, CombinedClassificationDataset"
        )
        source = source.replace(
            "from model.CostWeightedGEC.utils import (",
            "from model.common.utils import ("
        )
        # We need to add the import for run_kfold_cv
        # Just append it to the cell
        source += "\nfrom model.common.validation import run_kfold_cv\n"
        cell['source'] = [line + "\n" for line in source.split("\n")]
        # Fix the messy trailing newlines
        cell['source'] = [line.replace("\n\n", "\n") for line in cell['source']]
        continue

    # Update the CV block
    if "cv_results = {" in source and "cv_histories =" in source and "StratifiedGroupKFold" in source:
        new_source = """
cv_results = {
    'fold': [], 'val_auc': [], 'val_sensitivity': [], 'val_specificity': [], 
    'val_f1': [], 'best_threshold': [], 'best_epoch': [], 'pos_weight': [], 'class_cost_weights': []
}
cv_histories = {'train_loss': [], 'val_loss': []}

best_model_state = None
best_val_auc = 0.0
best_fold = -1
best_threshold_overall = 0.5
oof_preds = []
oof_targets = []

cv_patient_ids = [index_to_patient[i] for i in cv_indices]

if USE_CW_GEC_CHECKPOINT:
    if CW_GEC_CHECKPOINT_PATH is None:
        raise RuntimeError('Cost-Weighted GEC checkpoint mode is enabled but no checkpoint path was selected.')

    checkpoint_obj = torch.load(CW_GEC_CHECKPOINT_PATH, map_location=device)

    model = GraphEncoderClassifierAttention(
        in_features=IN_FEATURES,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        cond_dim=COND_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        classifier_hidden=CLASSIFIER_HIDDEN,
    ).to(device)

    model = load_frozen_encoder_from_gaae(model, GAAE_CHECKPOINT_PATH, device=device)
    if not FREEZE_ENCODER:
        model.unfreeze_encoder()

    if isinstance(checkpoint_obj, torch.nn.Module):
        model = checkpoint_obj.to(device)
        best_model_state = model.state_dict()
    elif isinstance(checkpoint_obj, dict):
        state_dict = checkpoint_obj.get('state_dict', checkpoint_obj.get('model_state_dict', checkpoint_obj))
        model.load_state_dict(state_dict)
        best_model_state = state_dict

    model.eval()
    best_model = model

    run_summary_path = Path(CW_GEC_SELECTED_RUN_DIR) / 'run_summary.json'
    cv_results_path = Path(CW_GEC_SELECTED_RUN_DIR) / 'cv_results.json'

    if run_summary_path.exists():
        with open(run_summary_path, 'r') as f:
            run_summary = json.load(f)

        loaded_cv_results = run_summary.get('cv_results', {})
        if isinstance(loaded_cv_results, dict):
            for key in cv_results.keys():
                values = loaded_cv_results.get(key)
                if isinstance(values, list):
                    cv_results[key] = values

        loaded_cv_histories = run_summary.get('cv_histories', {})
        if isinstance(loaded_cv_histories, dict):
            cv_histories['train_loss'] = loaded_cv_histories.get('train_loss', [])
            cv_histories['val_loss'] = loaded_cv_histories.get('val_loss', [])

        best_fold = int(run_summary.get('best_fold', best_fold))
        best_val_auc = float(run_summary.get('best_val_auc', best_val_auc))
        best_threshold_overall = float(run_summary.get('best_threshold', best_threshold_overall))
    elif cv_results_path.exists():
        with open(cv_results_path, 'r') as f:
            loaded_cv_results = json.load(f)
        if isinstance(loaded_cv_results, dict):
            for key in cv_results.keys():
                values = loaded_cv_results.get(key)
                if isinstance(values, list):
                    cv_results[key] = values

        if len(cv_results['val_auc']) > 0:
            val_auc_arr = np.array(cv_results['val_auc'], dtype=float)
            best_idx = int(np.argmax(val_auc_arr))
            best_val_auc = float(val_auc_arr[best_idx])
            best_fold = int(cv_results['fold'][best_idx]) if len(cv_results['fold']) > best_idx else best_fold

    print(f'Loaded Cost-Weighted GEC checkpoint from {CW_GEC_CHECKPOINT_PATH}')
    print('Cross-validation skipped because checkpoint mode is enabled')
else:
    config = {
        'N_FOLDS': N_FOLDS,
        'RANDOM_STATE': RANDOM_STATE,
        'BATCH_SIZE': BATCH_SIZE,
        'USE_CLASS_COST_WEIGHTS': USE_CLASS_COST_WEIGHTS,
        'NORMALIZE_CLASS_COST_WEIGHTS': NORMALIZE_CLASS_COST_WEIGHTS,
        'GAAE_CHECKPOINT_PATH': GAAE_CHECKPOINT_PATH,
        'FREEZE_ENCODER': FREEZE_ENCODER,
        'LEARNING_RATE': LEARNING_RATE,
        'EPOCHS': EPOCHS,
        'EARLY_STOPPING_PATIENCE': EARLY_STOPPING_PATIENCE,
        'WANDB_PROJECT': WANDB_PROJECT,
        'USE_SCHEDULER': USE_SCHEDULER,
    }

    model_kwargs = {
        'in_features': IN_FEATURES,
        'hidden_dim': HIDDEN_DIM,
        'latent_dim': LATENT_DIM,
        'cond_dim': COND_DIM,
        'num_heads': NUM_HEADS,
        'dropout': DROPOUT,
        'classifier_hidden': CLASSIFIER_HIDDEN,
    }

    cv_results, cv_histories, best_model_state, best_model, best_val_auc, best_fold, best_threshold_overall, oof_preds, oof_targets = run_kfold_cv(
        model_class=GraphEncoderClassifierAttention,
        model_kwargs=model_kwargs,
        combined_dataset=combined_dataset,
        cv_indices=cv_indices,
        cv_labels=cv_labels,
        cv_patient_ids=cv_patient_ids,
        index_to_patient=index_to_patient,
        all_labels=all_labels,
        config=config,
        device=device
    )
"""
        cell['source'] = [line + "\n" for line in new_source.split("\n")]
        cell['source'][-1] = cell['source'][-1].rstrip("\n") # Remove trailing newline from last line

with open('COST_WEIGHTED_GEC_DELCODE_WHOLE_BRAIN_REFACTORED.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
