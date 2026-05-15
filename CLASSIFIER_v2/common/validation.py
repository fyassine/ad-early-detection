import torch
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve
from copy import deepcopy
import wandb

from torch_geometric.loader import DataLoader
from CLASSIFIER.model.GEC.CostWeightedGEC.train import train_classifier
from common.utils import load_frozen_encoder_from_gaae, compute_class_weights, compute_class_cost_weights


def run_kfold_cv(
    model_class,
    model_kwargs,
    combined_dataset,
    cv_indices,
    cv_labels,
    cv_patient_ids,
    index_to_patient,
    all_labels,
    config,
    device
):
    N_FOLDS = config.get('N_FOLDS', 5)
    RANDOM_STATE = config.get('RANDOM_STATE', 42)
    BATCH_SIZE = config.get('BATCH_SIZE', 16)
    USE_CLASS_COST_WEIGHTS = config.get('USE_CLASS_COST_WEIGHTS', True)
    NORMALIZE_CLASS_COST_WEIGHTS = config.get('NORMALIZE_CLASS_COST_WEIGHTS', True)
    GAAE_CHECKPOINT_PATH = config.get('GAAE_CHECKPOINT_PATH', None)
    FREEZE_ENCODER = config.get('FREEZE_ENCODER', False)
    LEARNING_RATE = config.get('LEARNING_RATE', 0.001)
    EPOCHS = config.get('EPOCHS', 25)
    EARLY_STOPPING_PATIENCE = config.get('EARLY_STOPPING_PATIENCE', 30)
    WANDB_PROJECT = config.get('WANDB_PROJECT', None)
    USE_SCHEDULER = config.get('USE_SCHEDULER', True)

    cv_results = {
        'fold': [], 'val_auc': [], 'val_sensitivity': [], 'val_specificity': [], 
        'val_f1': [], 'best_threshold': [], 'best_epoch': [], 'pos_weight': [], 
        'class_cost_weights': []
    }
    cv_histories = {'train_loss': [], 'val_loss': []}

    best_model_state = None
    best_val_auc = 0.0
    best_fold = -1
    best_threshold_overall = 0.5
    oof_preds, oof_targets = [], []

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    print(f'Starting {N_FOLDS}-fold subject-level stratified cross-validation...')
    print('========================')

    for fold, (train_idx_in_cv, val_idx_in_cv) in enumerate(sgkf.split(cv_indices, cv_labels, groups=cv_patient_ids)):
        print('========================')
        print(f'FOLD {fold + 1}/{N_FOLDS}')
        print('========================')

        train_idx = [cv_indices[i] for i in train_idx_in_cv]
        val_idx = [cv_indices[i] for i in val_idx_in_cv]
        
        train_labels_fold = [all_labels[i] for i in train_idx]
        val_labels_fold = [all_labels[i] for i in val_idx]

        train_subjects_fold = {str(getattr(combined_dataset[i], 'patient_id', '')) for i in train_idx}
        val_subjects_fold = {str(getattr(combined_dataset[i], 'patient_id', '')) for i in val_idx}
        train_subjects_fold.discard('')
        val_subjects_fold.discard('')

        train_converter_subjects_fold = {str(getattr(combined_dataset[i], 'patient_id', '')) for i in train_idx if int(all_labels[i]) == 1}
        val_converter_subjects_fold = {str(getattr(combined_dataset[i], 'patient_id', '')) for i in val_idx if int(all_labels[i]) == 1}
        train_converter_subjects_fold.discard('')
        val_converter_subjects_fold.discard('')

        train_converter_scans = int(sum(train_labels_fold))
        val_converter_scans = int(sum(val_labels_fold))
        train_converter_scan_rate = (train_converter_scans / len(train_labels_fold) * 100) if train_labels_fold else 0.0
        val_converter_scan_rate = (val_converter_scans / len(val_labels_fold) * 100) if val_labels_fold else 0.0

        print(
            f'Train: scans={len(train_idx)}, subjects={len(train_subjects_fold)}, '
            f'converter_scans={train_converter_scans} ({train_converter_scan_rate:.1f}%), '
            f'converter_subjects={len(train_converter_subjects_fold)}'
        )
        print(
            f'Val: scans={len(val_idx)}, subjects={len(val_subjects_fold)}, '
            f'converter_scans={val_converter_scans} ({val_converter_scan_rate:.1f}%), '
            f'converter_subjects={len(val_converter_subjects_fold)}'
        )

        train_dataset = torch.utils.data.Subset(combined_dataset, train_idx)
        val_dataset = torch.utils.data.Subset(combined_dataset, val_idx)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  worker_init_fn=lambda wid: np.random.seed(RANDOM_STATE + wid))
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        pos_weight = compute_class_weights(train_labels_fold, device=device)
        if USE_CLASS_COST_WEIGHTS:
            class_cost_weights = compute_class_cost_weights(
                train_labels_fold, device=device, normalize=NORMALIZE_CLASS_COST_WEIGHTS
            )
        else:
            class_cost_weights = None

        print(f'  pos_weight: {float(pos_weight):.6f}')
        print(f'  class_cost_weights: {class_cost_weights.tolist() if class_cost_weights is not None else None}')

        model = model_class(**model_kwargs).to(device)

        if GAAE_CHECKPOINT_PATH is not None:
            model = load_frozen_encoder_from_gaae(model, GAAE_CHECKPOINT_PATH, device=device)

        if not FREEZE_ENCODER:
            model.unfreeze_encoder()

        optimizer = torch.optim.Adam(model.get_trainable_params(), lr=LEARNING_RATE)

        fold_model_state, history = train_classifier(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            pos_weight=pos_weight,
            class_cost_weights=class_cost_weights,
            epochs=EPOCHS,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            model_save_path=None,
            project_name=WANDB_PROJECT,
            use_scheduler=USE_SCHEDULER,
        )

        model.load_state_dict(fold_model_state)
        model.eval()

        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                cond_vec = torch.stack([batch.patient_age, batch.patient_sex.float()], dim=1).to(device)
                output, _ = model(batch.x, batch.edge_index, cond_vec, batch.batch)
                probs = torch.sigmoid(output).cpu().numpy()
                all_preds.extend(probs.flatten())
                all_targets.extend(batch.is_converter.cpu().numpy().flatten())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        oof_preds.extend(all_preds.tolist())
        oof_targets.extend(all_targets.tolist())

        fpr, tpr, thresholds = roc_curve(all_targets, all_preds)
        fold_auc = auc(fpr, tpr)
        j_scores = tpr - fpr
        best_threshold_idx = np.argmax(j_scores)
        fold_threshold = thresholds[best_threshold_idx]

        binary_preds = (all_preds >= fold_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(all_targets, binary_preds).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

        cv_results['fold'].append(fold + 1)
        cv_results['val_auc'].append(fold_auc)
        cv_results['val_sensitivity'].append(sensitivity)
        cv_results['val_specificity'].append(specificity)
        cv_results['val_f1'].append(f1)
        cv_results['best_threshold'].append(float(fold_threshold))
        cv_results['best_epoch'].append(len(history['train_loss']))
        cv_results['pos_weight'].append(float(pos_weight))
        cv_results['class_cost_weights'].append(class_cost_weights.detach().cpu().tolist() if class_cost_weights is not None else None)
        cv_histories['train_loss'].append(list(history.get('train_loss', [])))
        cv_histories['val_loss'].append(list(history.get('val_loss', [])))

        print(f'\nFold {fold+1} Results:')
        print(f'  AUC: {fold_auc:.4f}')
        print(f'  Sensitivity: {sensitivity:.4f}')
        print(f'  Specificity: {specificity:.4f}')
        print(f'  F1: {f1:.4f}')
        print(f'  Fold threshold (Youden): {fold_threshold:.4f}')

        if fold_auc > best_val_auc:
            best_val_auc = fold_auc
            best_model_state = deepcopy(fold_model_state)
            best_fold = fold + 1

        try:
            wandb.finish()
        except Exception:
            pass

    oof_preds_arr = np.array(oof_preds)
    oof_targets_arr = np.array(oof_targets)
    fpr_oof, tpr_oof, thresholds_oof = roc_curve(oof_targets_arr, oof_preds_arr)
    j_oof = tpr_oof - fpr_oof
    best_threshold_overall = float(thresholds_oof[np.argmax(j_oof)])
    print(f'\nOOF Youden threshold: {best_threshold_overall:.4f}')

    # F1-optimal threshold on OOF predictions
    precisions, recalls, pr_thresholds = precision_recall_curve(oof_targets_arr, oof_preds_arr)
    f1_scores_oof = np.where(
        (precisions[:-1] + recalls[:-1]) > 0,
        2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1]),
        0.0,
    )
    best_f1_threshold = float(pr_thresholds[np.argmax(f1_scores_oof)])
    best_f1_value = float(np.max(f1_scores_oof))
    print(f'OOF F1-optimal threshold: {best_f1_threshold:.4f}  (OOF F1={best_f1_value:.4f})')

    print('\nOOF metrics comparison:')
    for label, thr in [('Youden', best_threshold_overall), ('F1-optimal', best_f1_threshold)]:
        preds_bin = (oof_preds_arr >= thr).astype(int)
        tn_o, fp_o, fn_o, tp_o = confusion_matrix(oof_targets_arr, preds_bin).ravel()
        sens_o = tp_o / (tp_o + fn_o) if (tp_o + fn_o) > 0 else 0
        spec_o = tn_o / (tn_o + fp_o) if (tn_o + fp_o) > 0 else 0
        f1_o = 2 * tp_o / (2 * tp_o + fp_o + fn_o) if (2 * tp_o + fp_o + fn_o) > 0 else 0
        print(f'  [{label:10s}] thr={thr:.4f} | sens={sens_o:.3f} | spec={spec_o:.3f} | F1={f1_o:.3f}')

    if best_model_state is not None:
        best_model = model_class(**model_kwargs).to(device)
        if GAAE_CHECKPOINT_PATH is not None:
            best_model = load_frozen_encoder_from_gaae(best_model, GAAE_CHECKPOINT_PATH, device=device)
        if not FREEZE_ENCODER:
            best_model.unfreeze_encoder()
        best_model.load_state_dict(best_model_state)
        best_model.eval()
    else:
        best_model = None

    print('========================')
    print('CROSS-VALIDATION COMPLETE')
    print('========================')

    return cv_results, cv_histories, best_model_state, best_model, best_val_auc, best_fold, best_threshold_overall, best_f1_threshold, oof_preds, oof_targets
