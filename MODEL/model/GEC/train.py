import torch
import torch.nn as nn
import wandb
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report


def train_classifier(model, train_loader, val_loader, optimizer, device, 
                     pos_weight, epochs=100, early_stopping_patience=20, 
                     model_save_path="best_classifier.pth", project_name="gec-classification",
                     use_scheduler=True):
    
    wandb.init(project=project_name, config={
        "epochs": epochs,
        "early_stopping_patience": early_stopping_patience,
        "pos_weight": pos_weight.item() if isinstance(pos_weight, torch.Tensor) else pos_weight,
        "optimizer": optimizer.__class__.__name__,
        "learning_rate": optimizer.param_groups[0]['lr'],
        "use_scheduler": use_scheduler
    })

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )
    
    best_val_auc = 0.0
    best_model_state = model.state_dict()
    epochs_no_improve = 0
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': [],
        'train_auc': [], 'val_auc': [],
        'learning_rate': []
    }

    outer_bar = tqdm(range(epochs), desc="Training Progress")
    for epoch in outer_bar:
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []

        for batch in train_loader:
            batch = batch.to(device)
            x, edge_index, batch_mask = batch.x, batch.edge_index, batch.batch
            labels = batch.is_converter

            cond_vec = torch.stack([
                batch.patient_age,
                batch.patient_sex.float()
            ], dim=1).to(device)

            logits, _ = model(x, edge_index, cond_vec, batch_mask)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            preds = torch.sigmoid(logits).detach().cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())

        avg_train_loss = train_loss / max(1, len(train_loader))
        train_preds_binary = [1 if p > 0.5 else 0 for p in train_preds]
        train_acc = accuracy_score(train_labels, train_preds_binary)
        train_f1 = f1_score(train_labels, train_preds_binary, zero_division=0)
        train_auc = roc_auc_score(train_labels, train_preds) if len(set(train_labels)) > 1 else 0.0

        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                x, edge_index, batch_mask = batch.x, batch.edge_index, batch.batch
                labels = batch.is_converter

                cond_vec = torch.stack([
                    batch.patient_age,
                    batch.patient_sex.float()
                ], dim=1).to(device)

                logits, _ = model(x, edge_index, cond_vec, batch_mask)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                preds = torch.sigmoid(logits).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / max(1, len(val_loader))
        val_preds_binary = [1 if p > 0.5 else 0 for p in val_preds]
        val_acc = accuracy_score(val_labels, val_preds_binary)
        val_f1 = f1_score(val_labels, val_preds_binary, zero_division=0)
        val_auc = roc_auc_score(val_labels, val_preds) if len(set(val_labels)) > 1 else 0.0

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])

        outer_bar.set_postfix({
            'Train Loss': f"{avg_train_loss:.4f}",
            'Val Loss': f"{avg_val_loss:.4f}",
            'Val AUC': f"{val_auc:.4f}",
            'LR': f"{optimizer.param_groups[0]['lr']:.2e}"
        })

        wandb.log({
            "train_loss": avg_train_loss, "val_loss": avg_val_loss,
            "train_acc": train_acc, "val_acc": val_acc,
            "train_f1": train_f1, "val_f1": val_f1,
            "train_auc": train_auc, "val_auc": val_auc,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        if use_scheduler:
            scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict()
            epochs_no_improve = 0
            if model_save_path is not None:
                torch.save({
                    'model_state_dict': best_model_state,
                    'epoch': epoch,
                    'val_auc': best_val_auc
                }, model_save_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    return best_model_state, history


def evaluate_classifier(model, test_loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            x, edge_index, batch_mask = batch.x, batch.edge_index, batch.batch
            labels = batch.is_converter

            cond_vec = torch.stack([
                batch.patient_age,
                batch.patient_sex.float()
            ], dim=1).to(device)

            logits, _ = model(x, edge_index, cond_vec, batch_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = [1 if p > 0.5 else 0 for p in probs]

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    results = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0,
        'confusion_matrix': confusion_matrix(all_labels, all_preds),
        'classification_report': classification_report(all_labels, all_preds, zero_division=0),
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels
    }
    
    return results
