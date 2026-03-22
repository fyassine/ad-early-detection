import logging
import torch
import wandb
from tqdm.notebook import tqdm

from torch_geometric.utils import to_dense_adj
from .utils import create_mask
from .losses import total_loss_fn

def train_model_with_val_notebook_train_loss(model, train_loader, val_loader, optimizer, device, 
                batch_size, learning_rate, model_config, adj_loss_weight=1.0, 
                epochs=100, early_stopping_patience=50,
                dataset_info=None, project_name="graph-autoencoder-training"):
    
    wandb.init(project=project_name, config={
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": optimizer.__class__.__name__,
        "model_config": model_config,
        "adj_loss_weight": adj_loss_weight,
        "epochs": epochs,
        "early_stopping_patience": early_stopping_patience,
        "dataset_info": dataset_info
    })

    model.train()
    best_loss = float("inf")
    best_model = model.state_dict()
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': []}

    outer_bar = tqdm(range(epochs), desc="Training Progress")
    for epoch in outer_bar:
        total_train_loss = 0

        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            x, edge_index, edge_attr, batch_mask = batch.x, batch.edge_index, batch.edge_attr, batch.batch
            
            dense_adj = to_dense_adj(edge_index, batch=batch_mask).to(device)

            cond_vec = torch.stack([
                batch.patient_age,
                batch.patient_sex.float()
            ], dim=1).to(device)

            z, x_reconstructed, adj_reconstructed, _ = model(
                x, edge_index, edge_attr, cond_vec, batch_mask
            )

            dense_adj_reconstructed = to_dense_adj(
                edge_index, batch=batch_mask, edge_attr=adj_reconstructed
            ).to(device)

            batch_size_local = dense_adj.size(0)
            total_nodes = x.size(0)
            combined_dense_adj = torch.zeros((total_nodes, total_nodes), device=device)
            combined_dense_adj_reconstructed = torch.zeros((total_nodes, total_nodes), device=device)
            start_idx = 0
            for i in range(batch_size_local):
                num_nodes_in_graph = (batch_mask == i).sum().item()
                end_idx = start_idx + num_nodes_in_graph
                combined_dense_adj[start_idx:end_idx, start_idx:end_idx] = dense_adj[i, :num_nodes_in_graph, :num_nodes_in_graph]
                combined_dense_adj_reconstructed[start_idx:end_idx, start_idx:end_idx] = dense_adj_reconstructed[i, :num_nodes_in_graph, :num_nodes_in_graph]
                start_idx = end_idx

            mask = create_mask(batch_mask)

            loss, _, _ = total_loss_fn(
                x, x_reconstructed, combined_dense_adj, combined_dense_adj_reconstructed, mask, adj_loss_weight
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / max(1, len(train_loader))

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                x, edge_index, edge_attr, batch_mask = batch.x, batch.edge_index, batch.edge_attr, batch.batch
                dense_adj = to_dense_adj(edge_index, batch=batch_mask).to(device)

                cond_vec = torch.stack([
                    batch.patient_age,
                    batch.patient_sex.float()
                ], dim=1).to(device)

                z, x_reconstructed, adj_reconstructed, _ = model(
                    x, edge_index, edge_attr, cond_vec, batch_mask
                )

                dense_adj_reconstructed = to_dense_adj(
                    edge_index, batch=batch_mask, edge_attr=adj_reconstructed
                ).to(device)

                batch_size_local = dense_adj.size(0)
                total_nodes = x.size(0)
                combined_dense_adj = torch.zeros((total_nodes, total_nodes), device=device)
                combined_dense_adj_reconstructed = torch.zeros((total_nodes, total_nodes), device=device)
                start_idx = 0
                for i in range(batch_size_local):
                    num_nodes_in_graph = (batch_mask == i).sum().item()
                    end_idx = start_idx + num_nodes_in_graph
                    combined_dense_adj[start_idx:end_idx, start_idx:end_idx] = dense_adj[i, :num_nodes_in_graph, :num_nodes_in_graph]
                    combined_dense_adj_reconstructed[start_idx:end_idx, start_idx:end_idx] = dense_adj_reconstructed[i, :num_nodes_in_graph, :num_nodes_in_graph]
                    start_idx = end_idx

                mask = create_mask(batch_mask)

                loss, _, _ = total_loss_fn(
                    x, x_reconstructed, combined_dense_adj, combined_dense_adj_reconstructed, mask, adj_loss_weight
                )

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / max(1, len(val_loader))

        outer_bar.set_postfix({'Train Loss': f"{avg_train_loss:.6f}", 'Val Loss': f"{avg_val_loss:.6f}"})
        outer_bar.set_description(f"Epoch {epoch}")

        logging.info(f"Epoch {epoch}: Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}")
        wandb.log({"Train Loss": avg_train_loss, "Val Loss": avg_val_loss})

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            epochs_no_improve = 0
            best_model = model.state_dict()
            logging.info(f"New best model saved with training loss: {best_loss:.6f}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            logging.info("Early stopping triggered. Training stopped.")
            break

    # wandb.finish()
    return best_model, history
