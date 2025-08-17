import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics


class NCfoldLoss(nn.Module):
    """
        edge_arr: 4-class
        orient_mat: 3-class
    """
    def __init__(self, edge_weight=1.0, orient_weight=1.0, edge_weights=None, orient_weights=None):
        super().__init__()
        self.edge_weight = edge_weight  # edge_arr weight
        self.orient_weight = orient_weight  # orient_mat weight
        
        # class-weights, for imbalanced classification
        self.edge_criterion = nn.CrossEntropyLoss(weight=edge_weights)
        self.orient_criterion = nn.CrossEntropyLoss(weight=orient_weights)

    def forward(self, edge_pred, orient_pred, edge_true, orient_true):
        """
        Args:
            edge_pred: shape=(B, L, 4) -> (B, 4, L), to match CrossEntropyLoss
            orient_pred: shape=(B, 3, L, L)
            edge_true: shape=(B, L), \in {0, 1, 2, 3}
            orient_true: shape=(B, L, L), \in {0,1,2}
        """
        edge_pred_reshaped = edge_pred.transpose(1, 2)  # (B, 4, L)

        loss_edge = self.edge_criterion(edge_pred_reshaped, edge_true)
        loss_orient = self.orient_criterion(orient_pred, orient_true)
        
        total_loss = self.edge_weight * loss_edge + self.orient_weight * loss_orient
        return total_loss, loss_edge, loss_orient


def compute_metrics(edge_pred, orient_pred, edge_true, orient_true):
    """
    Returns:
        metrics_dict: F1
    """
    # predicted class: argmax
    edge_pred_class = torch.argmax(edge_pred, dim=-1)  # (B, L)
    # flatting for convenient computation of metrics
    edge_pred_flat = edge_pred_class.view(-1).cpu().numpy()
    edge_true_flat = edge_true.view(-1).cpu().numpy()
    
    edge_acc = metrics.accuracy_score(edge_true_flat, edge_pred_flat)
    edge_f1_macro = metrics.f1_score(
        edge_true_flat, edge_pred_flat, 
        average='macro', 
        labels=[0, 1, 2, 3]
    )
    edge_class_acc = {}
    for cls in range(4):
        mask = (edge_true_flat == cls)
        if mask.sum() == 0:
            edge_class_acc[f'cls_{cls}'] = 0.0
        else:
            edge_class_acc[f'cls_{cls}'] = (edge_pred_flat[mask] == cls).mean()
    
    orient_pred_class = torch.argmax(orient_pred, dim=1)  # (B, L, L)
    orient_pred_flat = orient_pred_class.view(-1).cpu().numpy()
    orient_true_flat = orient_true.view(-1).cpu().numpy()
    
    orient_acc = metrics.accuracy_score(orient_true_flat, orient_pred_flat)
    orient_f1_macro = metrics.f1_score(
        orient_true_flat, orient_pred_flat, 
        average='macro', 
        labels=[0, 1, 2]
    )
    orient_class_acc = {}
    for cls in range(3):
        mask = (orient_true_flat == cls)
        if mask.sum() == 0:
            orient_class_acc[f'cls_{cls}'] = 0.0
        else:
            orient_class_acc[f'cls_{cls}'] = (orient_pred_flat[mask] == cls).mean()
    
    return {
        'edge_acc': edge_acc,
        'edge_f1_macro': edge_f1_macro,
        **edge_class_acc, 
        'orient_acc': orient_acc,
        'orient_f1_macro': orient_f1_macro,
        ** orient_class_acc,
    }


if __name__ == "__main__":
    B, L = 2, 10
    edge_pred = torch.randn(B, L, 4)
    orient_pred = torch.randn(B, 3, L, L)
    edge_true = torch.randint(0, 4, (B, L))
    orient_true = torch.randint(0, 3, (B, L, L))
    
    edge_class_weights = torch.tensor([1.0, 1.0, 5.0, 5.0])
    orient_class_weights = torch.tensor([1.0, 1.0, 1.0])
    criterion = NCfoldLoss(
        edge_weight=1.0, 
        orient_weight=1.0,
        edge_weights=edge_class_weights,
        orient_weights=orient_class_weights
    )
    
    total_loss, loss_edge, loss_orient = criterion(edge_pred, orient_pred, edge_true, orient_true)
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Edge loss: {loss_edge.item():.4f}, Orient loss: {loss_orient.item():.4f}")
    
    metrics_dict = compute_metrics(edge_pred, orient_pred, edge_true, orient_true)
    print("\nMetrics:")
    for k, v in metrics_dict.items():
        print(f"{k}: {v:.4f}")
