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
        
        loss = self.edge_weight * loss_edge + self.orient_weight * loss_orient
        return loss, loss_edge, loss_orient


def compute_metrics(edge_pred, orient_pred, edge_true, orient_true, average='macro'):
    """
    Returns:
        metrics_dict: F1
    """
    # predicted class: argmax
    edge_pred_class = torch.argmax(edge_pred, dim=-1)  # (B, L)
    # flatting for convenient computation of metrics
    edge_pred_flat = edge_pred_class.view(-1).cpu().numpy()
    edge_true_flat = edge_true.view(-1).cpu().numpy()

    orient_pred_class = torch.argmax(orient_pred, dim=1)  # (B, L, L)
    orient_pred_flat = orient_pred_class.view(-1).cpu().numpy()
    orient_true_flat = orient_true.view(-1).cpu().numpy()
    ret = {}

    for flag, pred, gt in [('edge', edge_pred_flat, edge_true_flat), ('orient', orient_pred_flat, orient_true_flat)]:
    
        ret[f'{flag}_mcc'] = metrics.matthews_corrcoef(gt, pred)
        ret[f'{flag}_acc'] = metrics.accuracy_score(gt, pred)
        ret[f'{flag}_p'] = metrics.precision_score(gt, pred, average=average)
        ret[f'{flag}_r'] = metrics.recall_score(gt, pred, average=average)
        ret[f'{flag}_f1'] = metrics.f1_score(gt, pred, average=average)
        # TODO
        # ret[f'{flag}_roc'] = metrics.roc_auc_score(gt, pred, average=average, multi_class='ovr')
        for cls in range(4 if flag=='edge' else 3):
            mask = (gt == cls)
            if mask.sum() == 0:
                ret[f'{flag}_cls_{cls}'] = 0.0
            else:
                ret[f'{flag}_cls_{cls}'] = (pred[mask] == cls).mean()
    # TODO
    ret['edge_orient_score'] = ret['edge_f1'] + ret['orient_f1']
    return ret


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
    
    loss, loss_edge, loss_orient = criterion(edge_pred, orient_pred, edge_true, orient_true)
    print(f"Total loss: {loss.item():.4f}")
    print(f"Edge loss: {loss_edge.item():.4f}, Orient loss: {loss_orient.item():.4f}")
    
    metrics_dict = compute_metrics(edge_pred, orient_pred, edge_true, orient_true)
    print("\nMetrics:")
    for k, v in metrics_dict.items():
        print(f"{k}: {v:.4f}")
