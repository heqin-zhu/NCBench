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
        self.edge_criterion = nn.CrossEntropyLoss(weight=edge_weights) # , reduction='none'
        self.orient_criterion = nn.CrossEntropyLoss(weight=orient_weights)
        self.edge_num_class = 4
        self.orient_num_class = 3

    def forward(self, edge_pred, orient_pred, edge_true, orient_true):
        """
        Args:
            edge_pred: shape=(B, L, 4) -> (B, 4, L), to match CrossEntropyLoss
            orient_pred: shape=(B, 3, L, L)
            edge_true: shape=(B, L), \in {0, 1, 2, 3}, pad value=-1
            orient_true: shape=(B, L, L), \in {0,1,2}, pad value=-1
        """
        loss_dic = {}
        for flag, pred, gt, criterion, num_class in [
                             ('edge', edge_pred, edge_true, self.edge_criterion, self.edge_num_class), 
                             ('orient', orient_pred.permute(0, 2, 3, 1), orient_true, self.orient_criterion, self.orient_num_class)]:
            mask = gt!=-1
            # valid_count = torch.clamp(mask.sum(), min=1)
            pred_valid = pred[mask]
            gt_valid = gt[mask]
            loss_dic[flag] = criterion(pred_valid.view(-1, num_class), gt_valid.view(-1))

        loss_dic['loss'] = self.edge_weight * loss_dic['edge'] + self.orient_weight * loss_dic['orient']
        return loss_dic


def compute_metrics(edge_pred, orient_pred, edge_true, orient_true, average='macro'):
    """
    Returns:
        metrics_dict: F1
    """
    batched = len(edge_true.shape)==2
    # predicted class: argmax
    edge_pred_class = torch.argmax(edge_pred, dim=-1)  # (B, L)
    # flatting for convenient computation of metrics
    edge_pred_flat = edge_pred_class.view(-1).cpu().numpy()
    edge_true_flat = edge_true.view(-1).cpu().numpy()

    orient_pred_class = torch.argmax(orient_pred, dim=1 if batched else 0)  # (B, L, L)
    orient_pred_flat = orient_pred_class.view(-1).cpu().numpy()
    orient_true_flat = orient_true.view(-1).cpu().numpy()
    ret = {}
    for flag, pred, gt in [('edge', edge_pred_flat, edge_true_flat), ('orient', orient_pred_flat, orient_true_flat)]:
    
        valid_mask = (gt!=-1)
        pred = pred[valid_mask]
        gt = gt[valid_mask]
        ret[f'{flag}_mcc'] = metrics.matthews_corrcoef(gt, pred)
        ret[f'{flag}_acc'] = metrics.accuracy_score(gt, pred)
        ret[f'{flag}_p'] = metrics.precision_score(gt, pred, average=average, zero_division=0)
        ret[f'{flag}_r'] = metrics.recall_score(gt, pred, average=average, zero_division=0)
        ret[f'{flag}_f1'] = metrics.f1_score(gt, pred, average=average, zero_division=0)
        # TODO
        # ret[f'{flag}_roc'] = metrics.roc_auc_score(gt, pred, average=average, multi_class='ovr')
        for cls in range(4 if flag=='edge' else 3):
            mask = (gt == cls)
            if mask.sum() == 0:
                ret[f'{flag}_cls_{cls}'] = 0.0
            else:
                ret[f'{flag}_cls_{cls}'] = (pred[mask] == cls).mean()
    # TODO
    ret['edge_orient_score'] = (ret['edge_f1'] + ret['orient_f1'])/2
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
    
    loss_dic = criterion(edge_pred, orient_pred, edge_true, orient_true)
    print(loss_dic)
    
    metrics_dict = compute_metrics(edge_pred, orient_pred, edge_true, orient_true)
    print("\nMetrics:")
    for k, v in metrics_dict.items():
        print(f"{k}: {v:.4f}")
