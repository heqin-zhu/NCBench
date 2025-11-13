import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics


class NCfoldLoss(nn.Module):
    """
        edge_arr: 4-class
        orient_mat: 3-class
    """
    def __init__(self, edge_weight=1.0, 
                 orient_weight=1.0, 
                 edge_weights=None, 
                 orient_weights=None, 
                 label_smoothing=0.05, 
                 use_uncertainty_weighting=False):
        super().__init__()
        self.edge_weight = edge_weight  # edge_arr weight
        self.orient_weight = orient_weight  # orient_mat weight
        self.label_smoothing = label_smoothing
        self.use_uncertainty_weighting = use_uncertainty_weighting
        
        self.edge_weights = edge_weights
        self.orient_weights = orient_weights
        # class-weights, for imbalanced classification
        self.edge_criterion = nn.CrossEntropyLoss(weight=edge_weights) # , reduction='none'
        self.orient_criterion = nn.CrossEntropyLoss(weight=orient_weights)
        self.edge_num_class = 4
        self.orient_num_class = 3
        
        self.register_buffer("edge_w", edge_weights.float() if edge_weights is not None else None)
        self.register_buffer("orient_w", orient_weights.float() if orient_weights is not None else None)

        if self.use_uncertainty_weighting:
            # learnable weights for multi-task loss
            self.log_sigma_edge = nn.Parameter(torch.zeros(()))
            self.log_sigma_orient = nn.Parameter(torch.zeros(()))
            print("Using uncertainty weighting for multi-task loss.")

    @staticmethod
    def _safe_ce(logits: torch.Tensor, target: torch.Tensor, weight=None, label_smoothing=0.0):
        if target.numel() == 0:
            return torch.zeros((), device=logits.device, dtype=logits.dtype)
        return F.cross_entropy(
            logits, target.long(),
            weight=weight,
            label_smoothing=label_smoothing,
            reduction='mean'
        )
        
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
            if self.use_uncertainty_weighting:
                loss_dic[flag] = self._safe_ce(
                    pred_valid.view(-1, num_class),
                    gt_valid.view(-1),
                    weight=self.orient_weights if flag=='orient' else self.edge_weights,
                    label_smoothing=self.label_smoothing
                )
                sigma_e = torch.exp(self.log_sigma_edge)
                sigma_o = torch.exp(self.log_sigma_orient)
                
            else:
                loss_dic[flag] = criterion(pred_valid.view(-1, num_class), gt_valid.view(-1))
        
        if self.use_uncertainty_weighting:
            loss_dic['loss'] = (loss_dic['edge'] / (2 * sigma_e ** 2) + self.log_sigma_edge + \
                     loss_dic['orient'] / (2 * sigma_o ** 2) + self.log_sigma_orient)
        else:
            loss_dic['loss'] = self.edge_weight * loss_dic['edge'] + self.orient_weight * loss_dic['orient']
        return loss_dic


def compute_metrics(edge_pred, orient_pred, edge_true, orient_true, average='macro'):
    """
    Returns:
        metrics_dict: F1
    """
    is_batch_data = len(edge_true.shape)==2
    # predicted class: argmax
    edge_pred_class = torch.argmax(edge_pred, dim=-1)  # (B, L)
    # flatting for convenient computation of metrics
    edge_pred_flat = edge_pred_class.view(-1).cpu().numpy()
    edge_true_flat = edge_true.view(-1).cpu().numpy()

    orient_pred_class = torch.argmax(orient_pred, dim=1 if is_batch_data else 0)  # (B, L, L)
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
    ret['nc_score'] = (ret['edge_f1'] + ret['orient_f1'])/2
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
