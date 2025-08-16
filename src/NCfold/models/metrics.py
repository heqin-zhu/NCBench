import abc
import math

import torch
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    roc_auc_score)


class BaseMetrics(abc.ABC):
    """Base class for functional tasks metrics
    """

    def __init__(self, metrics):
        """
        Args:
            metrics: names in list
        """
        self.metrics = [x.lower() for x in metrics]

    @abc.abstractmethod
    def __call__(self, outputs, labels):
        """
        Args:
            kwargs: required args of model (dict)

        Returns:
            metrics in dict
        """
        preds = torch.argmax(outputs, axis=-1)
        preds = preds.cpu().numpy().astype('int32')
        labels = labels.cpu().numpy().astype('int32')

        res = {}
        for name in self.metrics:
            func = getattr(self, name)
            if func:
                if func == self.auc:
                    # given two neural outputs, calculate their logits
                    # and then calculate auc
                    logits = torch.sigmoid(outputs).cpu().numpy()
                    m = func(logits, labels)
                else:
                    m = func(preds, labels)
                res[name] = m
            else:
                raise NotImplementedError
        return res

    @staticmethod
    def accuracy(preds, labels):
        """
        All args have same shapes.
        Args:
            preds: predictions of model, (batch_size, 1)
            labels: ground truth, (batch_size, 1)

        Returns:
            accuracy
        """
        return accuracy_score(labels, preds)

    @staticmethod
    def precision(preds, labels):
        """
        All args have same shapes.
        Args:
            preds: predictions of model, (batch_size, 1)
            labels: ground truth, (batch_size, 1)

        Returns:
            precision
        """
        return precision_score(labels, preds, average='macro')

    @staticmethod
    def recall(preds, labels):
        """
        All args have same shapes.
        Args:
            preds: predictions of model, (batch_size, 1)
            labels: ground truth, (batch_size, 1)

        Returns:
            precision
        """
        return recall_score(labels, preds, average='macro')

    @staticmethod
    def f1s(preds, labels):
        """
        All args have same shapes.
        Args:
            preds: predictions of model, (batch_size, 1)
            labels: ground truth, (batch_size, 1)

        Returns:
            precision
        """
        return f1_score(labels, preds, average='macro')

    @staticmethod
    def mcc(preds, labels):
        """
        All args have same shapes.
        Args:
            preds: predictions of model, (batch_size, 1)
            labels: ground truth, (batch_size, 1)

        Returns:
            precision
        """
        return matthews_corrcoef(labels, preds)

    @staticmethod
    def auc(preds, labels):
        """
        All args have same shapes.
        Args:
            preds: predictions of model, (batch_size, 1)
            labels: ground truth, (batch_size, 1)

        Returns:
            precision
        """
        labels += 1
        preds = preds[:, 1]
        return roc_auc_score(labels, preds)


class NucClsMetrics(abc.ABC):
    """Base class for functional tasks metrics
    """

    def __init__(self, metrics):
        """
        Args:
            metrics: names in list
        """
        self.metrics = [x.lower() for x in metrics]

    @abc.abstractmethod
    def __call__(self, outputs, labels):
        """
        Args:
            kwargs: required args of model (dict)

        Returns:
            metrics in dict
        """
        preds = torch.argmax(outputs, axis=-1)
        preds = preds.cpu().numpy().astype('int32')
        labels = labels.cpu().numpy().astype('int32')

        res = {}
        for name in self.metrics:
            func = getattr(self, name)
            if func:
                if func == self.auc:
                    # given two neural outputs, calculate their logits
                    # and then calculate auc
                    logits = torch.sigmoid(outputs).cpu().numpy()
                    m = func(logits, labels)
                else:
                    m = func(preds, labels)
                res[name] = m
            else:
                raise NotImplementedError
        return res

    @staticmethod
    def accuracy(preds, labels):
        """
        All args have same shapes.
        Args:
            preds: predictions of model, (batch_size, 1)
            labels: ground truth, (batch_size, 1)

        Returns:
            accuracy
        """
        return accuracy_score(labels, preds)

    @staticmethod
    def precision(preds, labels):
        """
        All args have same shapes.
        Args:
            preds: predictions of model, (batch_size, 1)
            labels: ground truth, (batch_size, 1)

        Returns:
            precision
        """
        return precision_score(labels, preds, average='macro')

    @staticmethod
    def recall(preds, labels):
        """
        All args have same shapes.
        Args:
            preds: predictions of model, (batch_size, 1)
            labels: ground truth, (batch_size, 1)

        Returns:
            precision
        """
        return recall_score(labels, preds, average='macro')

    @staticmethod
    def f1s(preds, labels):
        """
        All args have same shapes.
        Args:
            preds: predictions of model, (batch_size, 1)
            labels: ground truth, (batch_size, 1)

        Returns:
            precision
        """
        return f1_score(labels, preds, average='macro')

    @staticmethod
    def mcc(preds, labels):
        """
        All args have same shapes.
        Args:
            preds: predictions of model, (batch_size, 1)
            labels: ground truth, (batch_size, 1)

        Returns:
            precision
        """
        return matthews_corrcoef(labels, preds)

    @staticmethod
    def auc(preds, labels):
        """
        All args have same shapes.
        Args:
            preds: predictions of model, (batch_size, 1)
            labels: ground truth, (batch_size, 1)

        Returns:
            precision
        """
        labels += 1
        preds = preds[:, 1]
        return roc_auc_score(labels, preds)


class NucClsMetrics(NucClsMetrics):
    def __call__(self, outputs, labels):
        return super().__call__(outputs, labels)

def compare_bpseq(ref, pred):
    L = len(ref) - 1
    tp = fp = fn = 0
    if (len(ref) > 0 and isinstance(ref[0], list)) or (isinstance(ref, torch.Tensor) and ref.ndim == 2):
        if isinstance(ref, torch.Tensor):
            ref = ref.tolist()
        ref = {(min(i, j), max(i, j)) for i, j in ref}
        pred = {(i, j) for i, j in enumerate(pred) if i < j}
        tp = len(ref & pred)
        fp = len(pred - ref)
        fn = len(ref - pred)
    else:
        assert (len(ref) == len(pred)), print(len(ref), ref, len(pred), pred)
        for i, (j1, j2) in enumerate(zip(ref, pred)):
            if j1 > 0 and i < j1:  # pos
                if j1 == j2:
                    tp += 1
                elif j2 > 0 and i < j2:
                    fp += 1
                    fn += 1
                else:
                    fn += 1
            elif j2 > 0 and i < j2:
                fp += 1
    tn = L * (L - 1) // 2 - tp - fp - fn
    return tp, tn, fp, fn


class SspMetrics(BaseMetrics):
    def __init__(self, metrics):
        super(SspMetrics, self).__init__(metrics=metrics)

    def __call__(self, tp, tn, fp, fn):
        ret = {}
        acc = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn > 0. else 0.
        sen = tp / (tp + fn) if tp + fn > 0. else 0.
        ppv = tp / (tp + fp) if tp + fp > 0. else 0.
        fval = 2 * sen * ppv / (sen + ppv) if sen + ppv > 0. else 0.
        mcc = ((tp * tn) - (fp * fn)) / math.sqrt(
            (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0. else 0.
        for m in self.metrics:
            if m == "accuracy":
                ret[m] = acc
            elif m == "recall":
                ret[m] = sen
            elif m == "precision":
                ret[m] = ppv
            elif m == "f1s":
                ret[m] = fval
            elif m == "mcc":
                ret[m] = mcc
        return ret
