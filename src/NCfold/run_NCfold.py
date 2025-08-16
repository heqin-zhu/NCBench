import os
import argparse

from torch.optim import AdamW

from RNAdata import NucClsDataset
from collators import NucClsCollator
from trainers import SeqClsTrainer
from models.losses import NucClsLoss
from models.metrics import NucClsMetrics
from utils.NCfold_kit import str2bool, str2list, count_para, get_config

from NCpair import NCpair_model

MODELS = ['NCpair_model']

TASKS = ["NCpair"]

parser = argparse.ArgumentParser('Implementation of RNA sequence classification.')
# model args
parser.add_argument('--model_name', type=str, default="NCpair_model", choices=MODELS)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--checkpoint_path', type=str)
parser.add_argument('--config_path', type=str, default="./configs/")
parser.add_argument('--dataset_dir', type=str, default="../data")
parser.add_argument('--dataset', type=str, default="NCpair", choices=TASKS)
parser.add_argument('--replace_T', type=bool, default=True)
parser.add_argument('--replace_U', type=bool, default=False)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_seq_len', type=int, default=512)
parser.add_argument('--dataloader_num_workers', type=int, default=0)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--train', type=str2bool, default=True)
parser.add_argument('--batch_size', type=int, default=32,
                    help='The number of samples used per step & per device.')
parser.add_argument('--num_train_epochs', type=int, default=60,
                    help='The number of epoch for training.')
parser.add_argument('--metrics', type=str2list,
                    default="F1s,Precision,Recall,Accuracy,Mcc",)
parser.add_argument('--logging_steps', type=int, default=1000,
                    help='Update visualdl logs every logging_steps.')
# save checkpoint
parser.add_argument('--output_dir', type=str)
args = parser.parse_args()


if __name__ == "__main__":
    assert args.replace_T ^ args.replace_U, "Only replace T or U."

    if args.model_name == "NCpair_model":
        model = NCpair_model(d_model=args.hidden_dim)
    else:
        raise ValueError("Unknown model name: {}".format(args.model_name))
    model.to(args.device)
    count_para(model)

    _loss_fn = NucClsLoss().to(args.device)
    if args.output_dir is None:
        run_name = os.path.basename(args.checkpoint_path)
        run_name = run_name[:run_name.rfind('.')]
        args.output_dir = os.path.join('outputs', run_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # ========== Prepare data
    dataset_train = NucClsDataset(data_dir=args.dataset_dir)
    dataset_eval = NucClsDataset(data_dir=args.dataset_dirtrain=False)
    print(f'dataset dir: {args.dataset_dir} {args.dataset}')
    print(f'dataset {args.dataset} train:test={len(dataset_train)}:{len(dataset_eval)}') 

    # ========== Create the data collator
    _collate_fn = NucClsCollator(max_seq_len=args.max_seq_len, replace_T=args.replace_T, replace_U=args.replace_U)

    # ========== Create the learning_rate scheduler (if need) and optimizer
    optimizer = AdamW(params=model.parameters(), lr=args.learning_rate)

    # ========== Create the metrics
    _metric = NucClsMetrics(metrics=args.metrics)

    # ========== Create the trainer
    seq_cls_trainer = SeqClsTrainer(
        args=args,
        model=model,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        data_collator=_collate_fn,
        loss_fn=_loss_fn,
        optimizer=optimizer,
        compute_metrics=_metric,
    )
    if args.train:
        for i_epoch in range(args.num_train_epochs):
            print("Epoch: {}".format(i_epoch))
            seq_cls_trainer.train(i_epoch)
            seq_cls_trainer.eval(i_epoch)
