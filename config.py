import torch
import torch.nn as nn

from transformers.optimization import get_linear_schedule_with_warmup
from transformers import AdamW
from transformers import AutoModel, AutoTokenizer

from dataset import BERTDataset
from model import MaliciousCommentDetector, MaliciousCommentDetectorwithKoMiniLM


def get_train_config(dataset_args, model_args, train_args):
    tok = AutoTokenizer.from_pretrained(model_args.modelpath)
    trainset_config = {
        "datapath": dataset_args.trainpath,
        "tokenizer": tok,
        "max_len": model_args.max_len,
        "pad": True,
        "pair": False,
        "device": train_args.device
    }
    validset_config = {
        "datapath": dataset_args.validpath,
        "tokenizer": tok,
        "max_len": model_args.max_len,
        "pad": True,
        "pair": False,
        "device": train_args.device
    }
    trainset = BERTDataset(trainset_config)
    validset = BERTDataset(validset_config)

    bert = AutoModel.from_pretrained(model_args.modelpath).to(model_args.device)
    if model_args.modelpath == "klue/bert-base":
        model = MaliciousCommentDetector(bert, model_args).to(model_args.device)
    elif model_args.modelpath == "BM-K/KoMiniLM":
        model = MaliciousCommentDetectorwithKoMiniLM(bert, model_args).to(model_args.device)
    else:
        print("No Available Model")
        exit()

    criterion = nn.CrossEntropyLoss()

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': train_args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=train_args.lr)

    config = {
        "model": model,
        "train_loader": torch.utils.data.DataLoader(trainset, batch_size=train_args.batch_size),
        "valid_loader": torch.utils.data.DataLoader(validset, batch_size=train_args.batch_size),
        "loss": criterion,
        "optim": optimizer,
        "epoch": train_args.epochs,
        "device": train_args.device,
        "max_grad_norm": train_args.max_grad_norm
    }

    if train_args.use_scheduler:
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=train_args.warmup_step,
                                                    num_training_steps=-1)
        config["scheduler"] = scheduler
    else:
        config["scheduler"] = None

    return config


def get_train_log():
    log = {
        "train_accuracy": 0.0,
        "valid_accuracy": 0.0,
        "train_loss": 0.0,
        "valid_loss": 0.0,
        "min_loss": None
    }
    return log


def get_test_config(dataset_args, model_args, train_args):
    tok = AutoTokenizer.from_pretrained(model_args.modelpath)
    testset_config = {
        "datapath": dataset_args.testpath,
        "tokenizer": tok,
        "max_len": model_args.max_len,
        "pad": True,
        "pair": False,
        "device": train_args.device
    }
    testset = BERTDataset(testset_config)

    bert = AutoModel.from_pretrained(model_args.modelpath).to(model_args.device)
    if model_args.modelpath == "klue/bert-base":
        model = MaliciousCommentDetector(bert, model_args).to(model_args.device)
    elif model_args.modelpath == "BM-K/KoMiniLM":
        model = MaliciousCommentDetectorwithKoMiniLM(bert, model_args).to(model_args.device)
    else:
        print("No Available Model")
        exit()

    criterion = nn.CrossEntropyLoss()

    config = {
        "model": model,
        "test_loader": torch.utils.data.DataLoader(testset, batch_size=1),
        "loss": criterion,
        "epoch": train_args.epochs,
        "device": train_args.device,
        "max_grad_norm": train_args.max_grad_norm
    }

    return config
