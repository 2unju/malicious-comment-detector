import argparse


def get_dataset_args():
    dataset_parser = argparse.ArgumentParser()

    dataset_parser.add_argument("--trainpath", default="./dataset/train.txt")
    dataset_parser.add_argument("--validpath", default="./dataset/valid.txt")
    dataset_parser.add_argument("--testpath", default="./dataset/test.txt")
    dataset_parser.add_argument("--device", default="cuda:1")

    args = dataset_parser.parse_args()
    return args


def get_model_args():
    model_parser = argparse.ArgumentParser()

    model_parser.add_argument("--max-len", default=128)
    model_parser.add_argument("--num-labels", default=2)
    model_parser.add_argument("--dr-rate", default=None, help="DROPOUT RATE")
    model_parser.add_argument("--hidden-size", default=768)
    model_parser.add_argument("--device", default="cuda:1")

    args = model_parser.parse_args()
    return args


def get_train_args():
    train_parser = argparse.ArgumentParser()

    train_parser.add_argument("--batch-size", default=32)
    train_parser.add_argument("--epochs", default=5)
    train_parser.add_argument("--lr", default=5e-5, help="LEARNING RATE")
    train_parser.add_argument("--hidden-size", default=768)
    train_parser.add_argument("--warmup-ratio", default=0.1)
    train_parser.add_argument("--warmup_step", default=0.1)
    train_parser.add_argument("--device", default="cuda:1")
    train_parser.add_argument("--num-workers", default=5)
    train_parser.add_argument("--weight-decay", default=1e-4)
    train_parser.add_argument("--use-scheduler", default=False)
    train_parser.add_argument("--max-grad-norm", default=1)

    args = train_parser.parse_args()
    return args
