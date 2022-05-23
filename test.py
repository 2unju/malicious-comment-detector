import torch

from utils import calc_accuracy
from arguments import get_model_args, get_train_args, get_dataset_args
from config import get_test_config

def test(config):
    acc = 0.0
    _loss = 0.0

    config["model"].load_state_dict(torch.load('./output/best.pth'))
    config["model"].eval()
    for batch_id, (comment, label) in enumerate(config["test_loader"]):
        input_ids = comment["input_ids"].squeeze().long().to(config["device"])
        token_type_ids = comment["token_type_ids"].squeeze().long().to(config["device"])
        label = label.long().to(config["device"])

        out = config["model"](input_ids, token_type_ids, comment["attention_mask"].squeeze().to(config["device"]))
        loss = config["loss"](out, label)

        acc += calc_accuracy(out, label)
        _loss += loss.data.cpu().numpy()

    print("acc {} loss {}".format(acc / (batch_id + 1), _loss / (batch_id + 1)))

if __name__ == "__main__":
    dataset_args = get_dataset_args()
    model_args = get_model_args()
    train_args = get_train_args()

    config = get_test_config(dataset_args, model_args, train_args)
    test(config)
