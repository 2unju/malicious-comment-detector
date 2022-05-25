import torch

from tqdm import tqdm

from utils import calc_accuracy
from arguments import get_model_args, get_train_args, get_dataset_args
from config import get_train_config, get_train_log


def train(config, train_log):
    for epoch in range(config["epoch"]):
        train_log["train_accuracy"] = 0.0
        train_log["valid_accuracy"] = 0.0
        train_log["train_loss"] = 0.0
        train_log["valid_loss"] = 0.0

        config["model"].train()
        for batch_id, (comment, label) in tqdm(enumerate(config["train_loader"])):
            input_ids = comment["input_ids"].squeeze(axis=-2).long().to(config["device"])
            token_type_ids = comment["token_type_ids"].squeeze(axis=-2).long().to(config["device"])
            label = label.long().to(config["device"])

            out = config["model"](input_ids, token_type_ids, comment["attention_mask"].to(config["device"]))

            config["optim"].zero_grad()
            loss = config["loss"](out, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(config["model"].parameters(), config["max_grad_norm"])
            config["optim"].step()

            if config["scheduler"]:
                config["scheduler"].step()
                config["scheduler"].set_postfix({'loss': loss.item(), 'lr': config["optim"].param_groups[0]["lr"]})

            train_log["train_accuracy"] += calc_accuracy(out, label)
            train_log["train_loss"] += loss.data.cpu().numpy()

        config["model"].eval()
        for dev_batch_id, (comment, label) in tqdm(enumerate(config["valid_loader"])):
            input_ids = comment["input_ids"].squeeze().long().to(config["device"])
            token_type_ids = comment["token_type_ids"].squeeze().long().to(config["device"])
            label = label.long().to(config["device"])

            out = config["model"](input_ids, token_type_ids, comment["attention_mask"].squeeze().to(config["device"]))

            loss = config["loss"](out, label)

            train_log["valid_accuracy"] += calc_accuracy(out, label)
            train_log["valid_loss"] += loss.data.cpu().numpy()

        train_log["train_accuracy"] /= (batch_id + 1)
        train_log["valid_accuracy"] /= (dev_batch_id + 1)
        train_log["train_loss"] /= (batch_id + 1)
        train_log["valid_loss"] /= (dev_batch_id + 1)

        print("epoch {} train acc {} train loss {}".format(epoch + 1, train_log["train_accuracy"],
                                                           train_log["train_loss"]))
        print("epoch {} test acc {} test loss {}".format(epoch + 1, train_log["valid_accuracy"],
                                                         train_log["valid_loss"]))

        if train_log["min_loss"] is None or train_log["min_loss"] > train_log["valid_loss"]:
            print("test loss {} ---> {}".format(train_log["min_loss"], train_log["valid_loss"]))
            train_log["min_loss"] = train_log["valid_loss"]
            print("Model Saving . . .")
            torch.save(config["model"].state_dict(), './output/best.pth')


if __name__ == "__main__":
    dataset_args = get_dataset_args()
    model_args = get_model_args()
    train_args = get_train_args()

    config = get_train_config(dataset_args, model_args, train_args)
    train_log = get_train_log()
    train(config, train_log)
