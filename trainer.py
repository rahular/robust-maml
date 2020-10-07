import os
import argparse
import logging
import torch

import numpy as np
import learn2learn as l2l

import utils
import data_utils
import model_utils

from tqdm import tqdm
from shutil import copyfile

from torch.optim import Adam
from torch.utils.data import ConcatDataset, DataLoader
from learn2learn.data import MetaDataset, TaskDataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_args():
    parser = argparse.ArgumentParser(description="Train a classifier")
    parser.add_argument(
        "--config_path",
        dest="config_path",
        type=str,
        help="Path of the config containing training params",
        required=True,
    )
    parser.add_argument(
        "--train_type",
        dest="train_type",
        type=str,
        help="Whether to perform MTL or meta-training",
        choices=["meta", "mtl"],
        required=False,
        default="meta",
    )
    return parser.parse_args()


def save(model, optimizer, config_path, last_epoch):
    save_dir = "./models/{}".format(utils.get_savedir_name())
    os.makedirs(save_dir, exist_ok=True)
    # logging.info("Saving model checkpoint to %s", save_dir)
    copyfile(config_path, "{}/config.json".format(save_dir))
    model_to_save = model.module if hasattr(model, "module") else model
    torch.save(model_to_save.state_dict(), os.path.join(save_dir, "best_model.th"))
    torch.save({"optimizer": optimizer.state_dict(), "last_epoch": last_epoch}, os.path.join(save_dir, "optim.th"))


def meta_train(args, config, train_set, dev_set, label_map, bert_model, clf_head):
    train_set = MetaDataset(train_set)
    train_taskset = TaskDataset(
        train_set,
        [
            # here n=1, because we want only one language in an episode
            l2l.data.transforms.FusedNWaysKShots(train_set, n=1, k=config.shots),
            l2l.data.transforms.LoadData(train_set),
        ],
        num_tasks=config.num_tasks,
    )

    dev_set = MetaDataset(dev_set)
    dev_taskset = TaskDataset(
        dev_set,
        [
            # here n=1, because we want only one language in an episode
            l2l.data.transforms.FusedNWaysKShots(dev_set, n=1, k=config.shots),
            l2l.data.transforms.LoadData(dev_set),
        ],
        num_tasks=config.num_tasks,
    )

    num_epochs = config.num_epochs
    meta_model = l2l.algorithms.MAML(clf_head, lr=config.inner_lr, first_order=config.is_fomaml)
    opt = Adam(meta_model.parameters(), lr=config.outer_lr)
    tqdm_bar = tqdm(range(num_epochs))
    num_episodes = config.num_episodes
    task_bs = config.task_batch_size
    inner_loop_steps = config.inner_loop_steps

    best_dev_error = np.inf
    patience_ctr = 0
    for iteration in tqdm_bar:
        dev_iteration_error = 0.0
        train_iteration_error = 0.0
        for _ in range(num_episodes):
            learner = meta_model.clone()
            train_task, _ = train_taskset.sample()
            dev_task, _ = dev_taskset.sample()

            for _ in range(inner_loop_steps):
                train_loader = DataLoader(
                    data_utils.InnerPOSDataset(train_task), batch_size=task_bs, shuffle=True, num_workers=0
                )
                train_error, train_metrics = utils.compute_loss_metrics(
                    train_loader, bert_model, learner, label_map=label_map
                )
                grads = torch.autograd.grad(train_error, learner.parameters(), create_graph=True, allow_unused=True)
                l2l.algorithms.maml_update(learner, config.inner_lr, grads)

            dev_loader = DataLoader(
                data_utils.InnerPOSDataset(dev_task), batch_size=task_bs, shuffle=True, num_workers=0
            )
            dev_error, dev_metrics = utils.compute_loss_metrics(dev_loader, bert_model, learner, label_map=label_map)
            dev_iteration_error += dev_error
            train_iteration_error += train_error

        dev_iteration_error /= num_episodes
        train_iteration_error /= num_episodes
        tqdm_bar.set_description(
            "Train. Loss: {:.3f} Train F1: {:.3f} Val. Loss: {:.3f} Val. F1: {:.3f}".format(
                train_iteration_error.item(), train_metrics["f1"], dev_iteration_error.item(), dev_metrics["f1"]
            )
        )
        opt.zero_grad()
        dev_iteration_error.backward()
        opt.step()
        meta_model.zero_grad()

        if dev_iteration_error < best_dev_error:
            # logging.info("Found new best model!")
            best_dev_error = dev_iteration_error
            save(meta_model, opt, args.config_path, iteration)
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr == config.patience:
                logging.info("Ran out of patience. Stopping training early...")
                break

    logging.info(f"Best validation loss = {best_dev_error}")
    logging.info("Best model saved at: {}".format(utils.get_savedir_name()))


def mtl_train(args, config, train_set, dev_set, label_map, bert_model, clf_head):
    train_loader = DataLoader(
        dataset=train_set,
        sampler=utils.BalancedTaskSampler(dataset=train_set, batch_size=config.batch_size),
        batch_size=config.batch_size,
        collate_fn=utils.pos_collate_fn,
        shuffle=False,
    )
    dev_loader = DataLoader(
        dataset=dev_set,
        sampler=utils.BalancedTaskSampler(dataset=dev_set, batch_size=config.batch_size),
        batch_size=config.batch_size,
        collate_fn=utils.pos_collate_fn,
        shuffle=False,
    )
    num_epochs = config.num_epochs
    opt = Adam(clf_head.parameters(), lr=config.outer_lr)

    best_dev_loss = np.inf
    for epoch in range(num_epochs):
        running_loss = 0.0
        epoch_iterator = tqdm(train_loader, desc="Training")
        for train_step, (input_ids, attention_mask, token_type_ids, labels, _) in enumerate(epoch_iterator):
            clf_head.zero_grad()
            with torch.no_grad():
                bert_output = bert_model(input_ids, attention_mask, token_type_ids)
            output = clf_head(bert_output, labels=labels, attention_mask=attention_mask)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(clf_head.parameters(), config.max_grad_norm)
            opt.step()
            running_loss += output.loss.item()
        logger.info(f"Finished epoch {epoch+1} with avg. training loss: {running_loss/(train_step + 1)}")

        dev_loss, dev_metrics = utils.compute_loss_metrics(dev_loader, bert_model, clf_head, label_map)
        logging.info(
            "Dev. metrics (p/r/f): {:.3f} {:.3f} {:.3f}".format(
                dev_metrics["precision"], dev_metrics["recall"], dev_metrics["f1"]
            )
        )
        if dev_loss < best_dev_loss:
            # logging.info("Found new best model!")
            best_dev_loss = dev_loss
            save(clf_head, opt, args.config_path, epoch)
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr == config.patience:
                logging.info("Ran out of patience. Stopping training early...")
                break

    logging.info(f"Best validation loss = {best_dev_loss}")
    logging.info("Best model saved at: {}".format(utils.get_savedir_name()))


def main():
    args = init_args()
    config = model_utils.Config(args.config_path)

    # for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    train_paths = config.train_paths
    dev_paths = config.dev_paths

    train_set = ConcatDataset([data_utils.POS(p, config.max_seq_length, config.model_type) for p in train_paths])
    dev_set = ConcatDataset([data_utils.POS(p, config.max_seq_length, config.model_type) for p in dev_paths])

    label_map = {idx: l for idx, l in enumerate(data_utils.get_pos_labels())}
    bert_model = model_utils.BERT(config)
    bert_model = bert_model.eval().to(DEVICE)
    clf_head = model_utils.SeqClfHead(len(label_map), config.hidden_dropout_prob, bert_model.get_hidden_size())
    clf_head = clf_head.to(DEVICE)

    if args.train_type == "meta":
        meta_train(args, config, train_set, dev_set, label_map, bert_model, clf_head)
    elif args.train_type == "mtl":
        mtl_train(args, config, train_set, dev_set, label_map, bert_model, clf_head)


if __name__ == "__main__":
    main()
