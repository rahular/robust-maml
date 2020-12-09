import os
import argparse
import logging
import torch

import numpy as np
import torch.nn as nn
import learn2learn as l2l

import utils
import data_utils
import model_utils

from tqdm import tqdm
from collections import defaultdict, deque
from shutil import copyfile

from torch.optim import Adam
from transformers.optimization import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import ConcatDataset, DataLoader

from optims import ALCGD

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
        "--load_from",
        dest="load_from",
        type=str,
        help="Warm start from n existing model path",
        default=None,
        required=False,
    )
    return parser.parse_args()


def save(model, optimizer, config_path, last_epoch, encoder=None):
    save_dir = "./models/{}".format(utils.get_savedir_name())
    os.makedirs(save_dir, exist_ok=True)
    # logger.info("Saving model checkpoint to %s", save_dir)
    copyfile(config_path, "{}/config.json".format(save_dir))
    to_save = model.module if hasattr(model, "module") else model
    torch.save(to_save.state_dict(), os.path.join(save_dir, "best_model.th"))
    torch.save({"optimizer": optimizer.state_dict(), "last_epoch": last_epoch}, os.path.join(save_dir, "optim.th"))
    if encoder:
        to_save = encoder.module if hasattr(encoder, "module") else encoder
        torch.save(to_save.state_dict(), os.path.join(save_dir, "best_encoder.th"))


def meta_train(args, config, train_set, dev_set, label_map, bert_model, clf_head):
    save_dir = "./models/{}".format(utils.get_savedir_name())
    tb_writer = SummaryWriter(os.path.join(save_dir, "logs"))

    train_taskset = data_utils.CustomLangTaskDataset(train_set, train_type=config.train_type)
    dev_taskset = data_utils.CustomLangTaskDataset(dev_set)
    num_epochs = config.num_epochs
    meta_model = l2l.algorithms.MAML(clf_head, lr=config.inner_lr, first_order=config.is_fomaml)
    num_episodes = config.num_episodes
    task_bs = config.task_batch_size
    inner_loop_steps = config.inner_loop_steps

    bert_model = bert_model.eval()
    if config.optim == "adam":
        opt_params = list(meta_model.parameters())
        if config.finetune_enc:
            # NOTE: this condition is never tested as we don't have infinite GPU memory
            bert_model = bert_model.train()
            opt_params += list(bert_model.parameters())
        if config.train_type != "metabase":
            opt_params += list(train_taskset.parameters())
        opt = Adam(opt_params, lr=config.outer_lr)
    elif config.optim == "alcgd":
        if config.train_type == "metabase":
            raise ValueError(f"ALCGD optimizer can only be used for `minmax` or `constrain` train types.")
        opt_params = list(meta_model.parameters())
        if config.finetune_enc:
            # NOTE: this condition is never tested as we don't have infinite GPU memory
            bert_model = bert_model.train()
            opt_params += list(bert_model.parameters())
        torch.backends.cudnn.benchmark = True
        opt = ALCGD(
            max_params=train_taskset.parameters(),
            min_params=opt_params,
            lr_max=config.outer_lr,
            lr_min=config.outer_lr,
            device=DEVICE,
        )
    else:
        raise ValueError(f"Invalid option: {config.optim} for `config.optim`")

    best_dev_error = np.inf
    if args.load_from:
        state_obj = torch.load(os.path.join(args.load_from, "optim.th"))
        opt.load_state_dict(state_obj["optimizer"])
        num_epochs = num_epochs - state_obj["last_epoch"]
        (dev_task, _), _ = dev_taskset.sample(k=config.shots)
        dev_loader = DataLoader(data_utils.InnerDataset(dev_task), batch_size=task_bs, shuffle=False, num_workers=0)
        dev_error, dev_metrics = utils.compute_loss_metrics(
            dev_loader, bert_model, clf_head, label_map, grad_required=False, return_metrics=False
        )
        best_dev_error = dev_error.mean()

    def save_dist(name):
        save_dir = "./models/{}".format(utils.get_savedir_name())
        with open(os.path.join(save_dir, name), "wb") as f:
            np.save(f, train_taskset.tau.detach().cpu().numpy())

    patience_ctr = 0
    constrain_loss_list = defaultdict(lambda: deque(maxlen=10))
    tqdm_bar = tqdm(range(num_epochs))
    for iteration in tqdm_bar:
        dev_iteration_error = 0.0
        train_iteration_error = 0.0
        opt.zero_grad()
        for episode_num in range(num_episodes):
            learner = meta_model.clone()
            (train_task, train_langs), imps = train_taskset.sample(k=config.shots)
            (dev_task, _), _ = dev_taskset.sample(k=config.shots, langs=train_langs)

            for _ in range(inner_loop_steps):
                train_loader = DataLoader(
                    data_utils.InnerDataset(train_task), batch_size=task_bs, shuffle=False, num_workers=0
                )
                train_error, train_metrics = utils.compute_loss_metrics(
                    train_loader, bert_model, learner, label_map=label_map, grad_required=True, return_metrics=False
                )
                train_error = train_error.mean()
                train_iteration_error += train_error
                grads = torch.autograd.grad(train_error, learner.parameters(), create_graph=True, allow_unused=True)
                # TODO: change `max_grad_norm` to something else?
                grads = tuple([g.clamp_(-config.max_grad_norm, config.max_grad_norm) for g in grads])
                l2l.algorithms.maml_update(learner, config.inner_lr, grads)
                opt.zero_grad()

            dev_loader = DataLoader(
                data_utils.InnerDataset(dev_task), batch_size=task_bs, shuffle=False, num_workers=0
            )
            dev_error, dev_metrics = utils.compute_loss_metrics(dev_loader, bert_model, learner, label_map, grad_required=True, return_metrics=False)
            if config.train_type == "minmax":
                dev_error *= imps
                dev_error = dev_error.sum()
            elif config.train_type == "constrain":
                constrain_val = config.constrain_val
                if hasattr(config, "constrain_type") and config.constrain_type == "dynamic":
                    constrain_val = torch.tensor(
                        [
                            np.mean(constrain_loss_list[lang])
                            if len(constrain_loss_list[lang]) > 5
                            else -config.constrain_val
                            for lang in train_langs
                        ]
                    ).to(dev_error.device)
                    for loss_val, lang in zip(dev_error, train_langs):
                        constrain_loss_list[lang].append(loss_val.item())
                dev_error = dev_error.mean() + ((dev_error - constrain_val) * imps).sum()

            elif config.train_type == "metabase":
                dev_error = dev_error.mean()
            else:
                raise ValueError(f"Invalid option: {config.train_type} for `config.train_type`")
            dev_iteration_error += dev_error

            tb_writer.add_scalar("metrics/loss", dev_error, (iteration * num_epochs) + episode_num)
            if dev_metrics is not None:
                tb_writer.add_scalar(
                    "metrics/precision", dev_metrics["precision"], (iteration * num_epochs) + episode_num
                )
                tb_writer.add_scalar("metrics/recall", dev_metrics["recall"], (iteration * num_epochs) + episode_num)
                tb_writer.add_scalar("metrics/f1", dev_metrics["f1"], (iteration * num_epochs) + episode_num)

        dev_iteration_error /= num_episodes
        train_iteration_error /= num_episodes
        if dev_metrics is not None:
            tqdm_bar.set_description(
                "Train. Loss: {:.3f} Train F1: {:.3f} Val. Loss: {:.3f} Val. F1: {:.3f}".format(
                    train_iteration_error.item(), train_metrics["f1"], dev_iteration_error.item(), dev_metrics["f1"]
                )
            )
        else:
            tqdm_bar.set_description(
                "Train. Loss: {:.3f} Val. Loss: {:.3f}".format(
                    train_iteration_error.item(), dev_iteration_error.item()
                )
            )
        if config.optim == "adam":
            dev_iteration_error.backward()
            torch.nn.utils.clip_grad_norm_(opt_params, config.max_grad_norm)
            opt.step()
        elif config.optim == "alcgd":
            opt.step(loss=dev_iteration_error)

        if dev_iteration_error < best_dev_error:
            logger.info("Found new best model!")
            best_dev_error = dev_iteration_error
            save(meta_model, opt, args.config_path, iteration)
            save_dist("best_minmax_dist.npy")
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr == config.patience:
                logger.info("Ran out of patience. Stopping training early...")
                break

        if config.train_type != "metabase" and iteration % 10 == 0:
            save_dist("minmax_dist.npy")

    logger.info(f"Best validation loss = {best_dev_error}")
    logger.info("Best model saved at: {}".format(utils.get_savedir_name()))


def mtl_train(args, config, train_set, dev_set, label_map, bert_model, clf_head):
    save_dir = "./models/{}".format(utils.get_savedir_name())
    tb_writer = SummaryWriter(os.path.join(save_dir, "logs"))

    train_set = ConcatDataset(train_set)
    train_loader = DataLoader(
        dataset=train_set,
        sampler=utils.BalancedTaskSampler(dataset=train_set, batch_size=config.batch_size),
        batch_size=config.batch_size,
        collate_fn=utils.collate_fn,
        shuffle=False,
        num_workers=0,
    )
    dev_set = ConcatDataset(dev_set)
    dev_loader = DataLoader(
        dataset=dev_set, batch_size=config.batch_size, collate_fn=utils.collate_fn, shuffle=False, num_workers=0
    )
    num_epochs = config.num_epochs

    if config.finetune_enc:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in list(bert_model.named_parameters()) + list(clf_head.named_parameters())
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in list(bert_model.named_parameters()) + list(clf_head.named_parameters())
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        opt = AdamW(optimizer_grouped_parameters, eps=1e-8, lr=config.outer_lr)
    else:
        opt = Adam(clf_head.parameters(), lr=config.outer_lr)
        bert_model = bert_model.eval()

    best_dev_error = np.inf
    if args.load_from:
        state_obj = torch.load(os.path.join(args.load_from, "optim.th"))
        opt.load_state_dict(state_obj["optimizer"])
        num_epochs = num_epochs - state_obj["last_epoch"]
        bert_model = bert_model.eval()
        clf_head = clf_head.eval()
        dev_loss, dev_metrics = utils.compute_loss_metrics(
            dev_loader, bert_model, clf_head, label_map, grad_required=False, return_metrics=False
        )
        best_dev_error = dev_loss.mean()

    patience_ctr = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        epoch_iterator = tqdm(train_loader, desc="Training")
        for train_step, (input_ids, attention_mask, token_type_ids, labels, _, _) in enumerate(epoch_iterator):
            # train
            if config.finetune_enc:
                bert_model = bert_model.train()
            clf_head = clf_head.train()
            opt.zero_grad()
            bert_output = bert_model(input_ids, attention_mask, token_type_ids)
            output = clf_head(bert_output, labels=labels, attention_mask=attention_mask)
            loss = output.loss.mean()
            loss.backward()
            if config.finetune_enc:
                torch.nn.utils.clip_grad_norm_(bert_model.parameters(), config.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(clf_head.parameters(), config.max_grad_norm)
            opt.step()
            running_loss += loss.item()
            # eval at the beginning of every epoch and after every `config.eval_freq` steps
            if train_step % config.eval_freq == 0:
                bert_model = bert_model.eval()
                clf_head = clf_head.eval()
                dev_loss, dev_metrics = utils.compute_loss_metrics(
                    dev_loader, bert_model, clf_head, label_map, grad_required=False, return_metrics=False
                )
                dev_loss = dev_loss.mean()

                tb_writer.add_scalar("metrics/loss", dev_loss, epoch)
                if dev_metrics is not None:
                    tb_writer.add_scalar("metrics/precision", dev_metrics["precision"], epoch)
                    tb_writer.add_scalar("metrics/recall", dev_metrics["recall"], epoch)
                    tb_writer.add_scalar("metrics/f1", dev_metrics["f1"], epoch)
                    logger.info(
                        "Dev. metrics (p/r/f): {:.3f} {:.3f} {:.3f}".format(
                            dev_metrics["precision"], dev_metrics["recall"], dev_metrics["f1"]
                        )
                    )

                if dev_loss < best_dev_error:
                    logger.info("Found new best model!")
                    best_dev_error = dev_loss
                    save(clf_head, opt, args.config_path, epoch, bert_model)
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                    if patience_ctr == config.patience:
                        logger.info("Ran out of patience. Stopping training early...")
                        return

        logger.info(f"Finished epoch {epoch+1} with avg. training loss: {running_loss/(train_step + 1)}")

    logger.info(f"Best validation loss = {best_dev_error}")
    logger.info("Best model saved at: {}".format(utils.get_savedir_name()))


def main():
    args = init_args()
    config = model_utils.Config(args.config_path)

    # for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    data_dir = config.data_dir
    train_paths = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith("train")])
    dev_paths = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith("dev")])

    if "/pos/" in data_dir:
        data_class = data_utils.POS
        label_map = {idx: l for idx, l in enumerate(data_utils.get_pos_labels())}
    elif "/ner/" in data_dir:
        data_class = data_utils.NER
        label_map = {idx: l for idx, l in enumerate(data_utils.get_ner_labels())}
    elif "/tydiqa/" in data_dir or "squad" in data_dir:
        data_class = data_utils.QA
        label_map = None
    else:
        raise ValueError(f"Unknown task or incorrect `config.data_dir`: {config.data_dir}")

    # NOTE: if `label_map` is None, the task is not sequence labeing
    bert_model = model_utils.BERT(config)
    if label_map is not None:
        train_max_examples = config.train_max_examples
        dev_max_examples = config.dev_max_examples
        logger.info("Creating train sets...")
        train_set = [
            data_class(p, config.max_seq_length, config.model_type, train_max_examples) for p in tqdm(train_paths)
        ]
        logger.info("Creating dev sets...")
        dev_set = [
            data_class(p, config.max_seq_length, config.model_type, dev_max_examples) for p in tqdm(dev_paths)
        ]
        clf_head = model_utils.SeqClfHead(len(label_map), config.hidden_dropout_prob, bert_model.get_hidden_size())
    else:
        logger.info("Creating train sets...")
        train_set = [
            data_class(p, config.max_clen, config.max_qlen, config.doc_stride, config.model_type)
            for p in tqdm(train_paths)
        ]
        logger.info("Creating dev sets...")
        dev_set = [
            data_class(p, config.max_clen, config.max_qlen, config.doc_stride, config.model_type)
            for p in tqdm(dev_paths)
        ]
        clf_head = model_utils.ClfHead(config.hidden_dropout_prob, bert_model.get_hidden_size())

    if args.load_from:
        logger.info(f"Resuming training with weights from {args.load_from}")
        utils.set_savedir_name(args.load_from.split("/")[-1])
        clf_head.load_state_dict(torch.load(os.path.join(args.load_from, "best_model.th")))
    if args.load_from or hasattr(config, "encoder_ckpt"):
        if args.load_from and hasattr(config, "encoder_ckpt"):
            raise ValueError("Conflict: both `load_from` and `encoder_ckpt` set.")
        load_path = args.load_from if not hasattr(config, "encoder_ckpt") else config.encoder_ckpt
        logger.info(f"Using encoder weights from {load_path}")
        bert_model.load_state_dict(torch.load(os.path.join(load_path, "best_encoder.th")))
    bert_model = bert_model.to(DEVICE)
    clf_head = clf_head.to(DEVICE)

    if config.train_type == "mtl":
        mtl_train(args, config, train_set, dev_set, label_map, bert_model, clf_head)
    else:
        meta_train(args, config, train_set, dev_set, label_map, bert_model, clf_head)


if __name__ == "__main__":
    main()
