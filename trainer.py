import os
import argparse
import logging
import torch

import numpy as np

import utils
import data_utils
import model_utils
import meta_utils

from tqdm import tqdm
from collections import defaultdict, deque
from shutil import copyfile

from torch.optim import Adam
from transformers.optimization import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import ConcatDataset, DataLoader

from optims import ALCGD, GDA

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
    torch.save(model.state_dict(), os.path.join(save_dir, "best_model.th"))
    torch.save({"optimizer": optimizer.state_dict(), "last_epoch": last_epoch}, os.path.join(save_dir, "optim.th"))
    if encoder is not None:
        torch.save(encoder.state_dict(), os.path.join(save_dir, "best_encoder.th"))


def meta_train(args, config, train_set, dev_set, label_map, bert_model, clf_head):
    save_dir = "./models/{}".format(utils.get_savedir_name())
    tb_writer = SummaryWriter(os.path.join(save_dir, "logs"))

    split_fraction = 1. * config.inner_loop_steps / (config.inner_loop_steps + 1)
    train_set_1, train_set_2 = [], []
    for dataset in train_set:
        ts1 = int(split_fraction * len(dataset))
        ts2 = len(dataset) - ts1
        td1, td2 = torch.utils.data.random_split(dataset, [ts1, ts2],
            generator=torch.Generator().manual_seed(config.seed))
        train_set_1.append(td1)
        train_set_2.append(td2)

    train_taskset = data_utils.CustomLangTaskDataset(train_set_1, train_type=config.train_type)
    dev_taskset = data_utils.CustomLangTaskDataset(train_set_2)

    eval_set = ConcatDataset(dev_set)
    eval_loader = DataLoader(
        dataset=eval_set, batch_size=config.task_batch_size, collate_fn=utils.collate_fn, shuffle=False, num_workers=0
    )

    num_epochs = config.num_epochs
    task_bs = config.task_batch_size
    inner_loop_steps = config.inner_loop_steps
    num_episodes = len(ConcatDataset(train_set_2)) // task_bs

    meta_clf = meta_utils.ParamMetaSGD(clf_head, lr=config.inner_lr, first_order=config.is_fomaml)
    if not config.finetune_enc:
        for param in bert_model.parameters():
            param.requires_grad = False
        extra = []
        meta_encoder = bert_model
    else:
        meta_encoder = meta_utils.ParamMetaSGD(bert_model, lr=config.inner_lr, first_order=config.is_fomaml)
        extra = [p for p in meta_encoder.parameters()]

    opt_params = list(meta_clf.parameters()) + extra
    if config.train_type == "metabase":
        opt = Adam(opt_params, lr=config.outer_lr)
    else:
        if config.optim == "adam":
            opt = GDA(
                max_params=train_taskset.parameters(),
                min_params=opt_params,
                lr_max=config.outer_lr,
                lr_min=config.outer_lr,
                device=DEVICE,
            )
        elif config.optim == "alcgd":
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
    eval_freq = config.eval_freq // (config.inner_loop_steps + 1)
    patience_over = False
    constrain_loss_list = defaultdict(lambda: deque(maxlen=10))
    tqdm_bar = tqdm(range(num_epochs))
    for iteration in tqdm_bar:
        dev_iteration_error = 0.0
        train_iteration_error = 0.0
        meta_encoder.train()
        meta_clf.train()
        episode_iterator = tqdm(range(num_episodes), desc="Training")
        for episode_num in episode_iterator:
            learner = meta_clf.clone()
            encoder = meta_encoder.clone() if config.finetune_enc else meta_encoder
            (train_task, train_langs), imps = train_taskset.sample(k=config.shots)
            (dev_task, _), _ = dev_taskset.sample(k=config.shots, langs=train_langs)

            for _ in range(inner_loop_steps):
                train_loader = DataLoader(
                    data_utils.InnerDataset(train_task), batch_size=task_bs, shuffle=True, num_workers=0
                )
                train_error, train_metrics = utils.compute_loss_metrics(
                    train_loader, encoder, learner, label_map=label_map,
                    return_metrics=False, enc_grad_required=config.finetune_enc
                )
                train_error = train_error.mean()
                train_iteration_error += train_error.item()
                learner.adapt(train_error, retain_graph=config.finetune_enc)
                if config.finetune_enc:
                    encoder.adapt(train_error, allow_unused=True)

            dev_loader = DataLoader(
                data_utils.InnerDataset(dev_task), batch_size=task_bs, shuffle=True, num_workers=0
            )
            dev_error, dev_metrics = utils.compute_loss_metrics(
                dev_loader,
                encoder,
                learner,
                label_map,
                return_metrics=False,
                enc_grad_required=config.finetune_enc,
            )
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

            if config.train_type == "metabase":
                dev_error.backward()
                opt.step()
            else:
                opt.step(loss=dev_error)
            opt.zero_grad()

            dev_iteration_error += dev_error.item()

            tb_writer.add_scalar("metrics/loss", dev_error, (iteration * num_epochs) + episode_num)
            if dev_metrics is not None:
                tb_writer.add_scalar(
                    "metrics/precision", dev_metrics["precision"], (iteration * num_epochs) + episode_num
                )
                tb_writer.add_scalar("metrics/recall", dev_metrics["recall"], (iteration * num_epochs) + episode_num)
                tb_writer.add_scalar("metrics/f1", dev_metrics["f1"], (iteration * num_epochs) + episode_num)

            if episode_num and episode_num % eval_freq == 0:
                dev_iteration_error /= eval_freq
                train_iteration_error /= eval_freq * inner_loop_steps
                if dev_metrics is not None:
                    tqdm_bar.set_description(
                        "Train. Loss: {:.3f} Train F1: {:.3f} Dev. Loss: {:.3f} Dev. F1: {:.3f}".format(
                            train_iteration_error, train_metrics["f1"], dev_iteration_error, dev_metrics["f1"]
                        )
                    )
                else:
                    tqdm_bar.set_description(
                        "Train. Loss: {:.3f} Dev. Loss: {:.3f}".format(train_iteration_error, dev_iteration_error)
                    )

                meta_clf.eval()
                meta_encoder.eval()
                eval_loss, _ = utils.compute_loss_metrics(
                    eval_loader, meta_encoder, meta_clf, label_map, grad_required=False, return_metrics=False
                )
                eval_error = eval_loss.mean()

                if eval_error < best_dev_error:
                    logger.info("Found new best model!")
                    best_dev_error = eval_error
                    save(meta_clf, opt, args.config_path, iteration, meta_encoder if config.finetune_enc else None)
                    save_dist("best_dist.npy")
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                    if patience_ctr == config.patience:
                        logger.info("Ran out of patience. Stopping training early...")
                        patience_over = True
                        break
                dev_iteration_error = 0.
                train_iteration_error = 0.

        if config.train_type != "metabase" and iteration % 10 == 0:
            save_dist("dist.npy")
        if patience_over:
            break

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

    if not config.finetune_enc:
        for param in bert_model.parameters():
            param.requires_grad = False
        extra = []
    else:
        extra = list(bert_model.named_parameters())

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in list(clf_head.named_parameters()) + extra if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in list(clf_head.named_parameters()) + extra if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    opt = AdamW(optimizer_grouped_parameters, eps=1e-8, lr=config.outer_lr)

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
            bert_model.train()
            clf_head.train()
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
                bert_model.eval()
                clf_head.eval()
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
        dev_set = [data_class(p, config.max_seq_length, config.model_type, dev_max_examples) for p in tqdm(dev_paths)]
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

    assert not (args.load_from and hasattr(config, "encoder_ckpt"))
    if args.load_from:
        logger.info(f"Resuming training with weights from {args.load_from}")
        utils.set_savedir_name(args.load_from.split("/")[-1])
        head_path = os.path.join(args.load_from, "best_model.th")
        encoder_path = os.path.join(args.load_from, "best_encoder.th")
        clf_head.load_state_dict(torch.load(head_path))
        if os.path.isfile(encoder_path):
            bert_model.load_state_dict(torch.load(encoder_path))
    if hasattr(config, "encoder_ckpt"):
        load_path = config.encoder_ckpt
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
