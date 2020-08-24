import os
import json
import argparse
import torch
import logging
import torch.nn as nn
import numpy as np

from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.data.data_collator import DefaultDataCollator

from typing import Dict, NamedTuple, Optional
from seqeval.metrics import f1_score, precision_score, recall_score
from data_utils import PosDataset, Split, get_data_config, get_labels
from simple_tagger import BERT, Classifier, get_model_config

import learn2learn as l2l

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _BatchedDataset(torch.utils.data.Dataset):
    def __init__(self, batched):
        self.input_ids = [s for s in batched[0][0]]
        self.attention_mask = [s for s in batched[0][1]]
        self.token_type_ids = [s for s in batched[0][2]]
        self.labels = [s for s in batched[0][3]]
        self.class_index = [y for y in batched[1]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.input_ids[idx],
            self.attention_mask[idx],
            self.token_type_ids[idx],
            self.labels[idx],
            self.class_index[idx],
        )


def str_to_tensor(x):
    return torch.tensor(list(map(int, x.split(" "))))


def compute_loss(task, bert_model, learner, batch_size):
    loss = 0.0
    for (
        batch,
        (input_ids, attention_mask, token_type_ids, labels, class_index),
    ) in enumerate(
        torch.utils.data.DataLoader(
            _BatchedDataset(task), batch_size=batch_size, shuffle=True, num_workers=0
        )
    ):

        input_ids = torch.stack([str_to_tensor(x) for x in input_ids]).to(DEVICE)
        attention_mask = torch.stack([str_to_tensor(x) for x in attention_mask]).to(
            DEVICE
        )
        token_type_ids = torch.stack([str_to_tensor(x) for x in token_type_ids]).to(
            DEVICE
        )
        labels = torch.stack([str_to_tensor(x) for x in labels]).to(DEVICE)

        # print("Batch:", batch)
        # print("Input_IDs:", input_ids)
        # print("Attention Mask:", attention_mask)
        # print("Token Type IDs:", token_type_ids)
        # print("Labels:", labels)
        # print("Class_Index:", class_index)
        with torch.no_grad():
            bert_output = bert_model(input_ids, attention_mask, token_type_ids)
        #bert_output = bert_output.to(DEVICE)
        output = learner(bert_output, labels=labels, attention_mask=attention_mask)
        # TODO: confirm that this loss is the mean across all elements of the batch
        curr_loss = output[0]
        loss += curr_loss

    loss /= (batch + 1)
    return loss


def get_model_save_path():
    return os.path.join(
        model_config["output_dir"],
        "posbert_i_{}_o_{}_s_{}_d_{}".format(model_config["inner_lr"], model_config["outer_lr"], model_config["shots"],
            args.datasets),
    )


def save(model, optimizer, last_epoch):
    save_dir = get_model_save_path()
    os.makedirs(save_dir, exist_ok=True)
    logger.info("Saving model checkpoint to %s", save_dir)
    # save model config as well
    with open(os.path.join(save_dir, "model_config.json"), "w") as f:
        f.write(json.dumps(model_config, indent=2))
    # save model weigths
    torch.save(model.state_dict(), os.path.join(save_dir, "best_model.th"))
    # save training state if required
    torch.save(
        {"optimizer": optimizer.state_dict(), "last_epoch": last_epoch,},
        os.path.join(save_dir, "optim.th"),
    )


def init_args():
    parser = argparse.ArgumentParser(
        description="Train POS tagging on various UD datasets"
    )
    parser.add_argument("--datasets", help="Dataset to train on")
    return parser.parse_args()


if __name__ == "__main__":
    args = init_args()
    data_config = get_data_config()
    model_config = get_model_config()

    datasets = args.datasets.split(",")

    dataset_paths = []
    for dataset in datasets:
        dataset_path = data_config[f"{dataset}_path"]
        dataset_paths.append(dataset_path)

    # for reproducibility
    torch.manual_seed(model_config["seed"])
    np.random.seed(model_config["seed"])

    tokenizer = AutoTokenizer.from_pretrained(model_config["model_type"])

    labels = []
    train_datasets = []
    test_datasets = []
    dev_datasets = []

    class_index = 0

    # load individual datasets
    for dataset_path in dataset_paths:
        label = get_labels(dataset_path)
        labels.append(label)

        # load datasets
        train_dataset = PosDataset(
            class_index,
            dataset_path,
            label,
            tokenizer,
            model_config["model_type"],
            model_config["max_seq_length"],
            mode=Split.train,
        )
        train_datasets.append(train_dataset)

        dev_dataset = PosDataset(
            class_index,
            dataset_path,
            label,
            tokenizer,
            model_config["model_type"],
            model_config["max_seq_length"],
            mode=Split.dev,
        )

        dev_datasets.append(dev_dataset)

        test_dataset = PosDataset(
            class_index,
            dataset_path,
            label,
            tokenizer,
            model_config["model_type"],
            model_config["max_seq_length"],
            mode=Split.test,
        )

        test_datasets.append(test_dataset)
        class_index += 1

    # concatenate individual datasets into a single dataset
    combined_train_dataset = ConcatDataset(train_datasets)
    combined_dev_dataset = ConcatDataset(dev_datasets)
    combined_test_dataset = ConcatDataset(test_datasets)

    # convert to metadataset which is suitable for sampling tasks in an episode
    train_dataset = l2l.data.MetaDataset(combined_train_dataset)
    dev_dataset = l2l.data.MetaDataset(combined_train_dataset)
    test_dataset = l2l.data.MetaDataset(combined_train_dataset)

    # shots = number of examples per task, ways = number of classes per task
    shots = model_config["shots"]
    ways = model_config["ways"]

    # create task generators
    train_gen = l2l.data.TaskDataset(
        train_dataset,
        num_tasks=20000,
        task_transforms=[
            l2l.data.transforms.FusedNWaysKShots(train_dataset, n=ways, k=shots),
            l2l.data.transforms.LoadData(train_dataset),
            # l2l.data.transforms.RemapLabels(train_dataset)
        ],
    )

    dev_gen = l2l.data.TaskDataset(
        dev_dataset,
        num_tasks=20000,
        task_transforms=[
            l2l.data.transforms.FusedNWaysKShots(dev_dataset, n=ways, k=shots),
            l2l.data.transforms.LoadData(dev_dataset),
            # l2l.data.transforms.RemapLabels(train_dataset)
        ],
    )

    test_gen = l2l.data.TaskDataset(
        test_dataset,
        num_tasks=20000,
        task_transforms=[
            l2l.data.transforms.FusedNWaysKShots(test_dataset, n=ways, k=shots),
            l2l.data.transforms.LoadData(test_dataset),
            # l2l.data.transforms.RemapLabels(train_dataset)
        ],
    )

    # for tensorboard logging
    tb_writer = SummaryWriter(os.path.join(get_model_save_path(), "logs"))

    # define the postagger model
    bert_model = BERT(model_config["model_type"])
    bert_model.eval()
    bert_model = bert_model.to(DEVICE)

    postagger = Classifier(len(labels[0]), model_config["hidden_dropout_prob"], bert_model.get_hidden_size())
    postagger.to(DEVICE)

    num_epochs = model_config["num_epochs"]
    if model_config["is_fomaml"]:
        meta_model = l2l.algorithms.MAML(
            postagger, lr=model_config["inner_lr"], first_order=True
        )
    else:
        meta_model = l2l.algorithms.MAML(postagger, lr=model_config["inner_lr"])
    # outer loop optimizer
    opt = optim.Adam(meta_model.parameters(), lr=model_config["outer_lr"])
    tqdm_bar = tqdm(range(num_epochs))
    num_episodes = model_config["num_episodes"]
    task_bs = model_config["task_batch_size"]
    inner_loop_steps = model_config["inner_loop_steps"]

    best_dev_error = np.inf
    for iteration in tqdm_bar:  # iterations loop
        dev_iteration_error = 0.0
        train_iteration_error = 0.0
        for episode in range(num_episodes):  # episode loop = task loop
            # print("Episode:", episode)
            # clone the meta model for use in inner loop. Back-propagating losses on the cloned module will populate the
            # buffers of the original module
            learner = meta_model.clone()
            # sample train and validation tasks
            train_task, dev_task = train_gen.sample(), dev_gen.sample()

            # Inner Loop
            for step in range(inner_loop_steps):  # inner loop
                # print("Step:", step)
                train_error = compute_loss(
                    train_task, bert_model, learner, batch_size=task_bs
                )
                grads = torch.autograd.grad(
                    train_error,
                    learner.parameters(),
                    create_graph=True,
                    allow_unused=True,
                )
                l2l.algorithms.maml_update(learner, model_config["inner_lr"], grads)
                # print("Train Error:", train_error)

            # Compute validation loss / query loss
            dev_error = compute_loss(dev_task, bert_model, learner, batch_size=task_bs)
            dev_iteration_error += dev_error
            #print("Dev_Error:", dev_error)
            train_iteration_error += train_error

        # average the validation and train loss over all tasks
        dev_iteration_error /= num_episodes
        train_iteration_error /= num_episodes
        tqdm_bar.set_description(
            "Validation Loss : {:.3f}".format(dev_iteration_error.item())
        )
        tb_writer.add_scalar("validation_loss", dev_iteration_error, iteration)
        tb_writer.add_scalar("training loss", train_iteration_error, iteration)

        # Outer Loop
        opt.zero_grad()
        # backprop validation error in the outer loop
        dev_iteration_error.backward()
        opt.step()
        meta_model.zero_grad()

        logger.info(
            f"Finished iteration {iteration+1} with avg. training loss: {train_iteration_error}"
        )
        logger.info(
            f"Finished iteration {iteration+1} with avg. validation loss: {dev_iteration_error}"
        )

        if dev_iteration_error < best_dev_error:
            logger.info("Found new best model!")
            best_dev_error = dev_iteration_error
            save(meta_model, opt, iteration)
            best_state_dict = meta_model.state_dict()
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr == model_config["patience"]:
                logger.info("Ran out of patience. Stopping training early...")
                break
    logger.info(f"Best validation loss = {best_dev_error}")
