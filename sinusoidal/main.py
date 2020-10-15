"""
Regression experiment using MAML
"""
import copy
import os
import time

import numpy as np
import scipy.stats as st
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim

import arguments
import utils
import tasks_sine
from logger import Logger
from maml_model import MamlModel, TaskSampler


def run(args, log_interval=5000, rerun=False):

    # see if we already ran this experiment
    code_root = os.path.dirname(os.path.realpath(__file__))
    exp_dir = utils.get_path_from_args(args) if not args.output_dir else args.output_dir
    path = '{}/results/{}'.format(code_root, exp_dir)
    if not os.path.isdir(path):
        os.makedirs(path)

    if os.path.exists(os.path.join(path, 'logs.pkl')) and not rerun:
        return utils.load_obj(os.path.join(path, 'logs'))

    start_time = time.time()

    # correctly seed everything
    utils.set_seed(args.seed)

    # --- initialise everything ---
    task_family_train = tasks_sine.RegressionTasksSinusoidal('train', args.skew_task_distribution)
    task_family_valid = tasks_sine.RegressionTasksSinusoidal('valid', args.skew_task_distribution)

    # initialise network
    model_inner = MamlModel(task_family_train.num_inputs,
                            task_family_train.num_outputs,
                            n_weights=args.num_hidden_layers,
                            device=args.device
                            ).to(args.device)
    model_outer = copy.deepcopy(model_inner)
    if args.detector == "minimax":
        task_sampler = TaskSampler(task_family_train.atoms // (2 if args.skew_task_distribution else 1))

    # intitialise meta-optimiser
    meta_optimiser = optim.Adam(model_outer.weights + model_outer.biases,
                                args.lr_meta)

    # initialise loggers
    logger = Logger()
    logger.best_valid_model = copy.deepcopy(model_outer)

    for i_iter in range(args.n_iter):

        # copy weights of network
        copy_weights = [w.clone() for w in model_outer.weights]
        copy_biases = [b.clone() for b in model_outer.biases]

        # get all shared parameters and initialise cumulative gradient
        meta_gradient = [0 for _ in range(len(copy_weights + copy_biases) + (2 if args.detector == "minimax" else 0))]

        # sample tasks
        if args.detector == "minimax":
            task_idxs, task_probs = task_sampler(args.tasks_per_metaupdate)
        else:
            task_idxs = None

        target_functions = task_family_train.sample_tasks(args.tasks_per_metaupdate, task_idxs=task_idxs)

        for t in range(args.tasks_per_metaupdate):

            # reset network weights
            model_inner.weights = [w.clone() for w in copy_weights]
            model_inner.biases = [b.clone() for b in copy_biases]

            # get data for current task
            train_inputs = task_family_train.sample_inputs(args.k_meta_train).to(args.device)

            for _ in range(args.num_inner_updates):

                # make prediction using the current model
                outputs = model_inner(train_inputs)

                # get targets
                targets = target_functions[t](train_inputs)

                # ------------ update on current task ------------

                # compute loss for current task
                loss_task = F.mse_loss(outputs, targets)

                # compute the gradient wrt current model
                params = [w for w in model_inner.weights] + [b for b in model_inner.biases]
                grads = torch.autograd.grad(loss_task, params, create_graph=True, retain_graph=True)

                # make an update on the inner model using the current model (to build up computation graph)
                for i in range(len(model_inner.weights)):
                    if not args.first_order:
                        model_inner.weights[i] = model_inner.weights[i] - args.lr_inner * grads[i]
                    else:
                        model_inner.weights[i] = model_inner.weights[i] - args.lr_inner * grads[i].detach()
                for j in range(len(model_inner.biases)):
                    if not args.first_order:
                        model_inner.biases[j] = model_inner.biases[j] - args.lr_inner * grads[i + j + 1]
                    else:
                        model_inner.biases[j] = model_inner.biases[j] - args.lr_inner * grads[i + j + 1].detach()

            # ------------ compute meta-gradient on test loss of current task ------------

            # get test data
            test_inputs = task_family_train.sample_inputs(args.k_meta_test).to(args.device)

            # get outputs after update
            test_outputs = model_inner(test_inputs)

            # get the correct targets
            test_targets = target_functions[t](test_inputs)

            # compute loss (will backprop through inner loop)
            if args.detector == "minimax":
                importance = task_probs[t]
            else:
                importance = 1. / args.tasks_per_metaupdate
            loss_meta = F.mse_loss(test_outputs, test_targets) * importance

            # compute gradient w.r.t. *outer model*
            outer_params = model_outer.weights + model_outer.biases
            if args.detector == "minimax":
                outer_params += [task_sampler.tau_amplitude, task_sampler.tau_phase]
            task_grads = torch.autograd.grad(loss_meta, outer_params, retain_graph=(True if args.detector == "minimax" else False))
            for i in range(len(outer_params)):
                meta_gradient[i] += task_grads[i].detach()

        # ------------ meta update ------------

        meta_optimiser.zero_grad()
        # print(meta_gradient)

        # assign meta-gradient
        for i in range(len(model_outer.weights)):
            model_outer.weights[i].grad = meta_gradient[i]
            meta_gradient[i] = 0
        for j in range(len(model_outer.biases)):
            model_outer.biases[j].grad = meta_gradient[i + j + 1]
            meta_gradient[i + j + 1] = 0
        if args.detector == "minimax":
            task_sampler.tau_amplitude.grad = meta_gradient[i + j + 2]
            task_sampler.tau_phase.grad = meta_gradient[i + j + 3]
            meta_gradient[i + j + 2] = 0
            meta_gradient[i + j + 3] = 0

        # do update step on outer model
        meta_optimiser.step()

        # ------------ logging ------------

        if i_iter % log_interval == 0:

            # evaluate on training set
            losses = eval(args, copy.copy(model_outer), task_family=task_family_train,
                                        num_updates=args.num_inner_updates)
            loss_mean, loss_conf = utils.get_stats(np.array(losses))
            logger.train_loss.append(loss_mean)
            logger.train_conf.append(loss_conf)

            # evaluate on valid set
            losses = eval(args, copy.copy(model_outer), task_family=task_family_valid,
                                        num_updates=args.num_inner_updates)
            loss_mean, loss_conf = utils.get_stats(np.array(losses))
            logger.valid_loss.append(loss_mean)
            logger.valid_conf.append(loss_conf)

            # save best model
            if logger.valid_loss[-1] == np.min(logger.valid_loss):
                print('saving best model at iter', i_iter)
                logger.best_valid_model = copy.copy(model_outer)

            # save logging results
            utils.save_obj(logger, os.path.join(path, 'logs'))

            # print current results
            logger.print_info(i_iter, start_time)
            start_time = time.time()

    return logger


def eval(args, model, task_family, num_updates, n_tasks=100, lr_inner=None, k_shot=None):
    if lr_inner is None:
        lr_inner = args.lr_inner
    if k_shot is None:
        k_shot = args.k_shot_eval

    # copy weights of network
    copy_weights = [w.clone() for w in model.weights]
    copy_biases = [b.clone() for b in model.biases]

    # get the task family (with infinite number of tasks)
    input_range = task_family.get_input_range().to(args.device)

    # logging
    losses = [[] for _ in range(n_tasks)]

    # --- inner loop ---

    for t in range(n_tasks):

        # reset network weights
        model.weights = [w.clone() for w in copy_weights]
        model.biases = [b.clone() for b in copy_biases]

        # sample a task
        target_function = task_family.sample_task()

        # get data for current task
        curr_inputs = task_family.sample_inputs(k_shot).to(args.device)
        curr_targets = target_function(curr_inputs)

        # ------------ update on current task ------------

        for _ in range(1, num_updates + 1):

            curr_outputs = model(curr_inputs)

            # compute loss for current task
            task_loss = F.mse_loss(curr_outputs, curr_targets)

            # update task parameters
            params = [w for w in model.weights] + [b for b in model.biases]
            grads = torch.autograd.grad(task_loss, params)

            for i in range(len(model.weights)):
                model.weights[i] = model.weights[i] - lr_inner * grads[i].detach()
            for j in range(len(model.biases)):
                model.biases[j] = model.biases[j] - lr_inner * grads[i + j + 1].detach()

            # compute true loss on entire input range
            losses[t].append(F.mse_loss(model(input_range), target_function(input_range)).detach().item())

    # reset network weights
    model.weights = [w.clone() for w in copy_weights]
    model.biases = [b.clone() for b in copy_biases]

    return losses

def test(args):
    # see if we already ran this experiment
    code_root = os.path.dirname(os.path.realpath(__file__))
    exp_dir = utils.get_path_from_args(args) if not args.output_dir else args.output_dir
    path = '{}/results/{}'.format(code_root, exp_dir)
    assert os.path.isdir(path)
    task_family_test = tasks_sine.RegressionTasksSinusoidal('test', args.skew_task_distribution)
    best_valid_model = utils.load_obj(os.path.join(path, 'logs')).best_valid_model
    k_shots = [5, 10, 20, 40]
    df = []
    for k_shot in k_shots:
        losses = np.array(eval(args, copy.copy(best_valid_model), task_family=task_family_test,
                          num_updates=10, lr_inner=0.01, n_tasks=1000, k_shot=k_shot))
        for grad_step, task_losses in enumerate(losses.T, 1):
            new_rows = [[k_shot, grad_step, tl] for tl in task_losses]
            df.extend(new_rows)

    df = pd.DataFrame(df, columns=['k_shot', 'grad_steps', 'loss'])
    df.to_pickle(os.path.join(path, 'res.pkl'))
    utils.plot_df(df, path)


if __name__ == "__main__":
    args = arguments.parse_args()
    run(args, log_interval=100, rerun=False)
    test(args)
