"""
Neural network models for the regression experiments
"""
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.categorical import Categorical


class Constrainer(nn.Module):
    def __init__(self,
                 n_tasks,
                 threshold=1.
                 ):
        """
        :param n_tasks:             the number of training tasks
        """
        super(Constrainer, self).__init__()
        self.n_tasks = n_tasks
        self.threshold = threshold
        self.tau_amplitude = nn.Parameter(torch.full((n_tasks, ), 1.))
        self.tau_phase = nn.Parameter(torch.full((n_tasks, ), 1.))
        self.softplus = nn.Softplus()

    def forward(self, amplitude_idxs, phase_idxs, losses):
        lambdas = self.softplus(self.tau_amplitude[amplitude_idxs]) * self.softplus(self.tau_phase[phase_idxs])
        aux_loss = (losses - self.threshold) * lambdas
        return aux_loss


class TaskSampler(nn.Module):
    def __init__(self,
                 n_tasks,
                 ):
        """
        :param n_tasks:             the number of training tasks
        """
        super(TaskSampler, self).__init__()
        self.n_tasks = n_tasks
        self.tau_amplitude = nn.Parameter(torch.full((n_tasks, ), 1. / n_tasks))
        self.tau_phase = nn.Parameter(torch.full((n_tasks, ), 1. / n_tasks))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, n_tasks):
        amplitude_distribution = Categorical(logits=self.tau_amplitude)
        phase_distribution = Categorical(logits=self.tau_phase)
        amplitude_idxs = amplitude_distribution.sample(sample_shape=(n_tasks, ))
        phase_idxs = phase_distribution.sample(sample_shape=(n_tasks, ))
        task_probs = self.softmax(self.tau_amplitude)[amplitude_idxs] * self.softmax(self.tau_phase)[phase_idxs]
        return (amplitude_idxs, phase_idxs), task_probs


class MamlModel(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_weights,
                 device,
                 ):
        """
        :param n_inputs:            the number of inputs to the network
        :param n_outputs:           the number of outputs of the network
        :param n_weights:           for each hidden layer the number of weights
        """
        super(MamlModel, self).__init__()

        # initialise lists for biases and fully connected layers
        self.weights = []
        self.biases = []

        # add one
        self.nodes_per_layer = n_weights + [n_outputs]

        # set up the shared parts of the layers
        prev_n_weight = n_inputs
        for i in range(len(self.nodes_per_layer)):
            w = torch.Tensor(size=(prev_n_weight, self.nodes_per_layer[i])).to(device)
            w.requires_grad = True
            self.weights.append(w)
            b = torch.Tensor(size=[self.nodes_per_layer[i]]).to(device)
            b.requires_grad = True
            self.biases.append(b)
            prev_n_weight = self.nodes_per_layer[i]

        self._reset_parameters()

    def _reset_parameters(self):
        for i in range(len(self.nodes_per_layer)):
            stdv = 1. / math.sqrt(self.nodes_per_layer[i])
            self.weights[i].data.uniform_(-stdv, stdv)
            self.biases[i].data.uniform_(-stdv, stdv)

    def forward(self, x):

        for i in range(len(self.weights) - 1):
            x = F.relu(F.linear(x, self.weights[i].t(), self.biases[i]))
        y = F.linear(x, self.weights[-1].t(), self.biases[-1])

        return y
