import torch
from torch import nn
from torch.autograd import grad

from learn2learn.algorithms.base_learner import BaseLearner
from learn2learn.utils import clone_module, clone_parameters
from learn2learn.algorithms.meta_sgd import meta_sgd_update

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ParamMetaSGD(BaseLearner):
    def __init__(self, model, lr=1.0, first_order=True, lrs=None):
        super(ParamMetaSGD, self).__init__()
        self.module = model
        if lrs is None:
            lrs = nn.ParameterList([nn.Parameter(torch.Tensor([lr]).to(DEVICE)) for p in model.parameters()])
        self.lrs = lrs
        self.first_order = first_order

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def clone(self):
        """
        **Descritpion**
        Akin to `MAML.clone()` but for MetaSGD: it includes a set of learnable fast-adaptation
        learning rates.
        """
        return ParamMetaSGD(clone_module(self.module),
                       lrs=clone_parameters(self.lrs),
                       first_order=self.first_order)

    def adapt(self, loss, first_order=None, retain_graph=False, allow_unused=False):
        """
        **Descritpion**
        Akin to `MAML.adapt()` but for MetaSGD: it updates the model with the learnable
        per-parameter learning rates.
        """
        if first_order is None:
            first_order = self.first_order
        second_order = not first_order
        gradients = grad(loss,
                         self.module.parameters(),
                         retain_graph=second_order or retain_graph,
                         create_graph=second_order,
                         allow_unused=allow_unused)
        self.module = meta_sgd_update(self.module, self.lrs, gradients)
