import time
import math
import torch
import torch.autograd as autograd

from .cgd_utils import conjugate_gradient, Hvp_vec, zero_grad


class ALCGD(object):
    def __init__(self, max_params, min_params,
                 lr_max=1e-3, lr_min=1e-3,
                 momentum=(0.9, 0.999), eps=1e-8,
                 device=torch.device('cpu')):
        self.max_params = list(max_params)
        self.min_params = list(min_params)
        num_max_params = sum(p.numel() for p in max_params)
        num_min_params = sum(p.numel() for p in min_params)
        self.state = {'lr_max': lr_max, 'lr_min': lr_min,
                      'momentum': momentum, 'eps': eps,
                      'step': 0, 'old_max': None, 'old_min': None,
                      'exp_avg_max': torch.zeros((num_max_params, )).to(device=device),
                      'exp_avg_min': torch.zeros((num_min_params, )).to(device=device),
                      'exp_avg_sq_max': torch.zeros((num_max_params, )).to(device=device),
                      'exp_avg_sq_min': torch.zeros((num_min_params, )).to(device=device)}

    def zero_grad(self):
        zero_grad(self.max_params)
        zero_grad(self.min_params)

    def state_dict(self):
        return self.state

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)
        print('Load state: {}'.format(state_dict))

    def set_lr(self, lr_max, lr_min):
        self.state.update({'lr_max': lr_max, 'lr_min': lr_min})
        print('Maximizing side learning rate: {:.4f}\n '
              'Minimizing side learning rate: {:.4f}'.format(lr_max, lr_min))

    def step(self, loss):
        """
            update rules:
            p_x = grad_x - lr * h_xy-v_y
            p_y = grad_y + lr * h_yx-v_x

            x = x + lr * p_x
            y = y - lr * p_y
        """
        lr_max = self.state['lr_max']
        lr_min = self.state['lr_min']
        time_step = self.state['step'] + 1
        self.state['step'] = time_step

        grad_x = autograd.grad(loss, self.max_params, allow_unused=True, create_graph=True, retain_graph=True)
        grad_x_vec = torch.cat([g.contiguous().view(-1) for g in grad_x])
        grad_y = autograd.grad(loss, self.min_params, allow_unused=True, create_graph=True, retain_graph=True)
        grad_y_vec = torch.cat([g.contiguous().view(-1) for g in grad_y])
        grad_x_vec_d = grad_x_vec.clone().detach()
        grad_y_vec_d = grad_y_vec.clone().detach()
        hvp_x_vec = Hvp_vec(grad_y_vec, self.max_params, grad_y_vec_d,
                            retain_graph=True)  # h_xy * d_y
        hvp_y_vec = Hvp_vec(grad_x_vec, self.min_params, grad_x_vec_d,
                            retain_graph=False)  # h_yx * d_x

        cg_x = torch.add(grad_x_vec_d, - lr_min * hvp_x_vec).detach_()
        cg_y = torch.add(grad_y_vec_d, lr_max * hvp_y_vec).detach_()

        beta1, beta2 = self.state['momentum']
        exp_avg_max, exp_avg_min = self.state['exp_avg_max'], self.state['exp_avg_min']
        exp_avg_sq_max, exp_avg_sq_min = self.state['exp_avg_sq_max'], self.state['exp_avg_sq_min']

        bias_correction1 = 1 - beta1 ** time_step
        bias_correction2 = 1 - beta2 ** time_step
        exp_avg_max.mul_(beta1).add_(1 - beta1, cg_x)
        exp_avg_min.mul_(beta1).add_(1 - beta1, cg_y)
        exp_avg_sq_max.mul_(beta2).addcmul_(1 - beta2, cg_x, cg_x)
        exp_avg_sq_min.mul_(beta2).addcmul_(1 - beta2, cg_y, cg_y)
        denom_max = (exp_avg_sq_max.sqrt() / math.sqrt(bias_correction2)).add_(self.state['eps'])
        denom_min = (exp_avg_sq_min.sqrt() / math.sqrt(bias_correction2)).add_(self.state['eps'])
        grad_max = exp_avg_max / denom_max
        grad_min = exp_avg_min / denom_min
        lr_max /= bias_correction1
        lr_min /= bias_correction1

        index = 0
        for p in self.max_params:
            p.data.add_(lr_max * grad_max[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        assert index == cg_x.numel(), 'Maximizer CG size mismatch'
        index = 0
        for p in self.min_params:
            p.data.add_(- lr_min * grad_min[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        assert index == cg_y.numel(), 'Minimizer CG size mismatch'