import numpy as np
import torch


class RegressionTasksSinusoidal:
    """
    Same regression task as in Finn et al. 2017 (MAML)
    """

    def __init__(self, split, skew=False):
        self.num_inputs = 1
        self.num_outputs = 1

        self.skew = skew
        self.split = split

        self.atoms = 1000
        self.amplitude_range = self._skew(np.linspace(0.1, 5.0, num=self.atoms))
        self.phase_range = self._skew(np.linspace(0, np.pi, num=self.atoms))

        self.input_range = [-5, 5]

    def get_input_range(self, size=100):
        return torch.linspace(self.input_range[0], self.input_range[1], steps=size).unsqueeze(1)

    def sample_inputs(self, batch_size, *args, **kwargs):
        inputs = torch.rand((batch_size, self.num_inputs))
        inputs = inputs * (self.input_range[1] - self.input_range[0]) + self.input_range[0]
        return inputs

    def _skew(self, prange):
        if self.skew:
            if self.split != "test":
                prange = prange[self.atoms // 2 :]
            else:
                prange = prange[: self.atoms // 2]
        return prange

    def sample_task(self):
        amplitude = np.random.choice(self.amplitude_range)
        phase = np.random.choice(self.phase_range)
        return self.get_target_function(amplitude, phase)

    @staticmethod
    def get_target_function(amplitude, phase):
        def target_function(x):
            if isinstance(x, torch.Tensor):
                return torch.sin(x - phase) * amplitude
            else:
                return np.sin(x - phase) * amplitude

        return target_function

    def sample_tasks(self, num_tasks, return_specs=False, task_idxs=None):
        if task_idxs is None:
            amplitude = np.random.choice(self.amplitude_range, size=num_tasks)
            phase = np.random.choice(self.phase_range, size=num_tasks)
        else:
            amplitude_idxs, phase_idxs = task_idxs
            amplitude = self.amplitude_range[amplitude_idxs]
            phase = self.phase_range[phase_idxs]

        target_functions = []
        for i in range(num_tasks):
            target_functions.append(self.get_target_function(amplitude[i], phase[i]))

        if return_specs:
            return target_functions, amplitude, phase
        return target_functions

    def sample_datapoints(self, batch_size):
        """
        Sample random input/output pairs (e.g. for training an orcale)
        :param batch_size:
        :return:
        """

        amplitudes = torch.Tensor(np.random.uniform(self.amplitude_range[0], self.amplitude_range[1], batch_size))
        phases = torch.Tensor(np.random.uniform(self.phase_range[0], self.phase_range[1], batch_size))

        inputs = torch.rand((batch_size, self.num_inputs))
        inputs = inputs * (self.input_range[1] - self.input_range[0]) + self.input_range[0]
        inputs = inputs.view(-1)

        outputs = torch.sin(inputs - phases) * amplitudes
        outputs = outputs.unsqueeze(1)

        return torch.stack((inputs, amplitudes, phases)).t(), outputs
