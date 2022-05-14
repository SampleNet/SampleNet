import einops
import numpy as np
import torch


class BaseNetwork(torch.nn.Module):
    """
    Base Network Class. Contains shared components among all other models.
    """

    def __init__(self, n_neurons=50,
                 activation='sigmoid',
                 input_dim=1,
                 output_dim=1,
                 min_var=1e-8):
        """
        :param n_neurons: size of network layers
        :param activation: activation function.
        :param input_dim: Single scalar representing the dimension of input features.
        :param output_dim: Single scalar representing the dimension of the output predicted vector.
        :param min_var: minimum variance. This value is added to the predicted variance to prevent NLL overflow.
        """
        super(BaseNetwork, self).__init__()
        if activation == 'sigmoid':
            act_func = torch.nn.Sigmoid()
        elif activation == 'relu':
            act_func = torch.nn.ReLU()
        elif activation == 'tanh':
            act_func = torch.nn.Tanh()
        elif activation == 'leaky_relu':
            act_func = torch.nn.LeakyReLU()
        elif activation == 'elu':
            act_func = torch.nn.ELU()
        else:
            raise (NotImplementedError,
                   'Activation Function Not Implemented. Needs to be one of: [sigmoid, relu, leaky_relu]')

        self.act_func = act_func
        self.n_neurons = n_neurons
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.min_var = min_var
        self.init_var_offset = np.log(np.exp(1.0 - min_var) - 1.0)
        self.epsilon = torch.finfo(torch.float32).eps


class SampleNet(BaseNetwork):
    """
    Sample generation network.
    """

    def __init__(self, n_neurons=50,
                 n_generated_samples=100,
                 activation='sigmoid',
                 input_dim=1,
                 output_dim=1,
                 min_var=1e-8,
                 minibatch_sample_size=100):
        super(SampleNet, self).__init__(n_neurons=n_neurons,
                                        activation=activation,
                                        input_dim=input_dim,
                                        output_dim=output_dim,
                                        min_var=min_var)
        self.n_generated_samples = n_generated_samples

        self.samples_subnet = torch.nn.Sequential(torch.nn.Linear(self.input_dim, n_neurons),
                                                  self.act_func,
                                                  torch.nn.Linear(n_neurons, self.output_dim * n_generated_samples))

        self.minibatch_sample_size = minibatch_sample_size

    def forward(self, x):
        samples = self.samples_subnet(x).reshape(-1, self.n_generated_samples, self.output_dim)
        return [samples, ]

    @staticmethod
    def normalize_data(y_data):
        return y_data

    def generate_prior_samples(self,
                               batch_size,
                               dim,
                               uniform_low=0.0,
                               uniform_high=1.0,
                               gaussian_loc=0.0,
                               gaussian_scale=1.0):
        if self.prior_type == 'uniform':
            dist = torch.distributions.Uniform(uniform_low, uniform_high)
        elif self.prior_type == 'gaussian':
            dist = torch.distributions.Normal(loc=gaussian_loc, scale=gaussian_scale)
        else:
            raise NotImplementedError("Prior choice not valid or Prior not implemented!")
        prior_samples = dist.sample((self.minibatch_sample_size, dim)).sort(dim=0)[0]
        prior_samples = einops.repeat(prior_samples, 'n d -> b n d', b=batch_size).contiguous()
        return prior_samples

    def normalize_samples(self, samples):
        if self.prior_type == 'uniform':
            samples_min = einops.repeat(samples.min(dim=1)[0], 'b d -> b n d', n=self.n_generated_samples)
            samples_max = einops.repeat(samples.max(dim=1)[0], 'b d -> b n d', n=self.n_generated_samples)
            normalized_samples = (samples - samples_min) / (samples_max - samples_min)
        elif self.prior_type == 'gaussian':
            samples_mean = samples.mean(dim=1, keepdims=True)
            samples_var = samples.var(dim=1, keepdims=True) + self.min_var
            normalized_samples = (samples - samples_mean) / samples_var.sqrt()
        else:
            raise NotImplementedError("Prior choice not valid or Prior not implemented!")
        return normalized_samples

    def inference_forward(self, x, **kwargs):
        self.eval()
        with torch.no_grad():
            samples = self.forward(x)[-1]
            mean = samples.mean(1, keepdim=False)
            var = samples.var(1, keepdim=False)

        return mean.cpu().numpy(), var.cpu().numpy(), samples