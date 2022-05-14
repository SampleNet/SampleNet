import geomloss
import numpy as np
import torch

class NeuralNetworkTrainer:
    def __init__(self, model,
                 num_training_iterations=10000,
                 lr=1e-2,
                 batch_size=256):
        self.model = model.train()
        self.which_loss = None
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.num_iterations = num_training_iterations
        self.batch_size = batch_size

        # Early stopping to avoid overfitting
        self.early_stopping_patience = None

        # Beta_NLL params
        self.beta = 0.0

    def compute_loss(self, network_output, y_data, which_loss):
        estimated_mean, estimated_variance = network_output
        if which_loss == 'rmse':
            loss = ((estimated_mean - y_data) ** 2)
        else:
            estimated_normal_dists = torch.distributions.Normal(loc=estimated_mean,
                                                                scale=estimated_variance.sqrt())
            if which_loss == 'nll':
                loss = -estimated_normal_dists.log_prob(y_data)
                if self.beta > 0.0:
                    loss = (loss * (estimated_variance.detach() ** self.beta))
            else:
                raise NotImplementedError("Loss choice not valid or Loss not implemented!")
        return loss

    def train_model(self, x_data, y_data,
                    learn_mean_first=True,
                    which_loss='nll',
                    beta=0.0):
        """
        Train a variance network model.

        :param x_data: (self.num_data_samples x 1): torch tensor containing input features.
        :param y_data: (self.num_data_samples): torch tensor containing output pairs.
        :param learn_mean_first: (boolean): Learns the mean for the first 1/2 of number of iterations, then learns the
        mean and the variance jointly for the remaining number of iterations.
        :param which_loss: (str): one of 'rmse', 'nll', 'es', ...
        :param beta: (float) beta_nll beta parameter

        :return: model: trained model
        """
        self.beta = beta
        self.which_loss = which_loss

        for i in range(self.num_iterations):
            self.optimizer.zero_grad()
            network_output = self.model(x_data)

            if (i < self.num_iterations / 2) & learn_mean_first:
                which_loss = 'rmse'
            else:
                which_loss = self.which_loss

            loss = self.compute_loss(network_output, y_data, which_loss)

            loss = loss.mean()
            loss.backward()

            if i % 500 == 0:
                print('Iter {0}/{1}, Loss {2}'.format(i, self.num_iterations, loss.item()))

            self.optimizer.step()
        return self.model


class SampleNetTrainer(NeuralNetworkTrainer):
    def __init__(self, model, num_training_iterations=10000,
                 lr=1e-2,
                 mmd_blur=0.05,
                 sinkhorn_blur=0.05,
                 sinkhorn_p_norm=2,
                 sinkhorn_loss_weight=1.0,
                 minibatch_sample_size=100,
                 minibatch_max_num_runs=1,
                 prior_type='uniform'):

        super(SampleNetTrainer, self).__init__(model, num_training_iterations, lr)

        self.energy_distance = geomloss.SamplesLoss(loss='energy')
        self.gaussian_mmd = geomloss.SamplesLoss(loss='gaussian', blur=mmd_blur)

        self.sinkhorn_divergence = geomloss.SamplesLoss(loss="sinkhorn",
                                                        p=sinkhorn_p_norm,
                                                        blur=sinkhorn_blur,
                                                        scaling=0.1)
        self.sinkhorn_loss_weight = sinkhorn_loss_weight
        self.minibatch_sample_size = minibatch_sample_size
        self.minibatch_max_num_runs = minibatch_max_num_runs

        # Update model parameters
        self.model.prior_type = prior_type

    def minibatch_sample_loss(self, samples_1,
                              gt_samples,
                              which_loss):
        """

        :param samples_1:
        :param gt_samples: ground truth samples
        :param which_loss: could be 'es' for energy score, 'mmd' for maximum mean discrepancy, and 'ot' for sinkhorn div
        :return:
        """
        num_runs = np.maximum(1, self.minibatch_max_num_runs)
        loss = 0.0
        for _ in range(num_runs):
            random_idxs = torch.rand(samples_1.size(0), samples_1.size(1), samples_1.size(2)).argsort(dim=1)
            random_idxs = random_idxs[:, :self.minibatch_sample_size, :].to(samples_1.device)
            minibatch_samples_1 = torch.gather(samples_1, 1, random_idxs)
            if which_loss == 'es':
                loss += self.energy_distance(minibatch_samples_1, gt_samples)
            elif which_loss == 'mmd':
                loss += self.gaussian_mmd(minibatch_samples_1, gt_samples)
            else:
                loss += self.sinkhorn_divergence(minibatch_samples_1, gt_samples)
        return loss / num_runs

    def compute_loss(self,
                     network_output,
                     y_data,
                     which_loss):

        if self.sinkhorn_loss_weight > 0.0:
            use_ot_constraint = True
        else:
            use_ot_constraint = False

        samples = network_output[-1]
        y_data = self.model.normalize_data(y_data)
        if which_loss == 'rmse':
            estimated_mean = samples.mean(1, keepdim=False)
            if len(y_data.shape) == 3:
                estimated_mean = estimated_mean.unsqueeze(1)
            loss = ((estimated_mean - y_data) ** 2).mean()
        else:
            if which_loss in ['es', 'mmd']:
                if len(y_data.shape) == 2:
                    y_data = y_data.unsqueeze(1)

            if which_loss == 'nll':
                estimated_mean = samples.mean(1, keepdim=False)
                estimated_variance = samples.var(1, keepdim=False)

                estimated_normal_dists = torch.distributions.Normal(loc=estimated_mean,
                                                                    scale=torch.sqrt(estimated_variance))
                loss = -estimated_normal_dists.log_prob(y_data).mean()
                if self.beta > 0.0:
                    loss = (loss * (estimated_variance.detach() ** self.beta)).mean()
            elif which_loss == 'es':
                loss = self.minibatch_sample_loss(samples,
                                                  y_data,
                                                  which_loss='es').mean()

            elif which_loss == 'mmd':
                loss = self.minibatch_sample_loss(samples,
                                                  y_data,
                                                  which_loss='mmd').mean()
            else:
                raise NotImplementedError("Loss choice not valid or Loss not implemented!")

        if use_ot_constraint:
            samples = self.model.normalize_samples(samples)

            prior_samples = self.model.generate_prior_samples(batch_size=samples.shape[0],
                                                              dim=samples.shape[-1]).to(
                samples.device).type(
                samples.type())

            loss_ot = self.sinkhorn_loss_weight * self.minibatch_sample_loss(samples,
                                                                             prior_samples,
                                                                             which_loss='ot').mean()
        else:
            loss_ot = 0.0
        loss += loss_ot
        return loss