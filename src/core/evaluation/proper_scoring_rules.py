import torch
import geomloss


class GaussianEvaluator:
    def __init__(self, es_num_samples=1000, num_repeats=5):
        """
        :param es_num_samples: number of samples used to compute energy score. Memory grows quadratically with this
        parameter.

        :param num_repeats: repeats the computation of es num_repeats times and then averages the results. Higher
        values mitigate the stochasticity due to sampling in the computation of the energy score, and provides more
        accurate ranking of predictions with no additional memory overhead.
        """
        self.energy_score = geomloss.SamplesLoss(loss='energy')

        self.num_samples = es_num_samples
        self.num_repeats = num_repeats

    @staticmethod
    def check_dims(y_pred, y_gt, var_pred):
        # Check dimensions
        B, N, dim = y_pred.shape

        assert y_gt.shape[0] == B, "y_pred and y_gt should have same batch size!"
        assert y_gt.shape[2] == dim, "y_pred and y_gt should have same final dimension!"

        assert (var_pred.shape[:-1] == y_pred.shape) & (
                var_pred.shape[-1] == dim), "Wrong shape of var_pred. Shape should be (batch " \
                                            "x " \
                                            "N x dim x dim)!"

        return B, N, dim

    def compute_nll(self, y_pred, y_gt, var_pred, epsilon=torch.finfo().eps):
        """
        Calculates the negative log likelihood given torch tensors as input.
        :param y_pred: (batch x N x dim) The neural network predicted expected value.
        :param y_gt: (batch x N x dim) The groundtruth output.
        :param var_pred: (batch x N x dim x dim) The neural network predicted variance/covariance matrix.
        :param epsilon: (float) value to be added to variance to avoid nll explosion

        :return: nll: (batch x N) scalar value of the negative log likelihood.
        """
        # Check dimensions
        self.check_dims(y_pred, y_gt, var_pred)

        # Compute NLL
        var_pred = var_pred + epsilon
        predictive_distributions = torch.distributions.MultivariateNormal(y_pred, covariance_matrix=var_pred)
        nll = -predictive_distributions.log_prob(y_gt)

        return nll

    def compute_energy_score(self, y_pred, y_gt, var_pred):
        """
        Calculates the exact energy score given torch tensors describing a multivariate Guassian Distribution as input.
        :param y_pred: (batch x N x dim) The neural network predicted expected value.
        :param var_pred: (batch x N x dim x dim) The neural network predicted variance/covariance matrix.
        :param y_gt: (batch x N x dim) The groundtruth output.

        :return: es: (batch x N) energy score for every input-output pair.
        """

        # Check dimensions
        B, N, dim = self.check_dims(y_pred, y_gt, var_pred)

        es_all_batches = []
        # Compute energy score for every batch independently. geomloss code is very memory intensive.
        for b in range(B):
            batch_y_pred = y_pred[b]
            batch_y_gt = y_gt[b].unsqueeze(1)
            batch_var_pred = var_pred[b] + torch.finfo(var_pred.dtype).eps

            # Maximum N 500. If N>500, we have to iterate to make sure we don't run out of memory.
            batch_y_gt_list = torch.split(batch_y_gt, 500, dim=0)
            batch_y_pred_list = torch.split(batch_y_pred, 500, dim=0)
            batch_var_pred_list = torch.split(batch_var_pred, 500, dim=0)

            n_batch_es_list = []
            # Fragment compute to not ever exceed 512 samples. This is because of geomloss memory issues.
            for y_pred_in, y_gt_in, var_pred_in in zip(batch_y_pred_list, batch_y_gt_list, batch_var_pred_list):
                # Create distributions
                predictive_distributions = torch.distributions.MultivariateNormal(y_pred_in,
                                                                                  covariance_matrix=var_pred_in)
                # for more accurate ranking, repeat computation 5 times and average.
                n_batch_es = 0
                for _ in range(self.num_repeats):
                    samples = predictive_distributions.sample((self.num_samples,))
                    samples = samples.permute((1, 0, 2))
                    n_batch_es = n_batch_es + self.energy_score(samples, y_gt_in)
                n_batch_es_list.extend(n_batch_es / self.num_repeats)
            es_all_batches.extend(torch.stack(n_batch_es_list, dim=0).view(1, N))
        es = torch.stack(es_all_batches, dim=0)
        return es


class SampleEvaluator:
    def __init__(self, score='energy', gaussian_std=0.5):
        """
        :param score: Can be either 'energy' or 'mmd'. 'gaussian' results in using the maximum mean discrepancy
        with a gaussian kernel.
        :param gaussian_std: The standard deviation σ of the convolution kernel. Used only if score = 'mmd'.
        """
        if score == 'energy':
            print("Using Energy Score as a metric.")
            self.score = geomloss.SamplesLoss(loss='energy')
        else:
            print("Using MMD as a metric.")
            self.score = geomloss.SamplesLoss(loss='gaussian', blur=gaussian_std)

    @staticmethod
    def check_dims(y_pred, y_gt):
        # Check dimensions
        B, N, M, dim = y_pred.shape

        assert y_gt.shape[0] == B, "y_pred and y_gt should have same batch size at dim=0!"
        assert y_gt.shape[1] == N, "y_pred and y_gt should have same number of elements at dim=1!"
        assert y_gt.shape[3] == dim, "y_pred and y_gt should have same final dimension at dim=3!"

        return B, N, M, dim

    def compute_score(self, y_pred, y_gt):
        """
        Calculates the proper scoring rule given predictions and groundtruth targets.
        :param y_pred: (batch x N x num_samples x dim) The neural network predicted expected value.
        :param y_gt: (batch x N x num_samples x dim) The groundtruth output.

        :return: score: (batch x N) proper score for every input-output pair.
        """

        # Check dimensions
        B, N, _, _ = self.check_dims(y_pred, y_gt)

        score_all_batches = []
        # Compute energy score for every batch independently. geomloss code is very memory intensive.
        for b in range(B):
            batch_y_pred = y_pred[b]
            batch_y_gt = y_gt[b]

            # Maximum N 500. If N>500, we have to iterate to make sure we don't run out of memory.
            batch_y_gt_list = torch.split(batch_y_gt, 500, dim=0)
            batch_y_pred_list = torch.split(batch_y_pred, 500, dim=0)

            n_batch_es_list = []
            # Fragment compute to not ever exceed 512 samples. This is because of geomloss memory issues.
            for y_pred_in, y_gt_in in zip(batch_y_pred_list, batch_y_gt_list):
                n_batch_es_list.extend(self.score(y_pred_in, y_gt_in))

            score_all_batches.extend(torch.stack(n_batch_es_list, dim=0).view(1, N))

        score = torch.stack(score_all_batches, dim=0)
        return score


def main():
    # Test gaussian evaluator
    gaussian_evaluator = GaussianEvaluator(es_num_samples=1000, num_repeats=1000)
    energy_sample_evaluator = SampleEvaluator(score='energy')
    mmd_sample_evaluator = SampleEvaluator(score='mmd')

    # Small test with known ranking. Batch size 1 and dim 1. y_pred are ranked from best to worst.
    B, N, dim = 1, 20, 1
    y_gt = torch.ones((B, N, dim)).cuda()

    y_pred = torch.rand((B, N, dim)).cuda()
    y_pred, _ = torch.sort(y_pred, dim=1, descending=True)
    var_pred = torch.ones((B, N, dim, dim)).cuda()

    # Compute nll
    nll = gaussian_evaluator.compute_nll(y_pred, y_gt, var_pred)

    # On average, energy score should rank the best and worst predictions same as nll. Most of the time,
    # whole ranking is also the same, but due to stochasticity in ES, outliers might occur and some rankings might be
    # wrong.
    es = gaussian_evaluator.compute_energy_score(y_pred, y_gt, var_pred)

    _, indices_nll = torch.sort(nll, dim=1)
    _, indices_es = torch.sort(es, dim=1)
    assert (torch.equal(indices_nll[:, 0], indices_es[:, 0]))
    assert (torch.equal(indices_nll[:, -1], indices_es[:, -1]))

    # Test with sample-based evaluators

    distributions = torch.distributions.MultivariateNormal(y_pred, covariance_matrix=var_pred)
    y_pred = distributions.sample((1000,)).permute((1, 2, 0, 3))

    es = energy_sample_evaluator.compute_score(y_pred, y_gt.unsqueeze(-1))
    mmd = mmd_sample_evaluator.compute_score(y_pred, y_gt.unsqueeze(-1))

    _, indices_es_samples = torch.sort(es, dim=1)
    _, indices_mmd_samples = torch.sort(mmd, dim=1)

    # Now Test with batch size, dim > 1.
    gaussian_evaluator = GaussianEvaluator(es_num_samples=1000, num_repeats=3)

    B, N, dim = 5, 5045, 2

    y_pred = torch.rand((B, N, dim)).cuda()
    y_gt = torch.rand((B, N, dim)).cuda()
    var_pred = torch.rand((B, N, dim, dim)).cuda()
    var_pred = torch.matmul(var_pred, torch.permute(var_pred, (0, 1, 3, 2))) + 0.01

    nll = gaussian_evaluator.compute_nll(y_pred, y_gt, var_pred)
    es = gaussian_evaluator.compute_energy_score(y_pred, y_gt, var_pred)

    # Test with samples instead of gaussian.
    y_pred = torch.rand((B, N, 1000, dim)).cuda()
    y_gt = torch.rand((B, N, 1, dim)).cuda()

    es = energy_sample_evaluator.compute_score(y_pred, y_gt)
    mmd = mmd_sample_evaluator.compute_score(y_pred, y_gt)


if __name__ == '__main__':
    main()
