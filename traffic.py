import numpy as np
import pandas as pd
import pytorch_lightning
import seaborn as sns
import torch

from matplotlib import pyplot as plt

from src.core.models import SampleNet
from src.core.trainers import SampleNetTrainer
from src.core.evaluation.proper_scoring_rules import SampleEvaluator

sns.set_theme(font_scale=1.7)


def hours_tensor_to_sin_cos(hours_tensor):
    hours_tensor_sin = torch.sin((np.pi * 2 * hours_tensor) / 24)
    hours_tensor_cos = torch.cos((np.pi * 2 * hours_tensor) / 24)
    hours_tensor = torch.concat((hours_tensor_sin,
                                 hours_tensor_cos), dim=1)

    return hours_tensor


class ToyTrafficDataGenerator:
    def __init__(self,
                 filepath="./data/traffic_volume.csv",
                 usecols=["date_time", "traffic_volume"],
                 header=0,
                 footer=0):
        """
        This data generator load weight and height data from a source file
        (downloaded from https://mrcc.purdue.edu/CLIMATE/Station/Daily/StnDyBTD.jsp)
        and create training set based on configuration
        :param filepath: filepath for weather data
        :param usecols: columns used on the file
        :param header: lines ignored at header
        :param footer: lines ignored at footer
        """
        df = pd.read_csv(filepath,
                         usecols=usecols,
                         header=header,
                         skipfooter=footer,
                         na_values="M")
        df["date_time"] = pd.to_datetime(df["date_time"])
        df["year"] = df["date_time"].dt.year
        df["hour"] = df["date_time"].dt.hour
        print(df.head())
        self.data = df
        self.min_length = None
        # compute the mean and std for each day

    def create_training_data(self, years=[2014],
                             device='cpu',
                             fraction=1.0):
        """
        Create training set from multiple years of traffic data
        :param device: device used for training. eg. 'cpu' or 'cuda:0'
        :param years: years selected as training data
        :return:
        """
        print("Creating datasets from years", years)
        x_train = []
        y_train = []
        x_train_multi_sample = []
        y_train_multi_sample = []
        for year in years:
            train_data = self.data[self.data["year"] == year]
            train_data = train_data.sample(int(len(train_data) * fraction))
            train_data_groups = train_data.groupby("hour")
            for group in train_data_groups:
                x_train_multi_sample.append(group[0])
                y_train_multi_sample.append(group[1]["traffic_volume"].to_numpy(dtype=np.float32))
            x_train.append(train_data["hour"].to_numpy(dtype=np.float32))
            y_train.append(train_data["traffic_volume"].to_numpy(dtype=np.float32))

        self.min_length = np.array([sample_y_train.shape[0] for sample_y_train in y_train_multi_sample]).min()
        x_train_multi_sample = torch.tensor(
            np.array(x_train_multi_sample).astype(np.float32).reshape(-1, 1)).to(device=device)
        y_train_multi_sample = torch.tensor(
            np.array([sample_y_train[0: self.min_length] for sample_y_train in y_train_multi_sample])).to(device=device)

        # Process multi-sample data to be compatible with non-sampling networks
        x_train = np.concatenate(x_train, 0)
        y_train = np.concatenate(y_train, 0)
        x_train = torch.tensor(x_train.reshape(-1, 1)).to(device=device)
        y_train = torch.tensor(y_train.reshape(-1, 1)).to(device=device)

        return hours_tensor_to_sin_cos(x_train), y_train, hours_tensor_to_sin_cos(
            x_train_multi_sample), y_train_multi_sample

    def create_test_data(self, years=[2018], device='cpu'):
        """
        create test data to evalute the performance of variance prediction
        :return:    test_x:
                    test_y:
        """
        _, y_test, x_test, y_test_multisample = self.create_training_data(years, device)
        return x_test, y_test, y_test_multisample


def plot_traffic_toy_example(
        y_test,
        y_samples_test,
        label):
    """
    """
    x_test = np.tile(np.arange(0, 24), 5)
    y_test = y_test.cpu().numpy()
    y_samples_test = y_samples_test.squeeze().cpu().numpy()
    y_samples_test[y_samples_test < 0] = 0
    fig, ax = plt.subplots(1, 1, figsize=(30, 15))
    # print the observations on training set

    data_frame = pd.DataFrame()
    data_frame['Hour'] = np.concatenate(
        (np.repeat(x_test, y_test.shape[1]), np.repeat(x_test, y_samples_test.shape[1])))
    data_frame['Traffic Volume'] = np.concatenate((y_test.flatten(), y_samples_test.flatten()))
    data_frame['Legend'] = ['Ground Truth' for _ in range(y_test.flatten().shape[0])] + ['Predictions' for _ in
                                                                                         range(
                                                                                             y_samples_test.flatten().shape[
                                                                                                 0])]
    color = sns.color_palette()
    p = sns.jointplot(data=data_frame,
                      x='Hour',
                      y='Traffic Volume',
                      hue='Legend',
                      ax=ax,
                      xlim=(-4, 28),
                      ylim=(-1000, 8000),
                      pallet=[color[1], color[0]],
                      kind="kde",
                      common_norm=False,
                      marginal_kws=dict(common_norm=False,
                                        fill=True))

    p.fig.suptitle(label)
    p.ax_joint.collections[0].set_alpha(0.7)
    p.fig.tight_layout()
    p.fig.subplots_adjust(top=0.95)
    p.ax_marg_x.set_axis_off()


def plot_traffic_toy_example_scatter(
        y_test,
        y_samples_test,
        label):
    """
    """
    x_test = np.tile(np.arange(1, 25), 5)
    y_test = y_test.cpu().numpy()
    y_samples_test = y_samples_test.squeeze().cpu().numpy()
    y_samples_test[y_samples_test < 0] = 0
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    # print the observations on training set

    data_frame = pd.DataFrame()
    data_frame['Hour'] = np.concatenate(
        (np.repeat(x_test, y_test.shape[1]), np.repeat(x_test, y_samples_test.shape[1])))
    data_frame['Traffic Volume'] = np.concatenate((y_test.flatten(), y_samples_test.flatten()))
    data_frame['Legend'] = ['Ground Truth' for _ in range(y_test.flatten().shape[0])] + ['Predictions' for _ in
                                                                                         range(
                                                                                             y_samples_test.flatten().shape[
                                                                                                 0])]

    color = sns.color_palette()
    sns.scatterplot(data=data_frame,
                    x='Hour',
                    y='Traffic Volume',
                    hue='Legend',
                    ax=ax,
                    alpha=0.7,
                    palette=[color[1], color[0]],
                    legend=False)

    ax.set(xlim=(0, 25))
    ax.set(ylim=(-300, 7500))
    ax.xaxis.set_ticks((1, 4, 8, 12, 16, 20, 24))
    ax.yaxis.set_ticks((0, 2000, 4000, 6000, 8000))
    fig.suptitle(label)


def main():
    # Training of small network is faster on CPU. Training GPs and other complicated models faster on cuda device.
    device = 'cuda:0'

    # Choose model and set random seed
    pytorch_lightning.seed_everything(2022)

    years_training = [2015, ]
    years_testing = [2018, 2013, 2014, 2016, 2017]

    data_generator = ToyTrafficDataGenerator()
    x_train, y_train, x_train_multi_sample, y_train_multi_sample = data_generator.create_training_data(years_training,
                                                                                                       device=device)
    x_test, y_test, y_test_multisample = data_generator.create_test_data(years_testing, device=device)
    num_test_samples = data_generator.min_length

    ####################################################
    # Sample based network trained with Various Losses #
    ####################################################
    # Create model
    model = SampleNet(n_neurons=50, activation='elu',
                      n_generated_samples=100,
                      input_dim=2, minibatch_sample_size=100).to(
        device=device)

    # Train model
    net_trainer = SampleNetTrainer(model, num_training_iterations=5000,
                                   lr=1e-1,
                                   mmd_blur=0.5,
                                   sinkhorn_blur=0.05,
                                   sinkhorn_p_norm=2,
                                   sinkhorn_loss_weight=0.0,
                                   minibatch_sample_size=100,
                                   minibatch_max_num_runs=1,
                                   prior_type='gaussian')
    # Works with both [x_train_multi_sample, y_train_multi_sample] or [x_train, y_train], achieves similar results.
    # [x_train, y_train] takes much longer to train.
    model = net_trainer.train_model(x_train_multi_sample,
                                    y_train_multi_sample.unsqueeze(2),
                                    learn_mean_first=False,
                                    which_loss='es')

    # Perform inference over whole space
    y_test_est, var_test_est, predicted_samples = model.inference_forward(x_test)

    label = "SampleNet"

    # Generate samples from predictive distributions for evaluation and plotting
    y_test_est_dist = torch.tensor(np.expand_dims(y_test_est, 0)).to(device=device)
    var_test_est_dist = torch.tensor(np.expand_dims(var_test_est, (0, 3))).to(device=device)
    if predicted_samples is None:
        predictive_distributions = torch.distributions.MultivariateNormal(y_test_est_dist,
                                                                          covariance_matrix=var_test_est_dist)
        predicted_samples = predictive_distributions.sample((num_test_samples,)).permute(1, 2, 0, 3).to(
            device=device)
    else:
        predicted_samples = predicted_samples.unsqueeze(0).to(device=device)

    # Generate test energy score for any data
    evaluator_samples = SampleEvaluator()
    samples_test = y_test_multisample.unsqueeze(0).unsqueeze(-1).permute(0, 1, 2, 3).to(device=device)
    es_test = evaluator_samples.compute_score(predicted_samples, samples_test).mean().cpu().numpy()
    print('Energy Score: ' + str(es_test))

    plot_traffic_toy_example_scatter(y_test_multisample, predicted_samples, label=label)
    plt.subplots_adjust(top=0.9, right=0.9)
    plt.show()


if __name__ == "__main__":
    main()
