import numpy as np
import pandas as pd
import pytorch_lightning
import seaborn as sns
import torch

from matplotlib import pyplot as plt


from src.core.models import SampleNet
from src.core.trainers import SampleNetTrainer
from src.core.evaluation.proper_scoring_rules import SampleEvaluator

sns.set_theme()


class ToyWeatherDataGenerator:
    def __init__(self,
                 filepath="../data/station_1.csv",
                 usecols=["Date", "TMAX"],
                 header=3,
                 footer=18):
        """
        This data generator load weather data from a source file
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
        df["Date"] = pd.to_datetime(df["Date"])
        df["TMAX"] = pd.to_numeric(df["TMAX"], errors='coerce')
        df["year"] = df["Date"].dt.year
        df["month-day"] = df["Date"].dt.month * 100 + df["Date"].dt.day
        print(df.head())
        self.data = df
        # compute the mean and std for each day
        self.mu = df.groupby(["month-day"])["TMAX"].mean()
        self.sigma = df.groupby(["month-day"])["TMAX"].std()
        self.first_year = df["year"].min()
        self.last_year = df["year"].max()

    def create_training_data(self, years=[2000], device='cpu'):
        """
        Create training set from multiple years of weather data
        :param device: device used for training. eg. 'cpu' or 'cuda:0'
        :param years: years selected as training data
        :return:
        """
        print("Creating datasets from years", years)
        x_train = []
        y_train = []
        for year in years:
            train_data = self.data[self.data["year"] == year].set_index(keys="month-day")
            train_data["x"] = np.arange(len(train_data))
            train_data.fillna(method="pad", inplace=True, limit=5)
            x_train.append(train_data["x"].to_numpy(dtype=np.float32))
            y_train.append(train_data["TMAX"].to_numpy(dtype=np.float32))

        # Process Training Data
        x_train_multi_sample = x_train[:]
        y_train_multi_sample = y_train[:]

        x_train_multi_sample = torch.tensor(x_train_multi_sample).to(device=device)[0].unsqueeze(1)
        y_train_multi_sample = torch.tensor(y_train_multi_sample).to(device=device).permute(1, 0)

        # Process multi-sample data to be compatible with non-sampling networks
        x_train = np.concatenate(x_train, 0)
        y_train = np.concatenate(y_train, 0)
        x_train = torch.tensor(x_train.reshape(-1, 1)).to(device=device)
        y_train = torch.tensor(y_train.reshape(-1, 1)).to(device=device)

        return x_train, y_train, x_train_multi_sample, y_train_multi_sample

    def create_test_data(self):
        """
        create test data to evalute the performance of variance prediction
        :return:    test_x: 1d numpy array of date index (0-365)
                    test_mu: mean of TMAX temperature of the date in history
                    test_sigma: std of TMAX temperature of the date in history
        """
        test_data = pd.DataFrame(
            {
                "mu": self.mu,
                "sigma": self.sigma,
            }
        )

        # Process t-max samples:
        test_data["x"] = np.arange(len(test_data))
        test_data.fillna(method="pad", inplace=True, limit=5)
        x_test = test_data["x"].to_numpy(dtype=np.float32)
        mu_test = test_data["mu"].to_numpy(dtype=np.float32)
        sigma_test = test_data["sigma"].to_numpy(dtype=np.float32)

        # Evaluation data
        samples_test = []
        for year in range(self.first_year + 1, self.last_year + 1):
            sample_data = self.data[self.data["year"] == year].set_index(keys="month-day")
            sample_data.fillna(method="backfill", inplace=True, limit=5)
            samples_test.append(sample_data["TMAX"].to_numpy(dtype=np.float32)[0:365])

        return x_test, mu_test, sigma_test, np.array(samples_test)


def plot_weather_toy_example_multi_station(
        x_train_multi_sample,
        y_train_multi_sample,
        x_test,
        y_test_samples,
        mu_test,
        sigma_test,
        label):
    """
    Plots groundtruth mean and variance in comparison to predicted mean and variance.
    :param x_train_multi_sample: training data to plot
    :param y_train_multi_sample: training data to plot
    :param x_test: input from test set (using linspace)
    :param y_test_samples: Samples generated for x_test
    :param x_train: training input data
    :param y_train: training output data
    :param mu_test: list of ground-truth means of x_test
    :param sigma_test: list of ground-truth stds of x_test
    :param label: legend label
    :return:
    """
    x_test = x_test.cpu().numpy().flatten()
    y_test_samples = y_test_samples.cpu().numpy()
    y_train_multi_sample = y_train_multi_sample.cpu().numpy(
    )
    x_train_multi_sample = x_train_multi_sample.cpu().numpy(
    )
    fig, ax = plt.subplots(1, 1, figsize=(20, 15))

    # print estimations on test set, along with 95% conf interval
    ax.scatter(np.repeat(np.expand_dims(x_test[0:365], 1), y_test_samples.shape[2], axis=1),
               y_test_samples[0, :, :, 0],
               alpha=0.2,
               s=130,
               marker='D',
               c='tab:blue',
               label=u'Predictions')

    # print the confidence bound on historical data
    colors = ['tab:orange', 'tab:red']
    i = 0
    for mu_test_i, sigma_test_i, color in zip(mu_test, sigma_test, colors):
        ax.scatter(x_train_multi_sample, y_train_multi_sample[:, i], c=color, marker='.', s=300,
                   label='Observations')
        ax.plot(x_test, mu_test_i - 1.96 * sigma_test_i, color=color, lw=10, linestyle=':')
        ax.plot(x_test, mu_test_i + 1.96 * sigma_test_i, color=color, lw=10, linestyle=':')
        i += 1

    ax.set_xlabel('Day')
    ax.set_ylabel('Maximum Temperature')
    ax.xaxis.set_ticks((100, 200, 300))
    ax.yaxis.set_ticks((20, 60, 100, 140))

    plt.title(label, fontsize=80)
    ax.axis([0, 366, 0, 150])

    for item in ([ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(80)


def main():
    is_multimodal = True
    learn_mean_first = False if is_multimodal else True

    pytorch_lightning.seed_everything(0)

    # Training of small network is faster on CPU. Training GPs and other complicated models faster on cuda device.
    device = 'cuda:0'

    # Generate data
    if is_multimodal:
        years = [1995]
        data_generator_station_1 = ToyWeatherDataGenerator(filepath="./data/station_1.csv")
        x_train_1, y_train_1, x_train_multi_sample_1, y_train_multi_sample_1 = data_generator_station_1.create_training_data(
            years, device)
        x_test_1, mu_test_1, sigma_test_1, samples_test_1 = data_generator_station_1.create_test_data()

        data_generator_station_2 = ToyWeatherDataGenerator(filepath="./data/station_2.csv")
        x_train_2, y_train_2, x_train_multi_sample_2, y_train_multi_sample_2 = data_generator_station_2.create_training_data(
            years, device)
        x_test_2, mu_test_2, sigma_test_2, samples_test_2 = data_generator_station_2.create_test_data()

        x_train = torch.cat((x_train_1, x_train_2), dim=0)
        y_train = torch.cat((y_train_1, y_train_2), dim=0)

        x_train_multi_sample = x_train_multi_sample_1
        y_train_multi_sample = torch.cat((y_train_multi_sample_1, y_train_multi_sample_2), dim=1)

        samples_test = np.concatenate((samples_test_1, samples_test_2), axis=0)

        x_test = torch.tensor(x_test_1.reshape(-1, 1)).to(device=device)

        mu_test = [mu_test_1, mu_test_2]
        sigma_test = [sigma_test_1, sigma_test_2]

    else:
        years = [1995]
        data_generator = ToyWeatherDataGenerator(filepath="./data/station_1.csv")
        x_train, y_train, x_train_multi_sample, y_train_multi_sample = data_generator.create_training_data(years,
                                                                                                           device)

        x_test, mu_test, sigma_test, samples_test = data_generator.create_test_data()
        x_test = torch.tensor(x_test.reshape(-1, 1)).to(device=device)

    # Normalize input variables
    x_train_plot = x_train[:]
    x_train = ((x_train - x_train.mean(0)) / x_train.var(0).sqrt())

    x_train_multi_sample_plot = x_train_multi_sample[:]
    x_train_multi_sample = ((x_train_multi_sample - x_train_multi_sample.mean(0)) / x_train_multi_sample.var(0).sqrt())

    x_test_plot = x_test[:]
    x_test = ((x_test - x_test.mean(0)) / x_test.var(0).sqrt())

    # train model
    ####################################################
    # Sample based network trained with Various Losses #
    ####################################################
    # Create model
    model = SampleNet(n_neurons=50, activation='elu', n_generated_samples=100,
                      minibatch_sample_size=100).to(
        device=device)

    # Train model
    net_trainer = SampleNetTrainer(model, num_training_iterations=5000,
                                   lr=1e-2,
                                   mmd_blur=0.5,
                                   sinkhorn_blur=0.05,
                                   sinkhorn_p_norm=2,
                                   sinkhorn_loss_weight=5.0,
                                   minibatch_sample_size=100,
                                   minibatch_max_num_runs=1,
                                   prior_type='gaussian')
    # Works with both [x_train_multi_sample, y_train_multi_sample] or [x_train, y_train], achieves similar results.
    # [x_train, y_train] takes longer to train since batch size doubles.
    model = net_trainer.train_model(x_train_multi_sample,
                                    y_train_multi_sample.unsqueeze(2),
                                    learn_mean_first=learn_mean_first,
                                    which_loss='es')

    # Perform inference over whole space
    y_test_est, var_test_est, predicted_samples = model.inference_forward(x_test)
    label = 'SampleNet'

    # Generate samples from predictive distributions for evaluation and plotting
    y_test_est_dist = torch.tensor(np.expand_dims(y_test_est, 0)).to(device=device)
    var_test_est_dist = torch.tensor(np.expand_dims(var_test_est, (0, 3))).to(device=device)
    if predicted_samples is None:
        predictive_distributions = torch.distributions.MultivariateNormal(y_test_est_dist,
                                                                          covariance_matrix=var_test_est_dist)
        predicted_samples = predictive_distributions.sample((100,)).permute(1, 2, 0, 3)[:, 0:365,
                            :, :].to(
            device=device)
    else:
        predicted_samples = predicted_samples[0:365].unsqueeze(0).to(device=device)

    # Generate test energy score for any data
    evaluator_samples = SampleEvaluator()

    # Remove NANs from samples
    samples_test = samples_test[~np.isnan(samples_test).any(axis=1)]
    samples_test = torch.tensor(np.expand_dims(samples_test, (0, 3))).permute(0, 2, 1, 3).to(device=device)
    es_test = evaluator_samples.compute_score(predicted_samples, samples_test).mean().cpu().numpy()
    print("Energy Score:" + str(es_test))

    plot_weather_toy_example_multi_station(x_train_multi_sample_plot, y_train_multi_sample, x_test_plot,
                                           predicted_samples[:, :, np.random.choice(100, 100),
                                           :],
                                           mu_test, sigma_test, label)
    plt.show()


if __name__ == '__main__':
    # Command Line Args
    main()
