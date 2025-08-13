import os, sys
import pandas as pd
import numpy as np
from tqdm import trange
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn.functional import kl_div, mse_loss
from torch.multiprocessing import Process
from torch.utils.tensorboard import SummaryWriter
from audtorch.metrics.functional import pearsonr
from sklearn.model_selection import KFold, train_test_split

from src.models.XCRISP.features import DELETION_FEATURES
from src.data.data_loader import get_common_samples
from src.config.test_setup import MIN_NUMBER_OF_READS

FEATURE_SETS = {
    "v4": ["Gap", "leftEdge", "rightEdge", "homologyLength", "homologyGCContent"],
}
MH_LESS_FEATURES = ["Gap", "leftEdge", "rightEdge"]
MH_FEATURES = ["homologyLength", "homologyGCContent"]

def _to_tensor(arr):
    if isinstance(arr, pd.DataFrame) or isinstance(arr, pd.Series):
        arr = arr.to_numpy()
    return torch.tensor(arr).to(DEVICE).float()

def load_data(dataset = "train", num_samples = None, fractions=True):
    data = pd.read_pickle(INPUT_F.format(dataset))
    counts = data["counts"]
    del_features = data["del_features"]
    samples = counts.index.levels[0]
    if num_samples is not None:
        samples = samples[:num_samples]
        counts = counts.loc[samples]
        del_features = del_features.loc[samples]
    y = counts.loc[counts.Type == "DELETION"]
    y = y.fraction if fractions else y.countEvents
    del_features["Gap"] = del_features["Size"] - del_features["homologyLength"]
    X = del_features
    return X, y, samples

def get_folds(s):
    kf = KFold(n_splits=NUM_FOLDS, random_state=RANDOM_STATE, shuffle=True)
    folds = list(kf.split(s))
    return folds

def init_weights(m):
    if type(m) == nn.Linear:
        m.weight = nn.Parameter(torch.randn(m.weight.shape) * 0.1)
        m.bias = nn.Parameter(torch.randn(m.bias.shape) * 0.1)

class NoExperimentDefined(Exception):
    pass

class InvalidExperiment(Exception):
    pass

class NeuralNetwork(nn.Module):
    def __init__(self, num_features):
        super(NeuralNetwork, self).__init__()
        self.linear_sigmoid_stack = nn.Sequential(
            nn.Linear(num_features, 16),
            nn.Sigmoid(),
            nn.Linear(16, 16),
            nn.Sigmoid(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        # self.linear_sigmoid_stack.apply(init_weights)

    def forward(self, x):
        out = self.linear_sigmoid_stack(x)
        return out

def batch(X, Y, samples, model, loss_fn, optimizer, lr_scheduler):
    loss = torch.zeros(1).to(DEVICE)

    # for now we do gradient descent, but should make this full grad descent
    for s in samples:
        # Get features for input sequence
        x = _to_tensor(X.loc[s])
        # compute loss
        y = _to_tensor(Y.loc[s])
        y = y/y.sum()
        y_pred = model(x)[:,0]
        y_pred = y_pred/y_pred.sum()

        if torch.isnan(y).any():
            # print("Something is wrong")
            continue
        if torch.isnan(y_pred).any():
            print("Something is definitely wrong")
        
        loss += loss_fn(torch.log(y_pred), y) if loss_fn == kl_div else loss_fn(y_pred, y)

    loss = torch.div(loss, len(samples)) 

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    lr_scheduler.step()

    return loss.cpu().detach().numpy()[0]

def test(X, Y, samples, model):
    model.eval()
    metrics = []
    with torch.no_grad():
        for s in samples:
            x = _to_tensor(X.loc[s])
            y_pred = model(x)[:,0]
            y_pred = y_pred/y_pred.sum()
            y = _to_tensor(Y.loc[s])
            y = y/y.sum()

            if torch.isnan(y).any():
                continue

            metrics.append((pearsonr(y_pred, y).cpu().detach().numpy()[0], \
                kl_div(torch.log(y_pred), y).cpu().detach().numpy(), \
                mse_loss(y_pred, y).cpu().detach().numpy()))
    metric = np.mean(metrics, axis=0)
    return metric

def predict(model, x):
    model.eval()
    with torch.no_grad():
        x = _to_tensor(x)
        y_pred = model(x)[:,0]
        y_pred = y_pred/y_pred.sum()
    return y_pred.cpu().detach().numpy()

def write(writer, loss, train_metric, val_metric, epoch, samples, X, Y, model, cv=""):
    writer.add_scalar("training_loss{}".format(cv), loss, epoch)
    writer.add_scalar("training_correlation{}".format(cv), train_metric, epoch)
    if val_metric is not None:
        writer.add_scalar("validation_correlation{}".format(cv), val_metric, epoch)

def cv(X, Y, samples_train, samples_val, fold, model, loss_fn, optimizer, lr_scheduler, experiment_name="", writer=None):
    pbar = trange(EPOCHS, desc = "Cross Validation...", leave=True)
    for t in pbar:
        permutation = torch.randperm(len(samples_train))

        for i in range(0, len(samples_train), BATCH_SIZE):
            samples_batch = samples_train[permutation[i:i+BATCH_SIZE]]
            loss = batch(X, Y, samples_batch, model, loss_fn, optimizer, lr_scheduler)
            train_metric = test(X, Y, samples_batch, model)
            val_metric = test(X, Y, samples_val, model)
            pbar.set_description("Experiment {}, Fold {}, Epoch {}, Train loss: {}, Train Pearson's Corr: {}, Val Pearson's Corr: {}".format(experiment_name, fold, t, loss, train_metric, val_metric))

        if writer is not None:
            write(writer, loss, train_metric, val_metric, t, samples, X, Y, model, cv=fold)
    
    return loss, train_metric, val_metric


def train(X, Y, samples, model, loss_fn, optimizer, lr_scheduler, writer=None, experiment_name = ""):
    train_metric = test(X, Y, samples, model)
    print("{}: Starting training loss: {}".format(experiment_name, train_metric))
    samples, val_samples = train_test_split(samples, test_size = 100, random_state=RANDOM_STATE)
    pbar = trange(EPOCHS, desc = "Training {}...".format(experiment_name), leave=True)
    metrics = []
    for t in pbar:
        permutation = torch.randperm(len(samples))
        batch_metrics = []
        for i in range(0, len(samples), BATCH_SIZE):
            samples_batch = samples[permutation[i:i+BATCH_SIZE]]
            loss = batch(X, Y, samples_batch, model, loss_fn, optimizer, lr_scheduler)
            train_corr, train_kld, train_mse = test(X, Y, samples_batch, model)
            val_corr, val_kld, val_mse = test(X, Y, val_samples, model)
            batch_metrics.append((t, loss, train_corr, val_corr, val_kld))
        mean_metrics = np.array(batch_metrics)[:,1:].mean(axis=0)
        pbar.set_description("{}: Epoch {}, Train loss: {:.5f}, Train Pearson's Corr: {:.5f}, Validation Pearson's Corr: {:.5f}".format(experiment_name, t, *mean_metrics[:3]))
        metrics.append((t, *mean_metrics))
        plot(metrics, experiment_name)

        if writer is not None:
            write(writer, loss, train_corr, val_corr, t, samples, X, Y, model)
            

def plot(metrics, experiment_name):
    metrics = pd.DataFrame(metrics, columns=["Epoch", "Loss", "Training Corr", "Validation Corr", "Validation Loss"])

    sns.set_style("ticks",{'axes.grid' : True})

    plt.figure()
    losses = metrics[["Epoch", "Loss", "Validation Loss"]]
    losses = losses.melt('Epoch', var_name='Loss', value_name='Value')
    lineplot = sns.lineplot(data=losses, x="Epoch", y="Value", hue="Loss")
    fig = lineplot.get_figure()
    fig.tight_layout()
    fig.savefig(LOGS_DIR + experiment_name + "_training_loss_curve.png")
    plt.close(fig)

    plt.figure()
    corrs = metrics[["Epoch", "Training Corr", "Validation Corr"]]
    corrs = corrs.melt('Epoch', var_name='Corr', value_name='Value')
    lineplot2 = sns.lineplot(data=corrs, x="Epoch", y="Value", hue="Corr")
    fig2 = lineplot2.get_figure()
    fig2.tight_layout()
    fig2.savefig(LOGS_DIR + experiment_name + "_training_correlation_curve.png")
    plt.close(fig2)


def init_model(num_features, learning_rate, loss_function_str = "kld"):
    model = NeuralNetwork(num_features).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate, betas=(0.99, 0.999))
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.999)
    loss_fn = kl_div if loss_function_str == "kld" else mse_loss
    return model, optimizer, lr_scheduler, loss_fn

def run_experiment(X, y, samples, experiment_name, do_CV=True, learning_rate=0.05):
    # set up tensorboard logger
    # writer = SummaryWriter(LOGS_DIR + experiment_name)

    if do_CV:
        cv_folds = []
        # cross validation
        folds = get_folds(samples)
        for i, f in enumerate(folds):
            print("Training fold {} with {} samples, {} features".format(i, len(samples[f[0]]), X.shape[1]))
            model, optimizer, lr_scheduler, loss_fn =init_model(X.shape[1], learning_rate, loss_function_str=loss_function_str)
            loss, train_metrics, val_metrics = cv(X, y, samples[f[0]], samples[f[1]], i, model, loss_fn, optimizer, lr_scheduler, experiment_name=experiment_name, writer=None)
            cv_folds.append((loss, train_metrics[0], train_metrics[1], train_metrics[2], val_metrics[0], val_metrics[1], val_metrics[2]))

        cv_df = pd.DataFrame(cv_folds, columns=["Loss", "Train Corr", "Train KLD", "Train MSE", "Val Corr", "Val KLD", "Val MSE"])
        cv_df.loc['mean'] = cv_df.mean()
        cv_df.to_csv(OUTPUT_MODEL_F.format(loss_function_str, learning_rate).replace("pth", "folds.tsv"), sep="\t")
        
    # final training
    model, optimizer, lr_scheduler, loss_fn = init_model(X.shape[1], learning_rate, loss_function_str=loss_function_str)
    train(X, y, samples, model, loss_fn, optimizer, lr_scheduler, writer = None, experiment_name=experiment_name)

    # save model
    os.makedirs(OUTPUT_MODEL_D, exist_ok=True)
    torch.save(model, OUTPUT_MODEL_F.format(loss_function_str, learning_rate))

    torch.save({
        "random_state": RANDOM_STATE,
        "model": model.state_dict(),
        "samples": samples,
        "loss_fn" : kl_div,
        "optimiser" : optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict()
    }, OUTPUT_MODEL_F.format(loss_function_str, learning_rate).replace("pth", "details"))

    print("Model saved to", OUTPUT_MODEL_F.format(loss_function_str, learning_rate))

def load_model(model_dir="./models/", loss_fn="kld", learning_rate=0.05):
    model_d = "{}deletion_{}_{}___model.pth".format(model_dir, loss_fn, learning_rate)
    model = torch.load(model_d, map_location=torch.device('cpu'), weights_only = False)
    return model

# set global vars
OUTPUT_DIR = os.environ['OUTPUT_DIR']
MIN_READS_PER_TARGET = MIN_NUMBER_OF_READS
INPUT_F = OUTPUT_DIR + "/model_training/data_{}x".format(MIN_READS_PER_TARGET) + "/X-CRISP/{}.pkl" 
DEVICE = "cpu"

if __name__ == "__main__":
    TRAINING_DATA = "train"
    LOGS_DIR = os.environ['LOGS_DIR'] + TRAINING_DATA + "/"
    os.makedirs(LOGS_DIR, exist_ok=True)
    OUTPUT_MODEL_D = OUTPUT_DIR + "/model_training/model/X-CRISP/"
    OUTPUT_MODEL_F = OUTPUT_MODEL_D + "deletion_{}_{}___model.pth"
    NUM_FOLDS = 5
    RANDOM_STATE = 1
    EPOCHS = 200
    BATCH_SIZE = 200
    # get devices
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # DEVICE = "cpu"
    
    print('Using {} device'.format(DEVICE))

    # set number of threads to 1 for various libraries
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'

    # set random seed
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    FEATURES = "v4"

    learning_rate = float(sys.argv[1])
    loss_function_str = sys.argv[2]

    # load data
    X, y, samples = load_data(dataset=TRAINING_DATA, num_samples=None)

    # use common samples for experiment consistency
    common_samples = get_common_samples(genotype=TRAINING_DATA, min_reads=MIN_READS_PER_TARGET)
    samples = np.intersect1d(samples, common_samples)

    experiment_name = "{}_{}_{}".format(FEATURES, loss_function_str, learning_rate)

    print("Training on {} samples, with {} features and a learning rate of {}".format(len(samples), X.shape[1], learning_rate))
    run_experiment(X.loc[:, FEATURE_SETS[FEATURES]], y, samples, experiment_name=experiment_name, do_CV=False, learning_rate=learning_rate)

    print("Finished.")
