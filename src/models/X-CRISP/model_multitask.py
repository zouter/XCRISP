import os, sys
import pandas as pd
import numpy as np
from tqdm import trange

import torch
from torch import nn
from torch.nn.functional import kl_div, mse_loss
from torch.multiprocessing import Process
from torch.utils.tensorboard import SummaryWriter
from audtorch.metrics.functional import pearsonr
from sklearn.model_selection import KFold, train_test_split

from features import DELETION_FEATURES
from bins import bin_repair_outcomes_by_length

sys.path.append("../")
from data_loader import get_common_samples
from test_setup import MIN_NUMBER_OF_READS

FEATURE_SETS = {
    "v5": ["leftEdge", "rightEdge", "homologyLength", "homologyGCContent"],
}

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
    X = del_features[DELETION_FEATURES]

    gap = del_features["rightEdge"] - del_features["leftEdge"]

    counts = counts.loc[counts.Type == "DELETION"]
    counts = counts.fraction if fractions else counts.countEvents

    y = (gap, counts)
    return X, y, samples

def get_folds(s):
    kf = KFold(n_splits=NUM_FOLDS, random_state=RANDOM_STATE, shuffle=True)
    folds = list(kf.split(s))
    return folds

# def init_weights(m):
#     if type(m) == nn.Linear:
#         m.weight = nn.Parameter(torch.randn(m.weight.shape) * 0.1)
#         m.bias = nn.Parameter(torch.randn(m.bias.shape) * 0.1)

class NoExperimentDefined(Exception):
    pass

class InvalidExperiment(Exception):
    pass

class NeuralNetwork(nn.Module):
    def __init__(self, num_features):
        super(NeuralNetwork, self).__init__()
        self.common_stack = nn.Sequential(
            nn.Linear(num_features, 16),
            nn.ReLU(),
        )
        self.gap_stack = nn.Sequential(
            nn.Linear(16, 1)
        )
        self.indel_stack = nn.Sequential(
            nn.Linear(5, 16),
            nn.Sigmoid(),
            nn.Linear(16, 16),
            nn.Sigmoid(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        g = x[:,1:2] - x[:,0:1] # gap = right edge - left edge 
        x = torch.cat((x, g), 1)
        # out = self.common_stack(x)
        x1 = g
        x2 = self.indel_stack(x)[:,0]
        return x1, x2

def batch(X, Y, samples, model, loss_fn, optimizer, lr_scheduler):
    loss = torch.zeros(1).to(DEVICE)

    # for now we do gradient descent, but should make this full grad descent
    for s in samples:
        # Get features for input sequence
        x = _to_tensor(X.loc[s])
        y_pred_gap, y_pred_indel = model(x)
        # compute loss
        y_gap = _to_tensor(Y[0].loc[s])
        # loss += loss_fn(y_pred_gap, y_gap)

        y_indel = _to_tensor(Y[1].loc[s])
        y_indel = y_indel/y_indel.sum()
        y_pred_indel = y_pred_indel/y_pred_indel.sum()
        loss += loss_fn(y_pred_indel, y_indel)
        

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
            y_pred_gap, y_pred_indel = model(x)
            # compute loss
            y_gap = _to_tensor(Y[0].loc[s])
            y_indel = _to_tensor(Y[1].loc[s])
            y_pred_indel = y_pred_indel/y_pred_indel.sum()
            
            metrics.append(pearsonr(y_pred_indel, y_indel).cpu().detach().numpy()[0])
    metric = np.mean(metrics)
    return metric

def predict(model, x):
    model.eval()
    with torch.no_grad():
        x = _to_tensor(x)
        _, y_pred_indel = model(x)
    return y_pred_indel.cpu().detach().numpy()

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
    print("{}: Starting training pearson's corr: {}".format(experiment_name, train_metric))
    samples, val_samples = train_test_split(samples, test_size = 100, random_state=RANDOM_STATE)
    pbar = trange(EPOCHS, desc = "Training {}...".format(experiment_name), leave=True)
    for t in pbar:
        permutation = torch.randperm(len(samples))

        for i in range(0, len(samples), BATCH_SIZE):
            samples_batch = samples[permutation[i:i+BATCH_SIZE]]
            loss = batch(X, Y, samples_batch, model, loss_fn, optimizer, lr_scheduler)
            train_metric = test(X, Y, samples_batch, model)
            val_metric = test(X, Y, val_samples, model)
            pbar.set_description("{}: Epoch {}, Train loss: {}, Train Pearson's Corr: {}, Validation Pearson's Corr: {}".format(experiment_name, t, loss, train_metric, val_metric))

        if writer is not None:
            write(writer, loss, train_metric, val_metric, t, samples, X, Y, model)


def init_model(num_features):
    model = NeuralNetwork(num_features).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), 0.01, betas=(0.99, 0.999))
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.999)
    loss_fn = mse_loss
    return model, optimizer, lr_scheduler, loss_fn

def run_experiment(X, y, samples, experiment_name, do_CV=False):
    # set up tensorboard logger
    writer = SummaryWriter(LOGS_DIR + experiment_name)

    if do_CV:
        cv_folds = []
        # cross validation
        folds = get_folds(samples)
        for i, f in enumerate(folds):
            print("Training fold {} with {} samples, {} features".format(i, len(samples[f[0]]), X.shape[1]))
            model, optimizer, lr_scheduler, loss_fn = init_model(X.shape[1])
            loss, train_corr, val_corr = cv(X, y, samples[f[0]], samples[f[1]], i, model, loss_fn, optimizer, lr_scheduler, experiment_name=experiment_name, writer=writer)
            cv_folds.append((loss, train_corr, val_corr))

        cv_df = pd.DataFrame(cv_folds, columns=["Loss", "Train Corr", "Val Corr"])
        cv_df.loc['mean'] = cv_df.mean()
        cv_df.to_csv(OUTPUT_MODEL_F.format(experiment_name, RANDOM_STATE).replace("pth", ".folds.tsv"), sep="\t")
        
    # final training
    model, optimizer, lr_scheduler, loss_fn = init_model(X.shape[1])
    train(X, y, samples, model, loss_fn, optimizer, lr_scheduler, writer = writer, experiment_name=experiment_name)

    # save model
    os.makedirs(OUTPUT_MODEL_D, exist_ok=True)
    torch.save(model, OUTPUT_MODEL_F.format(experiment_name, RANDOM_STATE))

    torch.save({
        "random_state": RANDOM_STATE,
        "model": model.state_dict(),
        "samples": samples,
        "loss_fn" : mse_loss,
        "optimiser" : optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict()
    }, OUTPUT_MODEL_F.format(experiment_name, RANDOM_STATE).replace("pth", "details"))


def load_model(feature_set, loss, num_reads=MIN_NUMBER_OF_READS, model_dir="./models/", random_state=1):
    model_d = "{}Sigmoid_{}x_{}_{}Loss_RS_{}_model.pth".format(model_dir, num_reads, feature_set, loss, random_state)
    model = torch.load(model_d)
    return model

# set global vars
OUTPUT_DIR = os.environ['OUTPUT_DIR']
MIN_READS_PER_TARGET = MIN_NUMBER_OF_READS
INPUT_F = OUTPUT_DIR + "/model_training/data_{}x".format(MIN_READS_PER_TARGET) + "/X-CRISP/{}.pkl" 
DEVICE = "cpu"

if __name__ == "__main__":
    LOGS_DIR = os.environ['LOGS_DIR']
    OUTPUT_MODEL_D = OUTPUT_DIR + "/model_training/model/X-CRISP/"
    OUTPUT_MODEL_F = OUTPUT_MODEL_D + "Sigmoid_{}x".format(MIN_READS_PER_TARGET) + "_{}_RS_{}_model.pth"
    NUM_FOLDS = 5
    RANDOM_STATE = 1
    EPOCHS = 300
    BATCH_SIZE = 200
    # get devices
    # DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DEVICE = "cpu"
    TRAINING_DATA = "train"
    print('Using {} device'.format(DEVICE))

    # set number of threads to 1 for various libraries
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'

    # set random seed
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    FEATURES = "v5"

    # load data
    X, y, samples = load_data(dataset=TRAINING_DATA, num_samples=200)

    # use common samples for experiment consistency
    common_samples = get_common_samples(genotype=TRAINING_DATA, min_reads=MIN_READS_PER_TARGET)
    samples = np.intersect1d(samples, common_samples)

    experiment_name = "{}_MultiTask".format(FEATURES)

    print("Training on {} samples, with {} features".format(len(samples), X.shape[1]))
    run_experiment(X.loc[:, FEATURE_SETS[FEATURES]], y, samples, experiment_name=experiment_name)

    print("Finished.")
