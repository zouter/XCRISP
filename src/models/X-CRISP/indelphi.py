import os, sys, random
from matplotlib.pyplot import axis
import pandas as pd
import numpy as np
from pandas.io.parquet import to_parquet
from torch.nn.modules.activation import Sigmoid
from torch.optim import lr_scheduler, optimizer
from tqdm import trange, tqdm

import torch
from torch import nn
from torch.nn.functional import kl_div, mse_loss
from torch.multiprocessing import Process
from torch.utils.tensorboard import SummaryWriter
from audtorch.metrics.functional import pearsonr
from sklearn.model_selection import KFold, train_test_split

from features import DELETION_FEATURES
from model import _to_tensor, init_weights, pearsons_mean_loss, NoExperimentDefined, InvalidExperiment

sys.path.append("../")
from data_loader import get_common_samples
from test_setup import MIN_NUMBER_OF_READS

# set global vars
OUTPUT_DIR = os.environ['OUTPUT_DIR']
MIN_READS_PER_TARGET = MIN_NUMBER_OF_READS
INPUT_F = OUTPUT_DIR + "/model_training/data_{}x".format(MIN_READS_PER_TARGET) + "/X-CRISP/{}.pkl" 
MH_LESS_FEATURES = ["Size"]
EXPERIMENTS = {
    "full+numrepeat": ["Size", "Start", "homologyLength", "numRepeats", "homologyGCContent", "homologyDistanceRank", "homologyLeftEdgeRank", "homologyRightEdgeRank", "homologyLengthRank"],
    "full": ["Size", "Start", "homologyLength", "homologyGCContent", "homologyDistanceRank", "homologyLeftEdgeRank", "homologyRightEdgeRank", "homologyLengthRank"],
    "inDelphi_features": ["homologyLength", "homologyGCContent"]
}

def _split_data_into_X_and_y(data, features=DELETION_FEATURES):
    y = data.fraction
    X = data[features + ["Indel"]]
    return X, y

def load_data(dataset = "train", num_samples = None):
    print("Loading data...")

    data = pd.read_pickle(INPUT_F.format(dataset))
    data["Indel"] = data.index.get_level_values("Indel")
    samples = data.index.levels[0]

    if num_samples is not None:
        samples = samples[:num_samples]
        data = data.loc[samples]
    
    # just MH data

    mh_data = data.loc[data["homologyLength"] != 0, :]
    # mh_data = mh_data[mh_data.fraction > 0]
    X_mh, y_mh = _split_data_into_X_and_y(mh_data)
    zero_mh = y_mh.groupby(["Sample_Name"]).sum() == 0
    zero_mh = set(zero_mh[zero_mh].index)
    

    # All data summarised up to deletion length

    # mh_less_data = data.loc[data["homologyLength"] == 0,] # no microhomologies
    # mh_less_data = mh_less_data.loc[data["homologyLength"] != data["Size"],] # not full mh
    mh_less_data = data.loc[data["Size"] < 29,] # del length less than 29
    mh_less_data = mh_less_data.loc[:, ["Indel", "Size", "fraction"]]
    y_mh_less = mh_less_data.groupby(["Sample_Name", "Size"]).sum()
    zero_mhless = y_mh_less.groupby(["Sample_Name"]).sum() == 0
    zero_mhless = set(zero_mhless[zero_mhless.fraction].index)

    samples = np.array(list(set(samples) - zero_mh.union(zero_mhless))) # remove any samples with zero counts in MH or deletion length data

    # y_mh_less = mh_less_data[mh_less_data.fraction > 0]
    # nonzero_mh = set(y_mh.reset_index()["Sample_Name"].unique())
    # nonzero_mhless = set(y_mh_less.reset_index()["Sample_Name"].unique())
    # samples = np.array(list(nonzero_mh.intersection(nonzero_mhless))) # remove any samples with zero counts
   
    return X_mh, y_mh, y_mh_less, samples

class DualNeuralNetwork(nn.Module):
    def __init__(self, num_features):
        super(DualNeuralNetwork, self).__init__()
        self.mh = nn.Sequential(
            nn.Linear(num_features, 16),
            nn.Sigmoid(),
            nn.Linear(16, 16),
            nn.Sigmoid(),
            nn.Linear(16, 1)
        )
        self.mh.apply(init_weights)

        self.mh_less = nn.Sequential(
            nn.Linear(1, 16),
            nn.Sigmoid(),
            nn.Linear(16, 16),
            nn.Sigmoid(),
            nn.Linear(16, 1)
        )
        self.mh_less.apply(init_weights)

    def _psi_score(self, phi, dl):
        return torch.exp(phi - (dl.reshape(-1, 1) * 0.25))

    def forward(self, x_mh, x_mh_less, experiment_name=sys.argv[1] if len(sys.argv)>1 else None):
        mh_indels = x_mh.index.to_list()
        mh_less_indels = x_mh_less

        #################################################
        # Predict MH Deletions
        #################################################
        del_lens = x_mh["Size"]
        # predict mh
        mh_scores = self._psi_score(self.mh(_to_tensor(x_mh[EXPERIMENTS[experiment_name]])), _to_tensor(del_lens))

        # add mh-less contributions at full MH lengths
        mh_vector = x_mh["homologyLength"]
        mhfull_contribution = torch.zeros(mh_vector.shape)
        for jdx in range(len(mh_vector)):
            if del_lens[jdx] == mh_vector[jdx]:
                mhless_score = self._psi_score(self.mh_less(_to_tensor(x_mh[jdx:jdx+1][MH_LESS_FEATURES])), _to_tensor(del_lens[jdx]))[0,0]
                mask = torch.cat((torch.zeros(jdx,), torch.ones(1,) * mhless_score, torch.zeros(len(mh_vector) - jdx - 1,)))
                mhfull_contribution = mhfull_contribution + mask
        mhfull_contribution = mhfull_contribution.reshape(-1, 1)
        y_mh_phi = mh_scores + mhfull_contribution

        #################################################
        # Predict MH-Less Deletions
        #################################################
        # y_mh_less_phi = self._psi_score(self.mh_less(_to_tensor(x_mh_less.reshape(-1, 1))), _to_tensor(x_mh_less))
        dls = torch.arange(1, 28+1)
        dls = dls.reshape(28, 1)
        dl2_scores = self._psi_score(self.mh_less(_to_tensor(dls)), _to_tensor(dls.flatten()))
        
        mh_contribution = torch.zeros(28,)
        for jdx in range(len(del_lens)):
            dl = del_lens[jdx]
            if dl > 28:
                continue
            mask = torch.cat((torch.zeros(dl - 1,), torch.ones(1, ) * mh_scores[jdx], torch.zeros(28 - (dl - 1) - 1,)))
            mh_contribution = mh_contribution + mask
        y_mh_less_phi = dl2_scores + mh_contribution.reshape(-1, 1)

        return y_mh_phi.flatten(), mh_indels, y_mh_less_phi.flatten(), mh_less_indels

    def predict(self, x_mh, experiment_name=None):
        self.eval()
        with torch.no_grad():
            mh_indels = x_mh.index.to_list()
            del_lens = x_mh["Size"]
            mh_len = x_mh["homologyLength"]
            #################################################
            # Predict MH Deletions
            #################################################
            # predict mh
            mh_scores = self._psi_score(self.mh(_to_tensor(x_mh[EXPERIMENTS[experiment_name]], _to_tensor(del_lens))))

            # add mh-less contributions at full MH lengths
            mh_vector = x_mh["homologyLength"]
            mhfull_contribution = torch.zeros(mh_vector.shape)
            for jdx in range(len(mh_vector)):
                if del_lens[jdx] == mh_vector[jdx]:
                    mhless_score = self._psi_score(self.mh_less(_to_tensor(x_mh[jdx:jdx+1][MH_LESS_FEATURES])), _to_tensor(del_lens[jdx]))[0,0]
                    mask = torch.cat((torch.zeros(jdx,), torch.ones(1,) * mhless_score, torch.zeros(len(mh_vector) - jdx - 1,)))
                    mhfull_contribution = mhfull_contribution + mask
            mhfull_contribution = mhfull_contribution.reshape(-1, 1)
            y_mh_phi = mh_scores + mhfull_contribution


            nonfull_dls = []
            for dl in range(1, 29):
                if dl not in del_lens:
                    nonfull_dls.append(dl)
                elif del_lens.count(dl) == 1:
                    idx = del_lens.index(dl)
                    if mh_len[idx] != dl:
                        nonfull_dls.append(dl)
                else:
                    nonfull_dls.append(dl)

            y_mh_less_phi = self._psi_score(self.mh_less(_to_tensor(nonfull_dls).reshape(-1, 1)), _to_tensor(nonfull_dls.flatten()))
        return self.normalise(self, y_mh_phi, mh_indels, y_mh_less_phi, nonfull_dls)

    def normalise(self, y_mh_phi, indels, y_mh_less_phi, nonfull_dls):
        y_pred = torch.cat((y_mh_phi, y_mh_less_phi))
        y_pred = y_pred/sum(y_pred)
        genotypes = indels + ["D" + str(dl) for dl in nonfull_dls]
        return y_pred, genotypes

def init_model(num_features):
    model = DualNeuralNetwork(num_features).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, betas=(0.99, 0.999), eps=10**-8)
    learning_rate_decay = 0.999
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=learning_rate_decay)
    loss_fn = mse_loss
    return model, optimizer, lr_scheduler, loss_fn

def batch(X_mh, y_mh, y_mh_less, samples, model, loss_fn, optimizer, lr_scheduler):
    loss = torch.ones(1).to(DEVICE)

    for s in samples:
        # get mh features
        X_mh_s = X_mh.loc[s]
        X_mh_less_s = y_mh_less.loc[s].index.to_numpy()
        # get predictions
        y_mh_phi, _, y_mh_less_phi, _ = model(X_mh_s, X_mh_less_s)

        # calculate losses
        y_mh_pred_norm = y_mh_phi/sum(y_mh_phi)
        y_mh_s = _to_tensor(y_mh.loc[s])
        y_mh_s_norm = y_mh_s/sum(y_mh_s)
        rsq1 = loss_fn(y_mh_pred_norm, y_mh_s_norm)
    
        y_mh_less_s = _to_tensor(y_mh_less.loc[s]).flatten()
        y_mh_less_s_norm = y_mh_less_s/sum(y_mh_less_s)
        y_mh_less_pred_norm = y_mh_less_phi/sum(y_mh_less_phi)
        rsq2 = loss_fn(y_mh_less_pred_norm, y_mh_less_s_norm) 
        if rsq1.isnan() or rsq2.isnan():
            print(s)
            continue  

        loss += rsq1
        loss += rsq2

    loss = loss/len(samples)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    lr_scheduler.step()

    batch_loss = loss.cpu().detach().numpy()[0]

    return batch_loss

def test(X_mh, y_mh, y_mh_less, samples, model):
    model.eval()
    metrics1 = []
    metrics2 = []
    metrics3 = []
    with torch.no_grad():
        for s in samples:
            X_mh_s = X_mh.loc[s]
            X_mh_less_s = y_mh_less.loc[s].index.to_numpy()
            # get predictions
            y_mh_phi, _, y_mh_less_phi, _ = model(X_mh_s, X_mh_less_s)

            y_mh_pred_norm = y_mh_phi/sum(y_mh_phi)
            y_mh_s = _to_tensor(y_mh.loc[s])
            y_mh_s_norm = y_mh_s/sum(y_mh_s)
            corr1 = pearsonr(y_mh_pred_norm, y_mh_s_norm)

            y_mh_less_pred_norm = y_mh_less_phi/sum(y_mh_less_phi)
            y_mh_less_s = _to_tensor(y_mh_less.loc[s]).flatten()
            y_mh_less_s_norm = y_mh_less_s/sum(y_mh_less_s)
            corr2 = pearsonr(y_mh_less_pred_norm, y_mh_less_s_norm)
            
            if corr1.isnan() or corr2.isnan():
                continue
            metrics1.append(corr1)
            metrics2.append(corr2)

            # overall 
            y_pred = torch.cat((y_mh_phi, y_mh_less_phi))
            y_pred = y_pred/y_pred.sum()

            y = torch.cat((y_mh_s_norm, y_mh_less_s_norm))
            y = y/y.sum()
            metrics3.append(pearsonr(y_pred, y)) 

    metric1 = np.mean(metrics1)
    metric2 = np.mean(metrics2)
    metric3 = np.mean(metrics3)
    return metric1, metric2, metric3

def load_model(experiment_name="full", min_num_reads=MIN_READS_PER_TARGET):
    model_d = "./models/{}x_{}_2NN_model.pth".format(min_num_reads, experiment_name)
    model = torch.load(model_d)
    return model

def write(writer, loss, train_metric, val_metric, epoch, cv=""):
    writer.add_scalar("training_loss{}".format(cv), loss, epoch)
    writer.add_scalar("nn1_training_correlation{}".format(cv), train_metric[0], epoch)
    writer.add_scalar("nn2_training_correlation{}".format(cv), train_metric[1], epoch)
    if val_metric is not None:
        writer.add_scalar("nn1_validation_correlation{}".format(cv), val_metric[0], epoch)
        writer.add_scalar("nn2_validation_correlation{}".format(cv), val_metric[1], epoch)

def train(X_mh, y_mh, y_mh_less, samples, model, loss_fn, optimizer, lr_scheduler, writer=None, experiment_name = ""):
    print("Training on {} samples, with {} features".format(len(samples), X_mh.columns))
    train_metric = test(X_mh, y_mh, y_mh_less, samples, model)
    print("{}: Starting training loss: {}".format(experiment_name, train_metric))

    samples, val_samples = train_test_split(samples, test_size = 100, random_state=RANDOM_STATE)

    # for t in range(EPOCHS):
    pbar = trange(EPOCHS, desc = "Training {}...".format(experiment_name), leave=True)
    for t in pbar:
        permutation = torch.randperm(len(samples))
        for i in range(0, len(samples), BATCH_SIZE):
            samples_batch = samples[permutation[i:i+BATCH_SIZE]]
            loss = batch(X_mh, y_mh, y_mh_less, samples_batch, model, loss_fn, optimizer, lr_scheduler)
            train_metric = test(X_mh, y_mh, y_mh_less, samples_batch, model)
            val_metric = test(X_mh, y_mh, y_mh_less, val_samples, model)
            pbar.set_description("{}: Epoch {}, Train loss: {}, Train Pearson's Corr: {}, Validation Pearson's Corr: {}".format(experiment_name, t, loss, train_metric, val_metric))

            if writer is not None:
                write(writer, loss, train_metric, val_metric, t)

def run_experiment(X_mh, y_mh, y_mh_less, samples, experiment_name):
    # set up tensorboard logger
    writer = SummaryWriter(LOGS_DIR + experiment_name)

    # final training
    model, optimizer, lr_scheduler, loss_fn = init_model(len(EXPERIMENTS[sys.argv[1]]))
    train(X_mh, y_mh, y_mh_less, samples, model, loss_fn, optimizer, lr_scheduler, writer = writer, experiment_name=experiment_name)

    # save model
    os.makedirs(OUTPUT_MODEL_D, exist_ok=True)
    torch.save(model, OUTPUT_MODEL_F.format(experiment_name))

if __name__ == "__main__":
    LOGS_DIR = os.environ['LOGS_DIR']
    OUTPUT_MODEL_D = OUTPUT_DIR + "/model_training/model/OurModel_2NN/"
    OUTPUT_MODEL_F = OUTPUT_MODEL_D + "{}x".format(MIN_READS_PER_TARGET) + "_{}_2NN_model.pth"
    NUM_FOLDS = 10
    RANDOM_STATE = 1
    EPOCHS = 140
    BATCH_SIZE = 200
    # get devices
    # DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DEVICE = "cpu"
    print('Using {} device'.format(DEVICE))

    # set number of threads to 1 for various libraries
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'

    # set random seed
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)

    # load data
    X_mh, y_mh, y_mh_less, samples = load_data(num_samples=None)

    # use common samples for experiment consistency
    common_samples = get_common_samples(min_reads=MIN_READS_PER_TARGET)
    samples = np.intersect1d(samples, common_samples)

    if len(sys.argv) == 1:
        raise NoExperimentDefined()
    elif sys.argv[1] == "full":
        run_experiment(X_mh.loc[:, EXPERIMENTS[sys.argv[1]]], y_mh, y_mh_less, samples, experiment_name="full")
    elif sys.argv[1] == "inDelphi_features":
        run_experiment(X_mh.loc[:, EXPERIMENTS[sys.argv[1]] + ["Size"]], y_mh, y_mh_less, samples, experiment_name="inDelphi_features")
    elif sys.argv[1] == "full+numrepeat":
        run_experiment(X_mh.loc[:, EXPERIMENTS[sys.argv[1]]], y_mh, y_mh_less, samples, experiment_name="full+numrepeat")
    else:
        raise InvalidExperiment()

    print("Finished.")
