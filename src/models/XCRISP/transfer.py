import os, sys, random
import pandas as pd
import pickle as pkl
import numpy as np
from sklearn.metrics import mean_gamma_deviance
from tqdm import trange
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchsummary import summary
from torch.nn.functional import kl_div, mse_loss
from torch.multiprocessing import Process
from torch.utils.tensorboard import SummaryWriter
from audtorch.metrics.functional import pearsonr
from sklearn.model_selection import KFold, train_test_split

from src.models.XCRISP.features import DELETION_FEATURES
from src.models.XCRISP.bins import bin_repair_outcomes_by_length

from src.data.data_loader import get_common_samples
from src.config.test_setup import MIN_NUMBER_OF_READS

FEATURES = {
    "v4": ["Gap", "leftEdge", "rightEdge", "homologyLength", "homologyGCContent"]
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
    y = counts.loc[counts.Type == "DELETION"]
    y = y.fraction if fractions else y.countEvents
    del_features["Gap"] = del_features["Size"] - del_features["homologyLength"]
    X = del_features[DELETION_FEATURES + ["Gap"]]
    return X, y, samples

def get_folds(s):
    kf = KFold(n_splits=NUM_FOLDS, random_state=RANDOM_STATE, shuffle=True)
    folds = list(kf.split(s))
    return folds

def init_weights(m):
    if type(m) == nn.Linear:
        m.weight = nn.Parameter(torch.randn(m.weight.shape) * 0.1)
        m.bias = nn.Parameter(torch.randn(m.bias.shape) * 0.1)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_sigmoid_stack = nn.Sequential(
            nn.Linear(5, 16),
            nn.Sigmoid(),
            nn.Linear(16, 16),
            nn.Sigmoid(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.linear_sigmoid_stack(x)
        return out

class TransferNeuralNetwork(nn.Module):
    def __init__(self, pretrained_model):
        super(TransferNeuralNetwork, self).__init__()
        trained_layers = list(pretrained_model.linear_sigmoid_stack.children())
        removed = trained_layers[:-2]
        remaining = trained_layers[-2:]
        self.pretrained_model = torch.nn.Sequential(*removed)
        self.new_layers = torch.nn.Sequential(*remaining)
        print(self.pretrained_model)
        print(self.new_layers)

    def forward(self, x):
        out = self.pretrained_model(x)
        out = self.new_layers(out)
        return out

class TransferNeuralNetworkOneFrozenLayer(nn.Module):
    def __init__(self, pretrained_model):
        super(TransferNeuralNetworkOneFrozenLayer, self).__init__()
        trained_layers = list(pretrained_model.linear_sigmoid_stack.children())
        removed = trained_layers[:2]
        remaining = trained_layers[2:]
        self.pretrained_model = torch.nn.Sequential(*removed)
        self.new_layers = torch.nn.Sequential(*remaining)
        print(self.pretrained_model)
        print(self.new_layers)

    def forward(self, x):
        out = self.pretrained_model(x)
        out = self.new_layers(out)
        return out

class TransferNeuralNetworkWithOneExtraHiddenLayer(nn.Module):
    def __init__(self, pretrained_model):
        super(TransferNeuralNetworkWithOneExtraHiddenLayer, self).__init__()
        removed = list(pretrained_model.linear_sigmoid_stack.children())[:-2]
        self.pretrained_model = torch.nn.Sequential(*removed)
        self.new_layers = nn.Sequential(
            nn.Linear(16, 16),
            nn.Sigmoid(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.new_layers.apply(init_weights)
        print(self.pretrained_model)
        print(self.new_layers)

    def forward(self, x):
        out = self.pretrained_model(x)
        out = self.new_layers(out)
        return out


def batch(X, Y, samples, model, loss_fn, optimizer, lr_scheduler):
    loss = torch.zeros(1).to(DEVICE)

    # for now we do gradient descent, but should make this full grad descent
    for s in samples:
        # Get features for input sequence
        x = _to_tensor(X.loc[s])
        # compute loss
        y = _to_tensor(Y.loc[s])
        y_pred = model(x)[:,0]
        y_pred = y_pred/y_pred.sum()
        if y.sum() == 0:
            continue

        # binning
        loss += loss_fn(torch.log(y_pred), y)

    loss = torch.div(loss, len(samples))

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # lr_scheduler.step()

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
            if y.sum() == 0:
                continue

            metrics.append((pearsonr(y_pred, y).cpu().detach().numpy()[0], \
                mse_loss(y_pred, y).cpu().detach().numpy()))
    metric = np.mean(metrics, axis=0)
    return tuple(metric)

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


def train(X, Y, samples, model, loss_fn, optimizer, lr_scheduler, writer=None, experiment_name = "", num_samples=100, mode=None):
    
    samples, val_samples = load_transfer_learning_samples(TRAINING_DATA, num_samples)

    train_corr, train_loss = test(X, Y, samples, model)
    val_corr, val_loss = test(X, Y, val_samples, model)
    print("{}, {}: Starting training corr: {} , validation corr: {}".format(experiment_name, mode, train_corr, val_corr))

    pbar = trange(0, EPOCHS if mode != "finetune" else FINE_TUNING_EPOCHS, desc = "Training {}...".format(experiment_name), leave=True)
    all_metrics = []
    for t in pbar:
        permutation = np.random.permutation(len(samples))
        train_corr, train_loss = test(X, Y, samples, model)
        val_corr, val_loss = test(X, Y, val_samples, model)
        for i in range(0, len(samples), BATCH_SIZE):
            samples_batch = samples[permutation[i:i+BATCH_SIZE]]
            loss = batch(X, Y, samples_batch, model, loss_fn, optimizer, lr_scheduler)
        metrics = (t, train_loss, train_corr, val_corr, val_loss)
        pbar.set_description("{}: Epoch {}, Train loss: {:.5f}, Train Pearson's Corr: {:.5f}, Validation Pearson's Corr: {:.5f}".format(experiment_name, t, *metrics[1:4]))
        all_metrics.append(metrics)
        plot(all_metrics, experiment_name, mode)

        if writer is not None:
            write(writer, loss, train_corr, val_corr, t, samples, X, Y, model)

def plot(metrics, experiment_name, mode):
    metrics = pd.DataFrame(metrics, columns=["Epoch", "Loss", "Training Corr", "Validation Corr", "Validation Loss"])

    sns.set_style("ticks",{'axes.grid' : True})

    plt.figure()
    losses = metrics[["Epoch", "Loss", "Validation Loss"]]
    losses = losses.melt('Epoch', var_name='Loss', value_name='Value')
    lineplot = sns.lineplot(data=losses, x="Epoch", y="Value", hue="Loss")
    fig = lineplot.get_figure()
    fig.tight_layout()
    fig.savefig(LOGS_DIR + experiment_name + "_" + mode + "_training_loss_curve.png")
    plt.close(fig)

    plt.figure()
    corrs = metrics[["Epoch", "Training Corr", "Validation Corr"]]
    corrs = corrs.melt('Epoch', var_name='Corr', value_name='Value')
    lineplot2 = sns.lineplot(data=corrs, x="Epoch", y="Value", hue="Corr")
    fig2 = lineplot2.get_figure()
    fig2.tight_layout()
    fig2.savefig(LOGS_DIR + experiment_name + "_" + mode + "_training_correlation_curve.png")
    plt.close(fig2)

def init_model(mode="pretrained", model=None):
    assert mode in ["pretrained", "baseline", "finetune", "pretrainedsamearch", "pretrainedplusonelayer", "pretrainedonefrozenlayer", "weightinit"]
    if mode == "finetune":
        assert model is not None

    # normal_learning_rate = 0.01
    # finetune_learning_rate = 0.0002
    normal_learning_rate = 0.05
    finetune_learning_rate = 0.0005


    if mode == "pretrained":
        # load pretrained model
        pretrained_model = load_pretrained_model()
        # instansiate new model with pretrained weights
        model = TransferNeuralNetwork(pretrained_model).to(DEVICE)
        # freeze weights of pretrained layers so to not lose information
        for parameter in model.pretrained_model.parameters():
            parameter.requires_grad = False
        # set up optimizers to begin learning
        optimizer = torch.optim.Adam(model.parameters(), normal_learning_rate, betas=(0.99, 0.999))
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.999)

    elif mode == "pretrainedplusonelayer":
        # load pretrained model
        pretrained_model = load_pretrained_model()
        # instansiate new model with pretrained weights
        model = TransferNeuralNetworkWithOneExtraHiddenLayer(pretrained_model).to(DEVICE)
        # freeze weights of pretrained layers so to not lose information
        for parameter in model.pretrained_model.parameters():
            parameter.requires_grad = False
        # set up optimizers to begin learning
        optimizer = torch.optim.Adam(model.parameters(), normal_learning_rate, betas=(0.99, 0.999))
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.999)

    elif mode == "pretrainedonefrozenlayer":
        # load pretrained model
        pretrained_model = load_pretrained_model()
        # instansiate new model with pretrained weights
        model = TransferNeuralNetworkOneFrozenLayer(pretrained_model).to(DEVICE)
        # freeze weights of pretrained layers so to not lose information
        for parameter in model.pretrained_model.parameters():
            parameter.requires_grad = False
        # set up optimizers to begin learning
        optimizer = torch.optim.Adam(model.parameters(), normal_learning_rate, betas=(0.99, 0.999))
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.999)

    elif mode == "pretrainedsamearch":
        model = load_pretrained_model()
        optimizer = torch.optim.Adam(model.parameters(), normal_learning_rate, betas=(0.99, 0.999))
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.999)

    elif mode == "baseline":
        model = NeuralNetwork().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), normal_learning_rate, betas=(0.99, 0.999))
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.999)

    elif mode == "weightinit":
        # unfreeze weights for finetuning
        model = load_pretrained_model()
        optimizer = torch.optim.Adam(model.parameters(), normal_learning_rate, betas=(0.99, 0.999))
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.999)
    
    elif mode == "finetune":
        # unfreeze weights for finetuning
        if isinstance(model, TransferNeuralNetwork) or isinstance(model, TransferNeuralNetworkWithOneExtraHiddenLayer):
            for parameter in model.pretrained_model.parameters():
                parameter.requires_grad = True
        optimizer = torch.optim.Adam(model.parameters(), finetune_learning_rate, betas=(0.99, 0.999))
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.999)

    loss_fn = kl_div
    return model, optimizer, lr_scheduler, loss_fn

def run_experiment(X, y, samples, experiment_name, do_CV=False, mode="pretrained", num_samples = 100):
    # set up tensorboard logger
    # writer = SummaryWriter(LOGS_DIR + experiment_name)
    writer = None

    if do_CV:
        cv_folds = []
        # cross validation
        folds = get_folds(samples)
        for i, f in enumerate(folds):
            print("Training fold {} with {} samples, {} features".format(i, len(samples[f[0]]), X.shape[1]))
            model, optimizer, lr_scheduler, loss_fn =init_model()
            loss, train_corr, val_corr = cv(X, y, samples[f[0]], samples[f[1]], i, model, loss_fn, optimizer, lr_scheduler, experiment_name=experiment_name, writer=writer)
            cv_folds.append((loss, train_corr, val_corr))

        cv_df = pd.DataFrame(cv_folds, columns=["Loss", "Train Corr", "Val Corr"])
        cv_df.loc['mean'] = cv_df.mean()
        cv_df.to_csv(OUTPUT_MODEL_F.format(experiment_name, RANDOM_STATE).replace("pth", ".folds.tsv"), sep="\t")
        
    # training (only fine tune when using exact same model and not replacing the output layer)
    model, optimizer, lr_scheduler, loss_fn = init_model(mode)
    if mode != "pretrainedsamearch":
        train(X, y, samples, model, loss_fn, optimizer, lr_scheduler, writer = writer, experiment_name=experiment_name, num_samples=num_samples, mode=mode)

    # fine tuning
    model, optimizer, lr_scheduler, loss_fn = init_model("finetune", model)
    train(X, y, samples, model, loss_fn, \
        optimizer, lr_scheduler, \
        writer = writer, experiment_name=experiment_name, num_samples=num_samples, mode="finetune")

    # save model
    os.makedirs(OUTPUT_MODEL_D, exist_ok=True)
    torch.save(model, OUTPUT_MODEL_F.format(experiment_name, RANDOM_STATE))

    print(f"Model saved under {OUTPUT_MODEL_F.format(experiment_name, RANDOM_STATE)}")

    torch.save({
        "random_state": RANDOM_STATE,
        "model": model.state_dict(),
        "samples": samples,
        "loss_fn" : mse_loss,
        "optimiser" : optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict()
    }, OUTPUT_MODEL_F.format(experiment_name, RANDOM_STATE).replace("pth", "details"))


def load_pretrained_model(model_dir="./src/models/X-CRISP/models/"):
    model_d = "{}Sigmoid_100x_{}_BaseLoss_RS_1_model.pth".format(model_dir, FEATURE_SET)
    model = torch.load(model_d)
    return model

def load_model(mode, genotype_short_name, num_samples, model_dir="./src/models/XCRISP/models/"):
    model_d = "{}transfer_kld_{}/transfer_kld_{}_{}_RS{}.pth".format(model_dir, genotype_short_name, mode, num_samples, 1)
    model = torch.load(model_d)
    return model

def load_transfer_learning_samples(dataset, n):
    tr_samples = pd.read_pickle("./src/models/X-CRISP/transfer_{}.pkl".format(dataset))
    return np.array(tr_samples[n]), np.array(tr_samples["validation"])

# set global vars
OUTPUT_DIR = os.environ['OUTPUT_DIR'] if 'OUTPUT_DIR' in os.environ else "./data/Transfer"
MIN_READS_PER_TARGET = MIN_NUMBER_OF_READS
INPUT_F = OUTPUT_DIR + "/model_training/data_{}x".format(MIN_READS_PER_TARGET) + "/X-CRISP/{}.pkl" 
DEVICE = "cpu"

if __name__ == "__main__":
    NUM_VAL_SAMPLES = 300
    if sys.argv[1] != "gen_oligos":
        TRAINING_DATA = sys.argv[3]
        GENOTYPE_SHORT_NAME = sys.argv[4]

        LOGS_DIR = "./data/interim"
        os.makedirs(LOGS_DIR, exist_ok=True)
        OUTPUT_MODEL_D = OUTPUT_DIR + "/model_training/model/X-CRISP/transfer_kld_{}/".format(GENOTYPE_SHORT_NAME)
        OUTPUT_MODEL_F = OUTPUT_MODEL_D + "{}_RS{}.pth"
        NUM_FOLDS = 5
        RANDOM_STATE = 1
        EPOCHS = 200
        FINE_TUNING_EPOCHS = 300
        BATCH_SIZE = 200
        FEATURE_SET = "v4"
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

        # get program arguments
        m = sys.argv[1]
        n = sys.argv[2]
        # load data
        X, y, samples = load_data(dataset=TRAINING_DATA, num_samples=None)
        samples = get_common_samples(genotype=TRAINING_DATA, min_reads=100, include_FORECasT=False)
        n = len(samples) - NUM_VAL_SAMPLES if n == "max" else int(n)

        experiment_name = "transfer_kld_{}_{}".format(m, n)
        print("Training {} on {} samples".format(m ,n))
        run_experiment(X.loc[:, FEATURES[FEATURE_SET]], y, samples, experiment_name=experiment_name, mode=m, num_samples=n)

        print("Finished.")

    elif sys.argv[1] == "gen_oligos":
        transfer_samples = {}
        samples = get_common_samples(genotype=sys.argv[2], min_reads=100, include_FORECasT=False)
        samples = list(samples)
        random.shuffle(samples)
        transfer_samples["validation"] = samples[-NUM_VAL_SAMPLES:]
    
        for n in [2, 5, 10, 20, 50, 100, 200, 500, 2000, 1000, len(samples)-NUM_VAL_SAMPLES]:
            transfer_samples[n] = samples[:n]
        
        with open("transfer_{}.pkl".format(sys.argv[2]), "wb") as outfile:
            pkl.dump(transfer_samples, outfile)

        print("file generated.")    
        
