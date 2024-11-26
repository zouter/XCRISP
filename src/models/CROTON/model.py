import os, sys
import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import optimizers
import keras
from src.data.data_loader import get_common_samples

# set global vars
OUTPUT_DIR = os.environ['OUTPUT_DIR']
MIN_READS_PER_TARGET = 100
INPUT_F = OUTPUT_DIR + "/model_training/data_{}x".format(MIN_READS_PER_TARGET) + "/CROTON/{}.pkl" 
DEVICE = "cpu"
TRAINING_DATA = "train"

def load_data(dataset = "train", num_samples = None, fractions=True):
    data = pkl.load(open(INPUT_F.format(dataset), "rb"))
    y = data["stats"]
    X = data["input_seq"]
    samples = data["samples"]
    if num_samples is not None:
        red_samples = samples[:num_samples]
        idx = np.isin(samples, red_samples)
        X = X[idx, :, :]
        y = y[idx, :]
        samples = red_samples
    return X, y, samples

if __name__ == "__main__":
    X, y, samples = load_data(dataset=TRAINING_DATA, num_samples=None)
    # use common samples for experiment consistency
    common_samples = get_common_samples(genotype=TRAINING_DATA, min_reads=MIN_READS_PER_TARGET)
    train_samples, val_samples = train_test_split(samples, test_size = 100, random_state=1)
    idx_train_samples = np.isin(samples, train_samples)
    idx_common_samples = np.isin(samples, common_samples)
    X_train = X[idx_train_samples & idx_common_samples, :, :]
    y_train = y[idx_train_samples & idx_common_samples, :]

    idx_val_samples = np.isin(samples, val_samples)
    X_val = X[idx_val_samples & idx_common_samples, :, :]
    y_val = y[idx_val_samples & idx_common_samples, :]

    pre_trained_model = load_model("./src/models/CROTON/models/CROTON_pretrained.h5")
    print(pre_trained_model.summary())

    model = tf.keras.models.clone_model(pre_trained_model)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryCrossentropy()]
    )

    history = model.fit(
        X_train,
        y_train,
        batch_size = 100,
        epochs = 100,
        validation_data = (X_val, y_val)
    )

    model.save("./src/models/CROTON/models/CROTON_new.h5")

    print("Generate predictions for 3 samples")
    predictions = model.predict(X_val[:3])
    print("Predictions shape:", predictions.shape)
    print("Predictions[0]:", val_samples[0], predictions[0,:], y_val[0,:])
    print("Predictions[1]:", val_samples[1], predictions[1,:], y_val[1,:] )
    print("Predictions[2]:", val_samples[2], predictions[2,:], y_val[2,:] )

    print("Done.")
