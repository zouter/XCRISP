
import torch
import numpy as np
import pandas as pd
from tensorflow import keras

from src.models.XCRISP.deletion import load_model, NeuralNetwork, FEATURE_SETS
from src.models.XCRISP.indels import gen_indels_v3
from src.models.XCRISP.features import get_features, get_insertion_features
from src.models.Lindel.features import onehotencoder


INSERTION_MODEL_F = "./models/Lindel/100x_insertion.h5"
INDEL_MODEL_F = "./models/Lindel/100x_indel.h5"
XCRISPR_MODEL_D = "./models/XCRISP/"
OUTPUT_F = "sample_output.txt"

guide = {
    "TargetSequence": "CTCCTATAATTCTAATCACTACAAGTCAGGAATGCCTGCGTTTGGCCGTCCAGTTAGTAACAGAAGGTCAGGTAAGAGG",
    "PAM Index": 42,
}

if __name__ == "__main__":
    # load models
    deletion_model = load_model(model_dir=XCRISPR_MODEL_D, loss_fn="kld", learning_rate="0.05")
    insertion_model = keras.models.load_model(INSERTION_MODEL_F)
    indel_model = keras.models.load_model(INDEL_MODEL_F)
    print("Models loaded")

    # get features
    cutsite = guide["PAM Index"]-3
    seq = guide["TargetSequence"][cutsite-32:cutsite+32]
    indels = gen_indels_v3(seq, 32, max_deletion_length=30).set_index(["Indel"])
    del_features = get_features(indels[indels["Type"] == "DELETION"])
    ins_features = get_insertion_features(guide["TargetSequence"], guide["PAM Index"] - 3)
    print("Created features")

    # predict
    with torch.no_grad():
        # deletion predictions from our Model
        x = torch.tensor(del_features[FEATURE_SETS["v4"]].to_numpy()).float()
        ds = deletion_model(x)
        ds = ds/sum(ds)
        ds = ds.detach().numpy()[:, 0]

    # insertion predictions from Lindel
    seq = guide["TargetSequence"][guide["PAM Index"]-33: guide["PAM Index"] + 27]
    pam = {'AGG':0,'TGG':0,'CGG':0,'GGG':0}
    guide = seq[13:33]
    input_indel = onehotencoder(guide)
    input_ins   = onehotencoder(guide[-6:])
    dratio, insratio = indel_model.predict(np.matrix(input_indel))[0,:]
    ins = insertion_model.predict(np.matrix(input_ins))[0,:]


    # combine predictions from both models
    y_hat = np.concatenate((ds*dratio,ins*insratio),axis=None)

    # get labels
    ins_labels = ['1+A', '1+T', '1+C', '1+G', '2+AA', '2+AT', '2+AC', '2+AG', '2+TA', '2+TT', '2+TC', '2+TG', '2+CA', '2+CT', '2+CC', '2+CG', '2+GA', '2+GT', '2+GC', '2+GG', '3+X']
    indels = list(indels.index[indels.Type == "DELETION"]) + ins_labels

    results = pd.DataFrame({
        "indel": indels,
        "predicted_frequency": y_hat,
    })

    results.to_csv(OUTPUT_F, sep="\t")
    print("Predictions saved to", OUTPUT_F)



