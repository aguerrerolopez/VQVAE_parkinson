import wandb
import yaml
import numpy as np
from preprocess import read_data
from postprocess import eval_performance
from preprocess import extract_mfcc_with_derivatives

# Import random forest
from sklearn.ensemble import RandomForestClassifier

# Read the configuration file
with open("config_real.yaml", "r") as stream:
    config = yaml.safe_load(stream)

hyperparams = {
    "frame_size_ms": 15,
    "hop_size_percent": 50,
    "n_mfcc": 12,
    "wandb_flag": False,
}


mpath = config["main"]["path_to_data"]

problem = "PATAKA"

path_to_data = mpath + "/" + problem + "/"


print("Reading data...")
# Read and preprocess the data
data, data_framed = read_data(path_to_data, hyperparams, wandb=False)

print("Extracting MFCCs...")
# Compute the MFCCs. Apply to each frame the mfcc function with receives as input the frame, the sampling rate
data_framed["mfcc"] = data_framed.apply(
    lambda x: extract_mfcc_with_derivatives(
        x["framed_signal"], x["sr"], hyperparams["n_mfcc"]
    ),
    axis=1,
)


# ============== Splitting ===========
folds = np.unique(data_framed["fold"])
for f in folds:
    if wandb_flag:
        # Initialize wandb
        wandb.init(
            project="parkinson",
            entity="alexjorguer",
            group="Baseline LR signal",
            name="Fold " + str(f),
            config=hyperparams,
        )
    # Select randomly one fold for testing and the rest for training
    train = data_framed[data_framed["fold"] != f]
    test = data_framed[data_framed["fold"] == f]

    # Check the balance in trianing sets and store the weights of each class to compensate training
    weights = train["label"].value_counts(normalize=True)

    # Define a columns vector with columns name: the first 12 are the mfcc_coeffs and the rest are the mfcc_deltas1 adn deltas2
    columns = (
        ["mfcc_" + str(i) for i in range(1, 13)]
        + ["mfcc_delta_" + str(i) for i in range(1, 13)]
        + ["mfcc_delta2_" + str(i) for i in range(1, 13)]
    )

    X_train = np.stack(train["mfcc"])
    y_train = train["label"]
    # binarize the labels
    y_train = np.where(y_train == "PD", 1, 0)

    X_test = np.stack(test["mfcc"])
    y_test = test["label"]
    # binarize the labels
    y_test = np.where(y_test == "PD", 1, 0)

    cl = RandomForestClassifier(
        n_estimators=100, max_depth=2, random_state=0, class_weight=weights
    )
    cl.fit(X_train, y_train)
    cl.score(X_train, y_train)
    cl.score(X_test, y_test)

    # Plot summary
    if wandb_flag:
        wandb.sklearn.plot_summary_metrics(best_clf, X_train, y_train, X_test, y_test)

        wandb.finish()
