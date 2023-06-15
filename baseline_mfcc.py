import wandb
import yaml
import numpy as np
from preprocess import read_data
from postprocess import eval_performance
from preprocess import extract_mfcc_with_derivatives
import pandas as pd

# Import random forest
from sklearn.ensemble import RandomForestClassifier

# Read the configuration file
with open("config_real.yaml", "r") as stream:
    config = yaml.safe_load(stream)


# ================== Hyperparameters ================== TODO: Change this for argparse
hyperparams = {
    "frame_size_ms": 15,
    "hop_size_percent": 50,
    "n_mfcc": 12,
    "wandb_flag": True,
}
wandb_flag = hyperparams["wandb_flag"]

mpath = config["main"]["path_to_data"]

problem = "PATAKA"

path_to_data = mpath + "/" + problem + "/"


print("Reading data...")

# Read and preprocess the data
data = read_data(path_to_data, hyperparams, wandb=False)

print("Extracting MFCCs...")
# Compute the MFCCs. Apply to each frame the mfcc function with receives as input the frame, the sampling rate
data["mfccs_with_derivatives"] = data.apply(
    lambda x: extract_mfcc_with_derivatives(
        x["norm_signal"],
        x["sr"],
        hyperparams["frame_size_ms"],
        hyperparams["n_mfcc"],
    ),
    axis=1,
)

# TODO: Preguntar: es necesario separar por frames? o se puede hacer directamente com heatmaps?
# Transpose data["mfccs_with_derivatives"] to have a list of lists of mfccs
data["mfccs_with_derivatives"] = data["mfccs_with_derivatives"].apply(
    lambda x: np.transpose(x)
)
# Explode
data = data.explode("mfccs_with_derivatives")

# ============== Splitting ===========
folds = np.unique(data["fold"])
for f in folds:
    if wandb_flag:
        # Initialize wandb
        wandb.init(
            project="parkinson",
            entity="alexjorguer",
            # The name of the group is the combinatno of hyperparameters
            group="Baseline RF MFCCS" + str(hyperparams),
            name="Fold " + str(f),
            config=hyperparams,
        )
    # Select randomly one fold for testing and the rest for training
    train = data[data["fold"] != f]
    test = data[data["fold"] == f]

    # Check the balance in trianing sets and store the weights of each class to compensate training
    weights = train["label"].value_counts(normalize=True)

    # Define a columns vector with columns name: the first 12 are the mfcc_coeffs and the rest are the mfcc_deltas1 adn deltas2
    columns = (
        ["mfcc_" + str(i) for i in range(1, 13)]
        + ["mfcc_delta_" + str(i) for i in range(1, 13)]
        + ["mfcc_delta2_" + str(i) for i in range(1, 13)]
    )

    X_train = np.vstack(train["mfccs_with_derivatives"])
    y_train = train["label"]
    # binarize the labels
    y_train = np.where(y_train == "PD", 1, 0)

    X_test = np.vstack(test["mfccs_with_derivatives"])
    y_test = test["label"]
    # binarize the labels
    y_test = np.where(y_test == "PD", 1, 0)

    # Cross-validate a RF model
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.utils.class_weight import compute_sample_weight

    # DEfine the param grid of a RFC
    param_grid = {
        "n_estimators": [100],
        "max_depth": [5, 10, 15],
        "class_weight": ["balanced"],
    }

    # Define the model
    clf = RandomForestClassifier(random_state=42)
    # Define the grid search
    grid_search = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=2,
    )
    # Fit the grid search
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_clf = grid_search.best_estimator_
    # Get the best parameters
    best_params = grid_search.best_params_

    eval_performance(best_clf, X_train, y_train, wandb_flag=False)

    # Eval performance in testing
    eval_performance(best_clf, X_test, y_test, wandb_flag=True)

    # Plot summary
    if wandb_flag:
        wandb.sklearn.plot_summary_metrics(best_clf, X_train, y_train, X_test, y_test)

        wandb.finish()
