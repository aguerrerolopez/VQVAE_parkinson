import wandb
import yaml
import numpy as np
from preprocess import read_data
from postprocess import eval_performance
from preprocess import extract_rasta_plp_with_derivatives
from matplotlib import pyplot as plt
import librosa
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

# Read the configuration file
with open("config_real.yaml", "r") as stream:
    config = yaml.safe_load(stream)


# ================== Hyperparameters ================== TODO: Change this for argparse
hyperparams = {
    "frame_size_ms": 15,
    "hop_size_percent": 50,
    "n_plps": 12,
    "wandb_flag": True,
}
wandb_flag = hyperparams["wandb_flag"]

mpath = config["main"]["path_to_data"]

problem = "PATAKA"


path_to_data = mpath + "/" + problem + "/"

if wandb_flag:
    wandb.init(
        project="parkinson",
        entity="alexjorguer",
        # The name of the group is the combinatno of hyperparameters
        group="Baseline RF plps" + str(hyperparams),
        name="Preprocessing",
        config=hyperparams,
    )

print("Reading data...")

# Read and preprocess the data
data = read_data(path_to_data, hyperparams, wandb_flag=wandb_flag)

print("Extracting PLPS...")
# Compute the PLPS. Apply to each frame the plp function with receives as input the frame, the sampling rate
data["plps_with_derivatives"] = data.apply(
    lambda x: extract_rasta_plp_with_derivatives(
        x["norm_signal"],
        x["sr"],
        hyperparams["frame_size_ms"],
        hyperparams["n_plps"],
    ),
    axis=1,
)

# Plot a random plps_with_derivatives using librosa specshow
plt.figure()
librosa.display.specshow(
    data["plps_with_derivatives"][0].T, sr=data["sr"][0], x_axis="time"
)
plt.title("PLPs with derivatives")
plt.colorbar()
plt.tight_layout()
plt.savefig("./results/plps_with_derivatives.png")
plt.close()


# Explode
data = data.explode("plps_with_derivatives")


if wandb_flag:
    wandb.finish()

# ============== Splitting ===========
folds = np.unique(data["fold"])
for f in folds:
    if wandb_flag:
        # Initialize wandb
        wandb.init(
            project="parkinson",
            entity="alexjorguer",
            # The name of the group is the combinatno of hyperparameters
            group="Baseline RF plps" + str(hyperparams),
            name="Fold " + str(f),
            config=hyperparams,
        )
    # Select randomly one fold for testing and the rest for training
    train = data[data["fold"] != f]
    test = data[data["fold"] == f]

    # Check the balance in trianing sets and store the weights of each class to compensate training
    weights = train["label"].value_counts(normalize=True)

    # Define a columns vector with columns name: the first 12 are the plp_coeffs and the rest are the plp_deltas1 adn deltas2
    columns = (
        ["plp_" + str(i) for i in range(1, 13)]
        + ["plp_delta_" + str(i) for i in range(1, 13)]
        + ["plp_delta2_" + str(i) for i in range(1, 13)]
    )

    X_train = np.vstack(train["plps_with_derivatives"])
    y_train = train["label"]
    # binarize the labels
    y_train = np.where(y_train == "PD", 1, 0)

    X_test = np.vstack(test["plps_with_derivatives"])
    y_test = test["label"]
    # binarize the labels
    y_test = np.where(y_test == "PD", 1, 0)

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


