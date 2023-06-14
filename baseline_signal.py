import wandb
import yaml
import numpy as np
from preprocess import read_data
from postprocess import eval_performance


# Read the configuration file
with open("config_real.yaml", "r") as stream:
    config = yaml.safe_load(stream)

hyperparams = {"frame_size_ms": 10, "hop_size_percent": 50}


mpath = config["main"]["path_to_data"]

problem = "PATAKA"

path_to_data = mpath + "/" + problem + "/"

# Read and preprocess the data
data, data_framed = read_data(path_to_data, hyperparams, wandb=False)


# ============== Splitting ===========
folds = np.unique(data_framed["fold"])
for f in folds:
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

    X_train = np.stack(train["framed_signal"])
    y_train = train["label"]
    # binarize the labels
    y_train = np.where(y_train == "PD", 1, 0)

    X_test = np.stack(test["framed_signal"])
    y_test = test["label"]
    # binarize the labels
    y_test = np.where(y_test == "PD", 1, 0)

    # Cross-validate a LR model simple
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.utils.class_weight import compute_sample_weight

    # Define your X_train and y_train

    # Compute class weights
    class_weights = compute_sample_weight("balanced", y_train)

    # Create the LinearRegressor
    clf = LogisticRegression()

    # Define the parameter grid for GridSearchCV
    param_grid = {"penalty": ["l1", "l2"], "C": [0.01, 0.1, 1, 10]}

    # Create the GridSearchCV object
    grid_search = GridSearchCV(clf, param_grid, cv=5, verbose=2, n_jobs=-1)

    # Fit the model with class weights
    grid_search.fit(X_train, y_train, sample_weight=class_weights)

    # Get the best model and its parameters
    best_clf = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Print the best parameters
    print("Best Parameters: ", best_params)

    # Eval performance in training
    eval_performance(best_clf, X_train, y_train, wandb=False)

    # Eval performance in testing
    eval_performance(best_clf, X_test, y_test, wandb=True)

    # Plot summary
    wandb.sklearn.plot_summary_metrics(best_clf, X_train, y_train, X_test, y_test)

    wandb.finish()
