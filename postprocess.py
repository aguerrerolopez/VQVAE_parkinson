import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    roc_curve,
    auc,
)
from wandb.sklearn import (
    plot_confusion_matrix,
    plot_roc,
    plot_feature_importances,
)
import wandb


def eval_performance(model, x, y, wandb_flag=False):
    # Compute the accuracy of the model
    accuracy = model.score(x, y)
    print("Accuracy: ", accuracy)

    # Compute the predictions of the model
    y_pred = model.predict(x)
    y_pred_proba = model.predict_proba(x)[:, 1]

    # Compute the confusion matrix
    cm = confusion_matrix(y, y_pred)
    print("Confusion Matrix: \n", cm)

    # Compute the precision and recall
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    print("Precision: ", precision)
    print("Recall: ", recall)

    # Compute ROC curves and ROC area for each class
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    print("ROC AUC: ", roc_auc)

    if wandb_flag:
        # Log the metrics
        wandb.log(
            {
                "test/accuracy": accuracy,
                "test/precision": precision,
                "test/recall": recall,
                "test/roc_auc": roc_auc,
            }
        )

    # Plot ROC curve
    plt.figure()
    lw = 2
    plt.plot(
        fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.show()
    plt.close()

    if wandb_flag:
        # Compute the confusion matrix
        # Binarize y
        plot_confusion_matrix(y, y_pred, ["HC", "PD"])

        # Compute the ROC curve
        y_pred_proba = model.predict_proba(x)
        plot_roc(y, y_pred_proba, ["HC", "PD"])

        # Compute the feature importances
        plot_feature_importances(model)
