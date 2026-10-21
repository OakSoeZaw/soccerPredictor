import pandas as pd
import numpy as np

from sklearn.metrics import(
    accuracy_score,
    log_loss,
    confusion_matrix,
    classification_report
)

def evaluate_model(y_true, y_pred, y_proba, classes, y_train):
    # Model metrics
    print("Model accuracy:")
    print(accuracy_score(y_true, y_pred))

    print("\nModel log loss:")
    print(log_loss(y_true, y_proba, labels = classes))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred, labels=classes))
    print("Class order:", classes)

    print("\nClassification report:")
    print(classification_report(
        y_true,
        y_pred,
        labels = classes,
        zero_division= 0        
    ))

    #BaseLine1: always predict the majority training class

    majority_class = y_train.value_counts().idxmax()
    majority_predictions = np.full(len(y_true), majority_class)

    print("Majority class:", majority_class)
    print("Majority baseline accuracy:")
    print(accuracy_score(y_true, majority_predictions))

    #Baseline2: use training class frequencies as probabilities
    class_frequencies = (
        y_train.value_counts(normalize = True)
        .reindex(classes, fill_value=0)
        .to_numpy()
    )
    baseline_probabilities = np.tile(
        class_frequencies,
        (len(y_true), 1)    
    )

    print("\nClass-frequency baseline log:")
    print(log_loss(
        y_true,
        baseline_probabilities,
        labels = classes
    ))

    