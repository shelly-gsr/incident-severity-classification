# Classification: Ordinal Logistic Regression

# This script performs Ordinal Logistic Regression classification.
# It uses the mord library to train an Ordinal Logistic Regression model on a given dataset with
# both numeric and categorical features, and evaluates its performance.

# requirements: sklearn, numpy, pandas, mord
# Install the required packages using the following command:
## pip install scikit-learn numpy pandas mord

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# We use mord for ordinal logistic regression
# pip install mord
import mord as m

def ordinal_logistic_regression_classifier(
    X,
    y,
    numeric_features,
    categorical_features,
    test_size=0.2,
    random_state=42
):
    """
    Perform Ordinal Logistic Regression with numeric and categorical features.

    Parameters:
        X (pd.DataFrame): Feature matrix
        y (array-like): Target variable (ordinal)
        numeric_features (list): Names of numeric columns
        categorical_features (list): Names of categorical columns
        test_size (float): Test split proportion
        random_state (int): Seed

    Returns:
        dict: Accuracy, classification report, fitted pipeline
    """

    # ---------------------------
    # Train / Test Split
    # ---------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # ---------------------------
    # Preprocessing
    # ---------------------------
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # ---------------------------
    # Ordinal Logistic Regression Model
    # ---------------------------
    model = m.LogisticIT()  # Proportional odds / ordinal logistic regression

    # ---------------------------
    # Pipeline
    # ---------------------------
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # ---------------------------
    # Train & Evaluate
    # ---------------------------
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    results = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Classification Report': classification_report(y_test, y_pred),
        'Model': pipeline,
        'y_test': y_test,
        'y_pred': y_pred  
    }

    return results

from sklearn.utils.class_weight import compute_class_weight
import mord as m

def ordinal_class_weighted_classifier(
    X,
    y,
    numeric_features,
    categorical_features,
    test_size=0.2,
    random_state=42
):
    """
    Perform Ordinal Logistic Regression with class weighting
    to address target imbalance.
    """

    # ---------------------------
    # Train / Test Split
    # ---------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # IMPORTANT
    )

    # ---------------------------
    # Compute class weights
    # ---------------------------
    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )
    class_weight_dict = dict(zip(classes, class_weights))

    sample_weights = np.array([class_weight_dict[y] for y in y_train])

    # ---------------------------
    # Preprocessing
    # ---------------------------
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # ---------------------------
    # Ordinal Logistic Regression
    # ---------------------------
    model = m.LogisticIT()

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # ---------------------------
    # Train with sample weights
    # ---------------------------
    pipeline.fit(
        X_train,
        y_train,
        classifier__sample_weight=sample_weights
    )

    y_pred = pipeline.predict(X_test)

    return {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Classification Report': classification_report(y_test, y_pred),
        'Class Weights': class_weight_dict,
        'Model': pipeline,
        'y_test': y_test,
        'y_pred': y_pred 
    }


# ---------------------------
# Example Usage
# ---------------------------
if __name__ == "__main__":

    X_example = pd.DataFrame({
        'age': [25, 32, 40, 28, 50, 45],
        'income': [50000, 64000, 82000, 58000, 120000, 90000],
        'gender': ['F', 'M', 'M', 'F', 'M', 'F'],
        'road_type': ['urban', 'highway', 'highway', 'urban', 'rural', 'rural']
    })

    # Example ordinal target (0 = minor, 1 = moderate, 2 = severe)
    y_example = np.array([0, 0, 1, 0, 2, 1])

    numeric_features = ['age', 'income']
    categorical_features = ['gender', 'road_type']

    ord_results = ordinal_logistic_regression_classifier(
        X_example,
        y_example,
        numeric_features,
        categorical_features
    )

    print("\n" + "="*60)
    print("Ordinal Logistic Regression Results")
    print("="*60)
    print(f"Accuracy: {ord_results['Accuracy']:.4f}\n")
    print("Classification Report:")
    print(ord_results['Classification Report'])

### End of file ******************************************************************************************************
