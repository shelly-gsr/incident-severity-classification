# Classification: Logistic Regression

# This script performs Logistic Regression classification.
# It uses the scikit-learn library to train a Logistic Regression model on a given dataset and evaluate its performance.

# requirements: sklearn, numpy
# Install the required packages using the following command:
## pip install scikit-learn numpy

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def logistic_regression_classifier(X, y, test_size=0.2, random_state=42, penalty='l2', C=1.0, max_iter=1000):
    """
    Perform Logistic Regression classification and return model performance metrics.
    
    Parameters:
        X (array-like): Feature matrix
        y (array-like): Target variable
        test_size (float): Proportion of dataset to include in test split
        random_state (int): Seed for reproducibility
        penalty (str): Regularization type ('l1', 'l2', 'elasticnet', 'none')
        C (float): Inverse of regularization strength
        max_iter (int): Maximum number of iterations
    
    Returns:
        dict: Accuracy and classification report
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    model = LogisticRegression(
        penalty=penalty,
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        solver='lbfgs'
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Classification Report': classification_report(y_test, y_pred)
    }
    
    return results

# Example Usage
if __name__ == "__main__":
    X_example = np.array([[1], [2], [3], [4], [5], [6]])
    y_example = np.array([0, 0, 1, 1, 1, 0])

    lr_results = logistic_regression_classifier(X_example, y_example)

    print("\n" + "="*60)
    print("Logistic Regression Results")
    print("="*60)
    print(f"Accuracy: {lr_results['Accuracy']:.4f}\n")
    print("Classification Report:")
    print(lr_results['Classification Report'])

### End of file ******************************************************************************************************