# Classification: Gradient Boosting

# This script performs Gradient Boosting classification.
# It uses the scikit-learn library to train a Gradient Boosting model on a given dataset and evaluate its performance.

# requirements: sklearn, numpy
# Install the required packages using the following command:
## pip install scikit-learn numpy

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

def gradient_boosting_classifier(
    X, y, test_size=0.2, random_state=42,
    n_estimators=100, learning_rate=0.1, max_depth=3
):
    """
    Perform Gradient Boosting classification and return model performance metrics.
    
    Parameters:
        X (array-like): Feature matrix
        y (array-like): Target variable
        test_size (float): Proportion of dataset to include in test split
        random_state (int): Seed for reproducibility
        n_estimators (int): Number of boosting stages
        learning_rate (float): Shrinks the contribution of each tree
        max_depth (int): Maximum depth of individual estimators
    
    Returns:
        dict: Accuracy and classification report
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state
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

    gb_results = gradient_boosting_classifier(X_example, y_example)

    print("\n" + "="*60)
    print("Gradient Boosting Classifier Results")
    print("="*60)
    print(f"Accuracy: {gb_results['Accuracy']:.4f}\n")
    print("Classification Report:")
    print(gb_results['Classification Report'])

### End of file ******************************************************************************************************