# Classification: Random Forest

# This script performs Random Forest classification.
# It uses the scikit-learn library to train a Random Forest model on a given dataset and evaluate its performance.

# requirements: sklearn, numpy
# Install the required packages using the following command:
## pip install scikit-learn numpy

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def random_forest_classifier(
    X, y, test_size=0.2, random_state=42,
    n_estimators=100, max_depth=None, min_samples_split=2
):
    """
    Perform Random Forest classification and return model performance metrics.
    
    Parameters:
        X (array-like): Feature matrix
        y (array-like): Target variable
        test_size (float): Proportion of dataset to include in test split
        random_state (int): Seed for reproducibility
        n_estimators (int): Number of trees in the forest
        max_depth (int or None): Maximum depth of each tree
        min_samples_split (int): Minimum samples required to split a node
    
    Returns:
        dict: Accuracy and classification report
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
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

    rf_results = random_forest_classifier(X_example, y_example)
    
    print("\n" + "="*60)
    print("Random Forest Classifier Results")
    print("="*60)
    print(f"Accuracy: {rf_results['Accuracy']:.4f}\n")
    print("Classification Report:")
    print(rf_results['Classification Report'])

### End of file ******************************************************************************************************