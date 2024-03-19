from sklearn.metrics import accuracy_score
from collections import Counter

def weighted_accuracy(true_array, pred_array, weights):
    """
    Calculate weighted accuracy between true labels and predicted labels.
    
    Parameters:
        true_array (array-like): Array containing true labels (0s and 1s).
        pred_array (array-like): Array containing predicted labels (0s and 1s).
        weights (dict): Dictionary containing weights for each class.

    Returns:
        float: Weighted accuracy score.
    """
    sample_weights = [1-weights[true_label] for true_label in true_array]
    return accuracy_score(true_array, pred_array, sample_weight=sample_weights)

def calculate_weights(true_array):
    """
    Calculate weights based on the prevalence of 0s and 1s in the true_array.
    
    Parameters:
        true_array (array-like): Array containing true labels (0s and 1s).

    Returns:
        dict: Dictionary containing weights for each class.
    """
    counts = Counter(true_array)
    total_samples = len(true_array)
    weights = {label: count / total_samples for label, count in counts.items()}
    return weights