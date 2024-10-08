import numpy as np

def calculate_metrics(confusion_matrix):
    confusion_matrix = np.array(confusion_matrix)
    
    # Accuracy (ACC)
    total_samples = np.sum(confusion_matrix)
    correct_predictions = np.trace(confusion_matrix)
    accuracy = correct_predictions / total_samples
    
    # Weighted Average Recall (WAR)
    weighted_recall = np.sum(np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1) * np.sum(confusion_matrix, axis=0)) / total_samples
    
    # Unweighted Average Recall (UAR)
    recall_per_class = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
    unweighted_recall = np.mean(recall_per_class)
    
    # Weighted F1 Score (WF1)
    precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
    recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
    f1_score_per_class = 2 * precision * recall / (precision + recall)
    weighted_f1_score = np.sum(f1_score_per_class * np.sum(confusion_matrix, axis=1) / total_samples)
    
    # Unweighted F1 Score (UF1)
    unweighted_f1_score = np.mean(f1_score_per_class)
    
    return accuracy, weighted_recall, unweighted_recall, weighted_f1_score, unweighted_f1_score
