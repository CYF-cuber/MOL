import numpy as np

def calculate_metrics(confusion_matrix):
    confusion_matrix = np.array(confusion_matrix)
    
    # Accuracy (ACC)
    total_samples = np.sum(confusion_matrix)
    correct_predictions = np.trace(confusion_matrix)
    accuracy = correct_predictions / total_samples if total_samples != 0 else 0
    
    # Weighted Average Recall (WAR)
    class_sums = np.sum(confusion_matrix, axis=1)
    recall_per_class = np.where(class_sums != 0, np.diag(confusion_matrix) / class_sums, 0)
    weighted_recall = np.sum(recall_per_class * class_sums) / total_samples if total_samples != 0 else 0
    
    # Unweighted Average Recall (UAR)
    unweighted_recall = np.mean(recall_per_class) if recall_per_class.size > 0 else 0
    
    # Weighted F1 Score (WF1)
    precision_sums = np.sum(confusion_matrix, axis=0)
    precision_per_class = np.where(precision_sums != 0, np.diag(confusion_matrix) / precision_sums, 0)
    f1_score_per_class = np.where((precision_per_class + recall_per_class) != 0, 2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class), 0)
    weighted_f1_score = np.sum(f1_score_per_class * class_sums) / total_samples if total_samples != 0 else 0
    
    # Unweighted F1 Score (UF1)
    unweighted_f1_score = np.mean(f1_score_per_class) if f1_score_per_class.size > 0 else 0
    
    return accuracy, weighted_recall, unweighted_recall, weighted_f1_score, unweighted_f1_score
