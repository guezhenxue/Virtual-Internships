import numpy as np

def compute_mean(scores, max_score=8.0):
    return np.mean(scores) / max_score

def compute_std(scores, max_score=8.0):
    return np.std(scores) / max_score

def compute_skewness(scores):
    scores = np.array(scores)
    mean = np.mean(scores)
    std = np.std(scores)
    if std < 1e-8:
        return 0.0
    return np.mean(((scores - mean)/std)**3)

def gini_coefficient(frequencies):
    frequencies = np.array(frequencies)
    n = len(frequencies)
    diff_sum = 0
    for i in range(n):
        for j in range(n):
            diff_sum += abs(frequencies[i] - frequencies[j])
    return diff_sum / (2 * n * np.sum(frequencies))