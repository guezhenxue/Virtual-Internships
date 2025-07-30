from components import compute_mean, compute_std, compute_skewness, gini_coefficient
import numpy as np

def decompose_loss_components(scores, speaking_freq):
    max_score = 8.0
    mu = compute_mean(scores, max_score)
    std = compute_std(scores, max_score)
    skewness = compute_skewness(scores)
    gini = gini_coefficient(speaking_freq)

    # Apply transformations
    mean_loss = 1 - mu
    std_loss = 1 + std  
    skew_loss = skewness  
    gini_loss = gini  

    return {
        '1 - mean': mean_loss,
        'std': std_loss,
        'abs(skew)': abs(skew_loss),
        'gini': gini_loss
    }

def total_weighted_loss(weights, components_list, lambda_reg=1e-4):
    alpha, beta, gamma, delta = weights
    total_loss = 0
    for comp in components_list:
        loss = (
            alpha * comp['1 - mean'] +
            beta * comp['std'] +
            gamma * comp['abs(skew)'] +
            delta * comp['gini']
        )
        total_loss += loss
    
    # Regularization term (L2 regularization)
    reg_term = lambda_reg * np.sum(np.square(weights))
    
    return total_loss / len(components_list) + reg_term

def gradient(weights, components_list, epsilon=1e-3):
    """
    Gradient function of total_weighted_loss, using numerical approximation.
    """
    grad = np.zeros_like(weights)
    base_loss = total_weighted_loss(weights, components_list)
    
    for i in range(len(weights)):
        perturbed = weights.copy()
        perturbed[i] += epsilon
        grad[i] = (total_weighted_loss(perturbed, components_list) - base_loss) / epsilon
        
    return grad

def gradient_descent(gradient, start, learn_rate, components_list, n_iter=50, tolerance=1e-6, decay=0.01):
    vector = start.copy()
    loss_history = []  # Initialize list to store loss values
    
    for n in range(n_iter):
        grad = gradient(vector, components_list)
        diff = -learn_rate * np.array(grad)
        
        if np.sum(np.abs(diff)) <= tolerance:
            print(f"Converged after {n} iterations.")
            break
        
        vector = vector + diff
        current_loss = total_weighted_loss(vector, components_list)
        loss_history.append(current_loss)  # Store current loss
        
        # Optionally decay the learning rate after each step
        learn_rate /= (1 + decay * n)
    
    return vector, loss_history