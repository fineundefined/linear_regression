import numpy as np
import matplotlib.pyplot as plt
import copy
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys

df = pd.read_csv("hf://datasets/RashidIqbal/house_prices_data/house_prices_data.csv")

def compute_cost(X_train,y_train, w,b):
    m=X_train.shape[0]
    
    cost = 0.0
    for i in range(m):                                
        f_wb_i = np.dot(X_train[i], w) + b           
        cost = cost + (f_wb_i - y_train[i])**2      
    cost = cost / (2 * m)                          
    return cost

def compute_gradient(X_train,y_train,w,b):
    m,n = X_train.shape
    dj_dw = np.zeros((n,))
    dj_db = 0
    for i in range(m):
        err = (np.dot(X_train[i],w) +b)- y_train[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X_train[i,j]
        dj_db = dj_db+err
    dj_dw = dj_dw/m
    dj_db = dj_db/m
    return dj_dw,dj_db


def animate_loading(iteration, total_iterations, cost):
    """
    Displays an animated loading indicator in the terminal.
    Args:
        iteration (int): Current iteration number.
        total_iterations (int): Total number of iterations.
        cost (float): Current cost value.
    """
    loading_symbols = ['|', '/', '-', '\\']  
    progress = (iteration / total_iterations) * 100  
    symbol = loading_symbols[iteration % len(loading_symbols)]  
    progressline = round(progress/10)
    progressline_dis = "="*(progressline-1)+">"+"-"*(11-progressline)
    sys.stdout.write(f"\r{symbol} Iteration {iteration}/{total_iterations} {progressline_dis} - Cost: {cost:.6f} - Progress: {progress:.2f}%")
    sys.stdout.flush()  


def gradient_descent(X_train, y_train, w_init, b_init, cost_function, gradient, alpha, num_iters):
    J_history = []
    w = copy.deepcopy(w_init)
    b = b_init

    for i in range(num_iters):
        dj_dw, dj_db = gradient(X_train, y_train, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        
        if i < 100000:  
            cost = cost_function(X_train, y_train, w, b)
            J_history.append(cost)

        
        animate_loading(i + 1, num_iters, cost)

    
    print("\nGradient Descent Complete!")
    return w, b, J_history
  
def z_normalize(data):
    mu = np.mean(data,axis=0)
    sigma = np.std(data,axis=0)
    norm = (data - mu)/sigma
    return (norm, sigma,mu)
        
if __name__ == "__main__":
    
    X_train = df[['Square_Footage', 'Num_Bedrooms', 'House_Age']].to_numpy()
    y_train = df['House_Price'].to_numpy()
    
    feature_columns = ['Square_Footage', 'Num_Bedrooms', 'House_Age']

    
    X_train_df = pd.DataFrame(X_train, columns=feature_columns)
    y_train_df = pd.DataFrame(y_train, columns=['House_Price'])

    
    
    data = pd.concat([X_train_df, y_train_df], axis=1)
    
    
    
    initial_w = np.zeros(X_train.shape[1])  
    initial_b = 0.0
    
   
    iterations = 100000
    alpha = 1.0e-2
    X_norm, x_sigma,x_mu = z_normalize(X_train)
    y_norm,y_sigma,y_mu = z_normalize(y_train)
   
    w_final, b_final, J_hist = gradient_descent(X_norm, y_norm, initial_w, initial_b,
                                                compute_cost, compute_gradient, 
                                                alpha, iterations)
    
    print(f"b, w found by gradient descent: {b_final:0.2f}, {w_final}")
    m, _ = X_train.shape
    all_predictions = []
    for i in range(m):
        prediction_norm = np.dot(X_norm[i], w_final) + b_final
        prediction_denorm = (prediction_norm * y_sigma)+y_mu
        print(f"Prediction: {prediction_denorm:0.2f}, Target value: {y_train[i]}")
    
        
    predictions_norm = np.dot(X_norm, w_final) + b_final
    predictions_denorm = (predictions_norm * y_sigma) + y_mu

    
    data['Predicted_House_Price'] = predictions_denorm

    
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['House_Price'], label='Actual Prices', color='blue', linestyle='-', marker='o')
    plt.plot(data.index, data['Predicted_House_Price'], label='Predicted Prices', color='red', linestyle='--', marker='x')
    plt.title('Actual vs Predicted House Prices')
    plt.xlabel('Sample Index')
    plt.ylabel('House Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    
    mae = mean_absolute_error(data['House_Price'], data['Predicted_House_Price'])
    mse = mean_squared_error(data['House_Price'], data['Predicted_House_Price'])
    rmse = np.sqrt(mse)
    r2 = r2_score(data['House_Price'], data['Predicted_House_Price'])
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R^2): {r2:.2f}")

    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    X_features = ['Square_Footage', 'Num_Bedrooms', 'House_Age']
    for i, feature in enumerate(X_features):
        ax[i].scatter(X_train[:, i], y_train, label='Actual Prices', color='blue', alpha=0.6)
        ax[i].scatter(X_train[:, i], predictions_denorm, label='Predicted Prices', color='orange', alpha=0.6)
        ax[i].set_xlabel(feature)
        ax[i].set_title(f"{feature} vs Price")
    ax[0].set_ylabel("Price")
    ax[0].legend()
    fig.suptitle("Actual vs Predicted House Prices")
    plt.show()
