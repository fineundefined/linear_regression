import matplotlib.pyplot as plt
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

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



def predict_act_plot(data):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['House_Price'], label='Actual Prices', color='blue', linestyle='-', marker='o')
    plt.plot(data.index, data['Predicted_House_Price'], label='Predicted Prices', color='red', linestyle='--', marker='x')
    plt.title('Actual vs Predicted House Prices')
    plt.xlabel('Sample Index')
    plt.ylabel('House Price')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def predict_act_features_plot(X_train,y_train,predictions_denorm,X_features):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    for i, feature in enumerate(X_features):
        ax[i].scatter(X_train[:, i], y_train, label='Actual Prices', color='blue', alpha=0.6)
        ax[i].scatter(X_train[:, i], predictions_denorm, label='Predicted Prices', color='orange', alpha=0.6)
        ax[i].set_xlabel(feature)
        ax[i].set_title(f"{feature} vs Price")
    ax[0].set_ylabel("Price")
    ax[0].legend()
    fig.suptitle("Actual vs Predicted House Prices")
    plt.show()


def metrics(data):
    mae = mean_absolute_error(data['House_Price'], data['Predicted_House_Price'])
    mse = mean_squared_error(data['House_Price'], data['Predicted_House_Price'])
    rmse = np.sqrt(mse)
    r2 = r2_score(data['House_Price'], data['Predicted_House_Price'])
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R^2): {r2:.2f}")
    
    
def prediction_target(X_train,y_train,prediction_denorm):
    m, _ = X_train.shape
    for i in range(m):
        print(f"Prediction: {prediction_denorm[i]:0.2f}, Target value: {y_train[i]:0.2f}")


