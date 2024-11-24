import numpy as np






########################
from calculations import *
from dataset_prep import *
from visualizations import *



if __name__ == "__main__":
   
    
    X_train,y_train,feature_columns,data = dataset_preprocess()
    
    initial_w = np.zeros(X_train.shape[1])  
    initial_b = 0.0
    
    iterations = 10000
    alpha = 1.0e-3
    X_norm, x_sigma,x_mu = z_normalize(X_train)
    y_norm,y_sigma,y_mu = z_normalize(y_train)
   
    w_final, b_final, J_hist = gradient_descent(X_norm, y_norm, initial_w, initial_b,
                                                compute_cost, compute_gradient, 
                                                alpha, iterations)
    
    print(f"b, w found by gradient descent: {b_final:0.2f}, {w_final}")
    
    
    predictions_norm = np.dot(X_norm, w_final) + b_final
    predictions_denorm = (predictions_norm * y_sigma) + y_mu
    
    data['Predicted_House_Price'] = predictions_denorm
    prediction_target(X_train,y_train,predictions_denorm)
    metrics(data)
    predict_act_plot(data)
    predict_act_features_plot(X_train,y_train,predictions_denorm,feature_columns)
        
    

    
    
    
    