import numpy as np
import copy

from visualizations import animate_loading

def z_normalize(data):
    mu = np.mean(data,axis=0)
    sigma = np.std(data,axis=0)
    norm = (data - mu)/sigma
    return (norm, sigma,mu)

def gradient_descent(X_train, y_train, w_init, b_init, cost_function, gradient, alpha, num_iters):
    J_history = []
    w = copy.deepcopy(w_init)
    b = b_init

    for i in range(num_iters):
        dj_dw, dj_db = gradient(X_train, y_train, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        
        if i < num_iters:  
            cost = cost_function(X_train, y_train, w, b)
            J_history.append(cost)

        
        animate_loading(i + 1, num_iters, cost)

    
    print("\nGradient Descent Complete!")
    return w, b, J_history

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

def compute_cost(X_train,y_train, w,b):
    m=X_train.shape[0]
    
    cost = 0.0
    for i in range(m):                                
        f_wb_i = np.dot(X_train[i], w) + b           
        cost = cost + (f_wb_i - y_train[i])**2      
    cost = cost / (2 * m)                          
    return cost
