import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import pandas as pd


# an underscore before a variable indicates that the variable should be private to the function/class
# this is correct usage indicating that lambda can take on different values outside of this function.
def compute_cost(X_train, Y_train, w, b, _lambda=2):
    m  = X_train.shape[0]
    # never used?
    n  = len(w)
    cost = 0.
    # If you have a function take a parameter don't redefine it. Either make it an optional parameter,
    # this kind of parameter will default to a value, but if a parameter is passed in it's place the value is overridden.
    # You have the option to not have it be a parameter of the function at all. If you want it as a parameter thats easy,
    # modify the constant LAMBDA and pass that in when you use it.
    #_lambda=2
    for i in range(m):
        f_wb_i = X_train[i]*w+ b                                 
        cost+= (f_wb_i - Y_train[i])**2                                        
    cost = np.sum(cost) / (2 * m)                                           
    reg_cost=0
    reg_cost+= (w**2)                                        
    reg_cost = (_lambda/(2*m)) * reg_cost                             
    
    total_cost = cost + reg_cost
    return (np.sum(total_cost))
    
def compute_gradient(X_train, Y_train, w, b, _lambda):
    m,n = X_train.shape           #(number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.
    err=0
    for i in range(m):                             
        f_wb_i = X_train[i]*w + b
        err+= dj_dw+ f_wb_i - Y_train[i]                                     
        dj_dw+= (f_wb_i-Y_train[i])* X_train[i]         
        dj_db= err                       
    dj_dw = (dj_dw)+_lambda*w/ m                                
    dj_db = dj_db / m 

    return dj_db, dj_dw
    
    
def gradient_descent(X_train, Y_train, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
     # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db,dj_dw = gradient_function(X_train, Y_train, w, b, LAMBDA)   ##None

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               ##None
        b = b - alpha * dj_db               ##None
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(X_train, Y_train, w, b, LAMBDA))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        
    return w, b, J_history #return final w,b and J history for graphing

#This is where the execution of your code starts.

# Read in excel files
df = pd.read_excel("CoviddataML1.xlsx", usecols=['date']) # Import columns by name
df_1 = pd.read_excel("CoviddataML1.xlsx", usecols=['avg.deaths'])

# Convert pandas dataframes to numpy arrays.
X_train = df.to_numpy()
Y_train = df_1.to_numpy()

# Another option is to define a constant value (which is good practice to make all caps) if the value will be used repeatedly
# Throughout your code.
LAMBDA = 2

# Print out the numpy arrays to see their values.
print (X_train)
print (Y_train)
print (Y_train.shape)

# Slice the X_train array
date=X_train[:, 0]

# Build the scatterplot
plt.scatter(X_train, Y_train)
x= np.arange(1, 31)
y= 0.318*x+6.41
plt.plot(x, y)

# Show the scatterplot, this piece is necessary outside of a Jupyter
# notebook. This is a key difference in python scripts vs Ipython (the command
# shell that Jupyter uses).
plt.show()

# Set up params to compute cost.
initial_w= np.array([0.5])
initial_b= 7.5

# You can choose to use this for the optional parameter
compute_cost(X_train, Y_train, initial_w, initial_b)

# For example, if LAMBDA's value was 5 at the time of the function call,
# then 5 would override (not overwrite) the optional value of 2 set within
# the function definition.
# If you were to try to use compute_cost again later, without
# overriding the optional parameter, the function would default to the
# function's definition, which is 2.
#compute_cost(X_train, Y_train, initial_w, initial_b, LAMBDA)

# Set up params to compute gradient.
w_init = np.array([ 0.1])
w_init = w_init.T
print (w_init)
b_init=7
tmp_dj_db, tmp_dj_dw = compute_gradient(X_train, Y_train, w_init, b_init, LAMBDA)
print(f'dj_db at initial w,b: {tmp_dj_db}')
print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')

# initialize parameters
initial_w = np.array([0.1])
initial_b = np.array([6])

# some gradient descent settings
iterations = 100000
alpha = .00000003

# run gradient descent 
w_final, b_final, J_hist = gradient_descent(X_train, Y_train, initial_w, initial_b,
                                                    compute_cost, compute_gradient, 
                                                    alpha, iterations)

print(f"b,w found by gradient descent: {b_final},{w_final} ")




