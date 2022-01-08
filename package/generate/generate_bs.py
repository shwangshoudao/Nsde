import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm

def bs(S0,V0,r,sigma,input):
    """generate heston function

    Args:
        S0 (float): initial asset price
        V0 (float): initial volatility
        r (float): risk free rate
        sigma : volatility
        input (n*2 numpy array): the first column is maturity date and the second column is the strike price

    Returns:
        option price: option price of the given input
    """
    T = input[:,0]
    K = input[:,1]
    
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = (np.log(S0/K) + (r - 0.5*sigma**2)*T)/(sigma * np.sqrt(T))
    
    p = (S0*norm.cdf(d1, 0.0, 1.0) - K*np.exp(-r*T)*norm.cdf(d2, 0.0, 1.0))
    return p
    




if __name__ == "__main__":
    ## setting
    S0 = 95
    V0 = 0.1
    r = 0.03
    sigma = 0.03
    
    ##generate data
    input = np.array([[1,1,1,1,2,2,2,2,3,3,3,3],[90,95,100,110,90,95,100,110,90,95,100,110]])
    input = input.T
    y_train = bs(S0,V0,r,sigma,input)
    
    ## calibration
    def fun(x,input,y):
        return bs(S0,V0,r,x,input) - y
    x0 = 0.06
    res_lsq = least_squares(fun, x0, args=(input, y_train))
    
    print(res_lsq.x)