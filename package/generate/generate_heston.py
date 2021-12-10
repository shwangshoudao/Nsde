import numpy as np
from scipy.optimize import least_squares


def heston(S0,V0,r,x,input):
    """generate heston function

    Args:
        S0 (float): initial asset price
        V0 (float): initial volatility
        r (float): risk free rate
        x (4*1 numpy array): 1. kappa 2. theta 3. lambda  4. rho
        input (n*2 numpy array): the first column is maturity date and the second column is the strike price

    Returns:
        option price: option price of the given input
    """
    kappa = x[0]
    theta = x[1]
    lambd = x[2]
    rho = x[3]
    T = input[0]
    K = input[1]
    I=complex(0,1)
    P, umax, N = 0, 1000, 10000
    du=umax/N
    
    aa= theta*kappa*T/lambd**2
    bb= -2*theta*kappa/lambd**2
    for i in range (1,N):
        u2=i*du
        u1=complex(u2,-1)
        a1=rho*lambd*u1*I
        a2=rho*lambd*u2*I
        d1=np.sqrt((a1-kappa)**2+lambd**2*(u1*I+u1**2))
        d2=np.sqrt((a2-kappa)**2+lambd**2*(u2*I+u2**2))
        g1=(kappa-a1-d1)/(kappa-a1+d1)
        g2=(kappa-a2-d2)/(kappa-a2+d2)
        b1=np.exp(u1*I*(np.log(S0/K)+r*T))*( (1-g1*np.exp(-d1*T))/(1-g1) )**bb
        b2=np.exp(u2*I*(np.log(S0/K)+r*T))*( (1-g2*np.exp(-d2*T))/(1-g2) )**bb
        phi1=b1*np.exp(aa*(kappa-a1-d1)\
        +V0*(kappa-a1-d1)*(1-np.exp(-d1*T))/(1-g1*np.exp(-d1*T))/lambd**2)
        phi2=b2*np.exp(aa*(kappa-a2-d2)\
        +V0*(kappa-a2-d2)*(1-np.exp(-d2*T))/(1-g2*np.exp(-d2*T))/lambd**2)
        P+= ((phi1-phi2)/(u2*I))*du
    return K*np.real((S0/K-np.exp(-r*T))/2+P/np.pi)

if __name__ == "__main__":
    ## setting
    S0 = 95
    V0 = 0.1
    r = 0.03
    kappa = 1.5768
    theta = 0.0398
    lambd = 0.575
    rho = -0.5711
    
    ##generate data
    input = np.array([[1,1,1,1,2,2,2,2,3,3,3,3],[90,95,100,110,90,95,100,110,90,95,100,110]])
    
    y_train = heston(S0,V0,r,[kappa,theta,lambd,rho],input)
    
    ## calibration
    def fun(x,input,y):
        return heston(S0,V0,r,x,input) - y
    x0 = np.array([1.3, 0.05, 0.6, 1])
    res_lsq = least_squares(fun, x0, args=(input, y_train))
    
    print(res_lsq.x)