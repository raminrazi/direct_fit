import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from fit_direct import fit_direct
from simulate import simulate

# Set ODE_str to 'fitzhugh' or 'lotka_volterra' or 'rossler'
ODE_str = 'fitzhugh'
#ODE_str = 'rossler'
#ODE_str = 'lotka_volterra'
# Look at the file simulate.py to see how to create noisy observations and clean states.
if(ODE_str == 'fitzhugh'):
    X,Y,dt = simulate(ODE_str, x0 = [-1,1], true_param = [.2,.5,3], end_t = 20, dt = .05, noise_var = .5)
    init_param = np.array([2, 2, 5])  # initialization of the parameters
elif(ODE_str == 'lotka_volterra'):
    X,Y,dt = simulate(ODE_str, x0 = [5,3], true_param = [2,1,4,1], end_t = 2, dt = .01, noise_var = .5)
    init_param = np.array([2, 2, 5, 2])  # initialization of the parameters
elif(ODE_str == 'rossler'):
    X,Y,dt = simulate(ODE_str, x0 = [1.13293, -1.74953, 0.02207], true_param = [.2,.2,3], end_t = 20, dt = .1, noise_var = 1)
    init_param = np.array([2, 2, 5])  # initialization of the parameters


# learn the parameters, estimated states, and predicted states
params,est_X, pred_X = fit_direct(Y,dt,init_param,ODE_str,max_iters=10000)

error = np.sum((pred_X - X)**2)
print('prediction error = ', error)
print('params', params)

# draw the results: visualize the first dimension
xax = np.arange(X.shape[0]).reshape(-1,1)*dt
plt.scatter(xax, X[:,0],s=4,label = 'clean states',color='g')
plt.scatter(xax, Y[:,0],s=4, label = 'noisy observations',color='m')
plt.scatter(xax, pred_X[:,0],s=4, label = 'predicted states', color='b')
plt.axis('scaled')
plt.legend(fancybox=True, framealpha=0.01)
plt.show()

