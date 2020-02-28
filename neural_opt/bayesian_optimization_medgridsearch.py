from objective_function import objective_function
import numpy as np
import GPyOpt
import random
import time
import pickle
import matplotlib.pyplot as plt
import torch

# set seed
seed = 4200
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

# learning rate
learning_rate = tuple(np.arange(0.0001,0.011 ,0.0001, dtype= np.float))
#optimizer (SGD, Adam)
optimizer_dict = {0: "SGD", 1: "ADAM"}
optimizer = (0, 1)

# activation function (relu, sigmoid)
act_func_dict = {0: "ReLU", 1: "Sigmoid"}
act_func = (0,1)


# define the dictionary for GPyOpt

domain = [{'name': 'learning_rate', 'type': 'discrete', 'domain': learning_rate},
            {'name': 'optimizer', 'type': 'categorical', 'domain': optimizer},
            {'name': 'act_func', 'type': 'categorical', 'domain': act_func}]


#optimization
opt = GPyOpt.methods.BayesianOptimization(f = objective_function,   # function to optimize
                                              domain = domain,         # box-constrains of the problem
                                              acquisition_type = 'EI' ,      # Select acquisition function MPI, EI, LCB
                                             )


opt.acquisition.exploration_weight = 1

t_opt = time.time()
opt.run_optimization(max_iter = 20) 
print("-"*30)
print("Optimization finished!")
print(f"Time used for optimization: {time.time()-t_opt} seconds")

x_best = opt.X[np.argmin(opt.Y)]
print(f"Best accuracy was obtained at {opt.fx_opt*-1} %")
print("The best parameters obtained: learning rate=" + str(x_best[0]) + ", optimizer=" + str(optimizer_dict[x_best[1]]) + ", activation function=" + str(act_func_dict[x_best[2]]))

# save 
with open("neural_opt/opt_params.pkl", "wb") as f:
    pickle.dump(opt, f)

with open("neural_opt/opt_params_best.pkl", "rb") as f:
    best_opt = pickle.load(f)

if opt.fx_opt*-1 > best_opt.fx_opt*-1:
    with open("neural_opt/opt_params_best.pkl", "wb") as f:
        pickle.dump(opt, f)


# plots
# GPyOpt.plotting.plots_bo.plot_acquisition(opt)
GPyOpt.plotting.plots_bo.plot_convergence(opt.X, opt.Y*-1)
plt.show()



