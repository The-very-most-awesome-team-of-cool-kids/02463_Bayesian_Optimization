import pickle
import GPyOpt
import matplotlib.pyplot

with open("opt_params.pkl", "rb") as f:
    opt = pickle.load(f)


GPyOpt.plotting.plots_bo.plot_convergence(opt.X, opt.Y)
plt.show()