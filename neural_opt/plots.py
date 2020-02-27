import pickle
import GPyOpt
import matplotlib.pyplot as plt
import numpy as np

with open("neural_opt/opt_params.pkl", "rb") as f:
    opt = pickle.load(f)

# opt.plot_convergence("test.png")
# plt.show()
GPyOpt.plotting.plots_bo.plot_convergence(opt.X, opt.Y*-1, filename = "convergence.png")
plt.ylabel("Accuracy [%]")
plt.show()

# get evaluations
evaluations = opt.get_evaluations()
opt.save_evaluations("evaluations")

learning_rates = evaluations[0][:, 0]
optimizers = evaluations[0][:, 1]
act_funcs = evaluations[0][:, 2]
accuracies = np.array([-evaluations[1][i][0] for i in range(len(evaluations[1]))])

# plot
fig = plt.figure()

plt.subplot(2,2, 1)
plt.plot(learning_rates, accuracies)

plt.subplot(3,1,2)
plt.plot(optimiz)