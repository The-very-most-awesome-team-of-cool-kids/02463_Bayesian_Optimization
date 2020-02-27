import pickle
import GPyOpt
import matplotlib.pyplot as plt

with open("neural_opt/opt_params_best.pkl", "rb") as f:
    opt = pickle.load(f)

# opt.plot_convergence("test.png")
# plt.show()
GPyOpt.plotting.plots_bo.plot_convergence(opt.X, opt.Y*-1, filename = "convergence.png")
plt.ylabel("Accuracy [%]")
plt.show()

evaluations = opt.get_evaluations()
opt.save_evaluations("evaluations")

learning_rates = evaluations[0][:, 0]
optimizers = evaluations[0][:, 1]
accuracies = list(evaluations[1])
print(accuracies)
# fig = plt.figure()

# plt.subplot(2,1, 1)
# plt.plot(evaluations[0][0], evaluations[1])