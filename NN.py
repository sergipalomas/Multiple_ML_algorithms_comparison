from sklearn.neural_network import MLPRegressor
from utils import *
import matplotlib.pyplot as plt

def nn(X_train, X_test, y_train, y_test):
    print("\n\n\n######################################################\n"
          "###############   Nerual Network (NN)   ##############\n"
          "######################################################")
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    for nhl in [20]:
        print("Working with %i hidden layers" % nhl)
        fig, ax = plt.subplots(3, 3, figsize=(18, 10), sharey=True)
        for i, activation in enumerate(['logistic', 'tanh', 'relu']):
            for j, solver in enumerate(['lbfgs', 'sgd', 'adam']):

                print("Working for NN using activation %s and solver %s" % (activation, solver))

                # Train and test the model
                m1 = MLPRegressor(hidden_layer_sizes=nhl, max_iter=6000, batch_size=5, random_state=11,
                                  activation=activation, solver=solver, shuffle=True, verbose=False)
                m1.fit(X_train, y_train)
                y_pred = m1.predict(X_test)

                # Metrics
                R2 = R2_nonlinear(y_test, y_pred)
                rmse = RMSE(y_test, y_pred)
                mae = MAE(y_test, y_pred)

                ax[i, j].plot(y_test, '-', color='r', label='True data')
                ax[i, j].plot(y_pred, '-', color='b', label="predicted data")
                ax[i, j].set_title("act.= %s, sol.= %s, R\u00b2= %.3f, RMSE= %.3f, MAE= %.3f" % (activation, solver, R2, rmse, mae))

                # plt.plot(m1.loss_curve_)
                # plt.title("Loss curve")
                # plt.xlabel("epoch")
                # plt.ylabel('loss function')
                # plt.show()
                # general_plot(y_test, y_pred, "NN")

        handles, labels = ax[0, 1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        fig.tight_layout()
        fig.suptitle("NN Regressor with " + str(nhl) + " hidden layers")
        plt.show()
