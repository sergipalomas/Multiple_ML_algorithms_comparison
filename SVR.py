import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from utils import *

def svr(X_train, X_test, y_train, y_test):
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    C = 15


    svr_rbf = SVR(kernel='rbf', C=C, gamma='auto',  epsilon=.1)
    svr_lin = SVR(kernel='linear', C=C, gamma='auto')
    svr_poly = SVR(kernel='poly', C=C, gamma='auto', degree=5, epsilon=.01, coef0=0.1)


    svrs = [svr_rbf, svr_lin, svr_poly]
    kernel_label = ['RBF', 'Linear', 'Polynomial']

    fig, ax = plt.subplots(3, 2, figsize=(12, 8), sharey=False, gridspec_kw={'width_ratios': [2.8, 1.2]})
    for i, svr in enumerate(svrs):
        svr_model = svr.fit(X_train, y_train)
        y_pred = svr_model.predict(X_test)
        R2 = R2_nonlinear(y_test, y_pred)
        rmse = RMSE(y_test, y_pred)
        mae = MAE(y_test, y_pred)
        #general_plot(y_test, y_pred, "SVR " + kernel_label[i])

        # True vs Pred plot
        ax[i, 0].plot(y_test, '-', color='r', label='True data')
        ax[i, 0].plot(y_pred, '-', color='b', label="predicted data")
        ax[i, 0].set_title("kernel = %s, R\u00b2 = %.3f, RMSE = %.3f, MAE = %.3f" % (kernel_label[i], R2, rmse, mae))

        # Scatter plot
        ax[i, 1].plot(y_test, y_pred, 'o', color='b')
        ax[i, 1].set_xlabel('True values')
        ax[i, 1].set_ylabel('Predicted values')
        xpoints = ypoints = (y_test.min(), y_test.max())
        ax[i, 1].plot(xpoints, ypoints, linestyle='--', color='k', lw=1, scalex=False, scaley=False)



    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left')
    fig.tight_layout()
    fig.suptitle("SVR")
    plt.show()
