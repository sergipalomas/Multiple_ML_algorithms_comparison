from sklearn.ensemble import RandomForestRegressor
from utils import *
import numpy as np

def randomForest(X_train, X_test, y_train, y_test):
    print("\n\n\n######################################################\n"
          "#############   Running Random Forest   ##############\n"
          "######################################################")

    fig, ax = plt.subplots(2, 1, figsize=(8, 16), sharey=True)
    for i, n_estimators in enumerate([32, 64]):
        print("Using %i trees" % n_estimators)
        rf_regressor = RandomForestRegressor(n_estimators=n_estimators, random_state=11, bootstrap=True, criterion='mse')
        rf_regressor.fit(X_train, y_train.values.ravel())
        pred = rf_regressor.predict(X_test)
        y_pred = np.array([pred]).T
        ax[i].plot(y_test.values, '-', color='r', label='True data')
        ax[i].plot(y_pred, '-', color='b', label="predicted data")
        R2 = R2_nonlinear(y_test.values, y_pred)
        rmse = RMSE(y_test.values, y_pred)
        mae = MAE(y_test.values, y_pred)
        ax[i].set_title("n_estimators = %i, R\u00b2 = %.3f, RMSE = %.3f, MAE = %.3f" % (n_estimators, R2, rmse, mae))
        general_plot(y_test.values, y_pred, "Random Forest with " + str(n_estimators) + " estimators")
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.tight_layout()
    fig.suptitle("RF with different number of trees without shuffle")
    plt.show()

