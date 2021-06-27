import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from utils import R2_nonlinear, RMSE, MAE


# @TODO: Test with and without shuffle
def knn(X_train, X_test, y_train, y_test):
    print("\n\n\n######################################################\n"
          "############   K-nearest neighbor (KNN)   ############\n"
          "######################################################")
    # See the data
    plt.plot(y_train.values)
    plt.show()

    # Loop for different n_neighbors and weights
    fig, ax = plt.subplots(6, 2, figsize=(15, 15), sharey=True)
    for i, n_neighbors in enumerate([1, 8, 16, 32, 64, 128]):
        for j, weight in enumerate(['uniform', 'distance']):
            knn = KNeighborsRegressor(n_neighbors, weights=weight)
            pred_res = knn.fit(X_train, y_train).predict(X_test)
            R2 = R2_nonlinear(y_test.values, pred_res)
            rmse = RMSE(y_test.values, pred_res)
            mae = MAE(y_test.values, pred_res)
            ax[i, j].plot(y_test.values, '-', color='r', label='True data')
            ax[i, j].plot(pred_res, '-', color='b', label="predicted data")
            ax[i, j].set_title("k = %i, wgt. = %s, R\u00b2 = %.3f, RMSE = %.3f, MAE = %.3f" % (n_neighbors, weight, R2, rmse, mae))

    handles, labels = ax[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.tight_layout()
    fig.suptitle("KNN regressor with shuffle and without normalization")
    plt.show()