from utils import R2_nonlinear, RMSE, general_plot
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


def plot_results(x1, x2, x3, n, hist_loss, test_O3, test_T, test_rh, test_true, method):
    # Out-of-sample
    pred_res = x1 * test_O3 + x2 * test_T + x3 * test_rh + n
    # plt.plot(test_true, '-', label='True data')
    # plt.plot(pred_res, '-', color='r', label="predicted data")
    # plt.title('Test vs predicted values using ' + method + ' gradient method')
    # plt.show()
    title = 'True vs predicted values using ' + method + ' gradient method'
    general_plot(test_true.values, pred_res.values, title)

    plt.plot(hist_loss)
    plt.title(method + ' gradient descent loss function per epoch')
    plt.xlabel('epochs')
    plt.ylabel('loss function')
    plt.show()

def gd_batch(train_O3, train_T, train_rh, train_true, N):
    # Variables y = x1*Sensor_O3 + x2*Temp + x3*RelHum + n
    x1 = 0
    x2 = 0
    x3 = 0
    n = 0
    L_rate = 0.001
    epochs = 600

    # Loss function
    # E = 1/N * sum(true_res - pred_res)**2
    # Gradient Descent loss function
    hist_loss = list()
    for i in range(epochs):
        pred_res = x1 * train_O3 + x2 * train_T + x3
        d_x1 = (-2 / N) * (train_O3 * (train_true - pred_res)).sum()
        d_x2 = (-2 / N) * (train_T * (train_true - pred_res)).sum()
        d_x3 = (-2 / N) * (train_rh * (train_true - pred_res)).sum()
        d_n = (-2 / N) * (train_true - pred_res).sum()

        x1 = x1 - L_rate * d_x1
        x2 = x2 - L_rate * d_x2
        x3 = x3 - L_rate * d_x3
        n = n - L_rate * d_n

        hist_loss.append(1/N * sum((train_true - pred_res)**2))
    return x1, x2, x3, n, hist_loss


def gd_stochastic(train_O3, train_T, train_rh, train_true, N):
    # Variables y = x1*Sensor_O3 + x2*Temp + x3*RelHum + n
    x1 = 0
    x2 = 0
    x3 = 0
    n = 0
    L_rate = 0.001
    samples = 500

    # Loss function
    # E = 1/N * sum(true_res - pred_res)**2
    # Gradient Descent loss function
    hist_loss = list()
    epochs = min(N, samples)
    for i in range(epochs):
        pred_res = x1 * train_O3.iloc[i] + x2 * train_T.iloc[i] + x3 * train_rh.iloc[i] + n
        d_x1 = (-2/1) * train_O3.iloc[i]*(train_true.iloc[i] - pred_res)
        d_x2 = (-2/1) * train_T.iloc[i]*(train_true.iloc[i] - pred_res)
        d_x3 = (-2/1) * train_rh.iloc[i]*(train_true.iloc[i] - pred_res)
        d_n = (-2/1) * train_true.iloc[i] - pred_res

        x1 = x1 - L_rate * d_x1
        x2 = x2 - L_rate * d_x2
        x3 = x3 - L_rate * d_x3
        n = n - L_rate * d_n

        hist_loss.append(1/N * (train_true.iloc[i] - pred_res)**2)
    return x1, x2, x3, n, hist_loss

def gd_minibatch(train_O3, train_T, train_rh, train_true, N):
    # Variables y = x1*Sensor_O3 + x2*Temp + x3*RelHum + n
    x1 = 0
    x2 = 0
    x3 = 0
    n = 0
    L_rate = 0.1
    batch_size = 25
    epochs = N // batch_size

    # Loss function
    # E = 1/N * sum(true_res - pred_res)**2
    # Gradient Descent loss function
    hist_loss = list()
    for i in range(epochs):
        idx = slice(i*batch_size, (i+1)*batch_size)
        pred_res = x1 * train_O3[idx] + x2 * train_T[idx] + x3 * train_rh[idx] + n
        d_x1 = (-2/batch_size) * (train_O3[idx]*(train_true[idx] - pred_res)).sum()
        d_x2 = (-2/batch_size) * (train_T[idx]*(train_true[idx] - pred_res)).sum()
        d_x3 = (-2/batch_size) * (train_rh[idx]*(train_true[idx] - pred_res)).sum()
        d_n = (-2/batch_size) * (train_true[idx] - pred_res).sum()

        x1 = x1 - L_rate * d_x1
        x2 = x2 - L_rate * d_x2
        x3 = x3 - L_rate * d_x3
        n = n - L_rate * d_n

        hist_loss.append(1/N * ((train_true[idx] - pred_res)**2).sum())
    return x1, x2, x3, n, hist_loss


# TODO: Show that RelHum is not needed (P>|T|) = 0.955.
def simple_MLR(data):
    train_df = data.sample(frac=.66, random_state=11)
    test_df = data[~data.isin(train_df)].dropna()
    true_res = test_df['RefSt'].values
    lm = smf.ols(formula='RefSt ~ Sensor_O3 + Temp + RelHum', data=train_df).fit()
    print(lm.params)
    print(lm.summary()) # Check this: https://medium.com/swlh/interpreting-linear-regression-through-statsmodels-summary-4796d359035a
    pred_res = lm.predict(test_df).values
    plt.plot(true_res, '-', label='True data')
    plt.plot(pred_res, '-', color='r', label="predicted data")
    plt.show()
    # Out-of-sample R square
    general_plot(true_res, pred_res, "MLR with normal equations")


def MLR(X_train, X_test, y_train, y_test):
    print("\n\n\n######################################################\n"
          "###################   Running MLR  ###################\n"
          "######################################################")
    train_O3 = X_train['Sensor_O3']
    train_T = X_train['Temp']
    train_rh = X_train['RelHum']
    train_true = y_train['RefSt']
    test_O3 = X_test['Sensor_O3']
    test_T = X_test['Temp']
    test_rh = X_test['RelHum']
    test_true = y_test['RefSt']
    N_train = y_train.shape[0]

    # Run Gradient Descent default
    method = 'Batch'
    print("MLR with " + method)
    x1, x2, x3, n, hist_loss = gd_batch(train_O3, train_T, train_rh, train_true, N_train)
    plot_results(x1, x2, x3, n, hist_loss, test_O3, test_T, test_rh, test_true, method)
    print("#########################")

    method = 'Stochastic'
    print("MLR with " + method)
    x1, x2, x3, n, hist_loss = gd_stochastic(train_O3, train_T, train_rh, train_true, N_train)
    plot_results(x1, x2, x3, n, hist_loss, test_O3, test_T, test_rh, test_true, method)
    print("#########################")

    method = 'Mini-batch'
    print("MLR with " + method)
    x1, x2, x3, n, hist_loss = gd_minibatch(train_O3, train_T, train_rh, train_true, N_train)
    plot_results(x1, x2, x3, n, hist_loss, test_O3, test_T, test_rh, test_true, method)




