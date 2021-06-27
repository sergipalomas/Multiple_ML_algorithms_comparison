import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.options.mode.chained_assignment = None

def normalize_pd_column(arr):
    mean = arr.mean()
    std = arr.std()
    return (arr - mean) / std

def normalize_df(df):
    for feature_name in df.columns:
        mean = df[feature_name].mean()
        std = df[feature_name].std()
        df[feature_name] = (df[feature_name] - mean) / std

# R^2 for nonlinear: https://stats.stackexchange.com/questions/334004/can-r2-be-greater-than-1
def R2_nonlinear(true_v, pred_v):
    mean_true_res = np.mean(true_v)
    SSe = sum((true_v - pred_v)**2)
    SSt = sum((true_v - mean_true_res)**2)
    R2 = 1 - SSe / SSt
    return R2

def RMSE(test_v, pred_v):
    N = len(test_v)
    return np.sqrt(1 / N * ((test_v - pred_v) ** 2).sum())

def MAE(test_v, pred_v):
    N = len(test_v)
    return (np.absolute(test_v - pred_v)).sum()/N

def general_plot(true_v, pred_v, title):
    plt.plot(true_v, '-', color='r', label='True data')
    plt.plot(pred_v, '-', color='b', label="predicted data")
    plt.legend()
    plt.title(title)
    plt.show()

    plt.scatter(true_v, pred_v)
    plt.xlabel('True values')
    plt.ylabel('Predicted values')
    xpoints = ypoints = (true_v.min(), true_v.max())
    plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=1, scalex=False, scaley=False)
    plt.title(title)
    plt.show()
    print("R\u00b2: %.3f" % R2_nonlinear(true_v, pred_v))
    print("RMSE = %.3f" % RMSE(true_v, pred_v))
    print("MAE = %.3f" % (MAE(true_v, pred_v)))