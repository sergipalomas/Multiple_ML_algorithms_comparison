from utils import normalize_pd_column
import matplotlib.pyplot as plt
import numpy as np

def understand_data(data):
    print(data.columns)
    ## Plot Sensor_O3 and RefSt evolution
    ax = data.plot('date', 'Sensor_O3', ylabel="KΩ", title="MOX sensor measurments (Sensor_O3) and \n"
                                                           "Reference Station O3 (RefSt) evolution")
    ax2 = data.plot('date', 'RefSt', secondary_y=True, ax=ax)
    ax2.set_ylabel('µgr/m^3')
    ax.set_xlabel('date')
    plt.xticks([], [])
    plt.show()

    features = ('RefSt', 'Sensor_O3', 'Temp', 'RelHum')
    fig1, ax1 = plt.subplots(len(features), len(features)-1, figsize=(12, 9))
    fig2, ax2 = plt.subplots(len(features), len(features) - 1, figsize=(12, 9))
    fig3, ax3 = plt.subplots(2, 2, figsize=(8, 5))
    for i, feature_x in enumerate(features):
        data.hist(column=feature_x, ax=ax3[i//2, i % 2])
        other_features = [x for x in features if not x in feature_x]
        for j, feature_y in enumerate(other_features):
            # Prior to normalization
            x = data[feature_x]
            y = data[feature_y]
            ax1[i, j].plot(x, y, 'o')
            ax1[i, j].set_xlabel(feature_x)
            ax1[i, j].set_ylabel(feature_y)
            xpoints = ypoints = ax1[i, j].get_xlim()
            ax1[i, j].plot(xpoints, ypoints, linestyle='--', color='k', lw=1, scalex=False, scaley=False)
            ax1[i, j].set_title('Pearson corr. = %.3f' % (np.corrcoef(x, y)[0, 1]))

            # After normalization
            x_norm = normalize_pd_column(x)
            y_norm = normalize_pd_column(y)
            ax2[i, j].plot(x_norm, y_norm, 'o')
            ax2[i, j].set_xlabel(feature_x + ' norm')
            ax2[i, j].set_ylabel(feature_y + ' norm')
            xpoints = ypoints = ax2[i, j].get_xlim()
            ax2[i, j].plot(xpoints, ypoints, linestyle='--', color='k', lw=1, scalex=False, scaley=False)
            ax2[i, j].set_title('Pearson corr. = %.3f' % (np.corrcoef(x, y)[0, 1]))

    fig1.suptitle("Scatter plots to establish relations between features")
    fig2.suptitle("Scatter plots to establish relations between features after normalization")
    fig3.suptitle("Histogram for each feature")
    plt.show()

    features_df = data[['RefSt', 'Sensor_O3', 'Temp', 'RelHum']]
    corr = features_df.corr()
    corr.style.background_gradient(cmap='coolwarm', axis=None).set_precision(2)
