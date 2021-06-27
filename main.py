import pandas as pd
from explore_data import understand_data
from sklearn.model_selection import train_test_split
from utils import *
from MLR import simple_MLR, MLR
from KNN import knn
from RF import randomForest
from SVR import svr
from NN import nn

def menu():
    print("Select the ML algorithm you want to use:\n\n"
          "[0] Explore the data\n"
          "[1] Multiple linear regression (MLR)\n"
          "[2] K-nearest neighbor (KNN)\n"
          "[3] Random Forest (RF)\n"
          "[4] Kernel regression\n"
          "[5] Gaussian Process (GP)\n"
          "[6] Support Vector Regression (SVR)\n"
          "[7] Neural Network (NN)\n"
          "\n[-1] Quit\n"
          )


if __name__ == '__main__':
    # Read data
    path = './data/data_TOML_proj2.csv'
    data = pd.read_csv(path, sep=';', parse_dates=True)

    # Split train and test data
    X = data[['Sensor_O3', 'Temp', 'RelHum']]
    y = data[['RefSt']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=11, shuffle=True)

    print("Welcome!\n")
    print("Do you want to normalize all the samples (does not apply for 'observe the data')?")
    norm = str(input("y/n: "))
    if (norm == 'y'):
        for ele in X_train, X_test, y_train, y_test:
            normalize_df(ele)

    menu()
    option = int(input("Enter an option: "))
    while option != -1:

        ## Exploring the data
        if option == 0:
            understand_data(data)

        ## MLR
        elif option == 1:
            #simple_MLR(data)
            MLR(X_train, X_test, y_train, y_test)

        ## K-nearest neighbor (KNN)
        elif option == 2:
            knn(X_train, X_test, y_train, y_test)

        ## Random Forest (RF)
        elif option == 3:
            randomForest(X_train, X_test, y_train, y_test)

        ## Kernel Regression
        elif option == 4:
            pass

        ## Gaussian Prosses (GP)
        elif option == 5:
            pass

        ## Suport Vector Regression (SVR)
        elif option == 6:
            svr(X_train, X_test, y_train, y_test)

        ## Neural Network (NN)
        elif option == 7:
            nn(X_train, X_test, y_train, y_test)

        else:
            print("Invalid option!")

        print("\n\n"
              "###########################\n")
        menu()
        option = int(input("Enter an option: "))
