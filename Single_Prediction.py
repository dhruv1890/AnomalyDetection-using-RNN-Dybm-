if __name__ == "__main__":
    import numpy as np
import matplotlib.pyplot as plt
import pandas
import argparse
import sys
from sklearn.preprocessing import MinMaxScaler
import os

from pydybm.time_series.dybm import LinearDyBM
from pydybm.time_series.rnn_gaussian_dybm import RNNGaussianDyBM, GaussianDyBM
from pydybm.base.sgd import RMSProp
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.metrics import mean_squared_error
dataframe = pandas.read_csv("/home/dhruv/Downloads/BH11D_CF_Algo (labelled) - BH11D_CF_Algo (labelled).csv",
                        usecols=[1], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')
print(dataset)
np.random.seed(2)
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
RNN_dim = 10
input_dim = 1
max_epochs = 5
saveResults = False
SGD = RMSProp
decay = [0.5]

print "Number of Observations/datapoints in each dimension", len(dataset)
# split into train and test sets (default 75% train, 25% test)
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    return np.array(dataX), np.array(dataY)
look_back = 1

trainPercentage = 0.75
train_size = int(len(dataset) * trainPercentage)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
trainX = np.reshape(trainX, (trainX.shape[0], input_dim))
testX = np.reshape(testX, (testX.shape[0], input_dim))
for delay in [3]:
    dybm = RNNGaussianDyBM(input_dim, input_dim, RNN_dim, 0.3,
                                   0.1, delay, decay_rates=decay, leak=1.0,
                                   SGD=SGD())

dybm.set_learning_rate(0.001)
dybm.init_state
dybm.init_state()
result = dybm.learn(trainX, get_result=True)
result2 = dybm.learn(testX, get_result=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--x',type=float,default=1.0,help ='what is next value of CH4 concentration')
    args = parser.parse_args()
    sys.stdout.write(str(singlePrediction(args)))
def singlePrediction(args):
    singleX = list(args.x)
    print(singleX)
    singleX = np.array(singleX)
    #singleX = np.reshape(singleX, (singleX.shape[0], input_dim))
    a = dybm.learn(singleX, get_result=True)
    args.x = a
    return args.x
main()