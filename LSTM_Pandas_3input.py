from pandas import read_csv
from datetime import datetime
from pandas import read_csv, DataFrame, concat
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers import Dense, LSTM

from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np


dataset = read_csv('Data/12.13.1.csv', index_col=0)
dataset.index.name = 'date'
dataset.columns = ['current', 'squeeze_speed', 'diameter']

# print(dataset.head(5))


# values = dataset.values
# groups = [0, 1, 2, 3]
# i = 1
# pyplot.figure()
# for group in groups:
#     pyplot.subplot(len(groups), 1, i)
#     pyplot.plot(values[:, group])
#     pyplot.title(dataset.columns[group], y=0.5, loc='right')
#     i += 1
# pyplot.show()

prediction_basesize = 10
cell_size = 100
train_apochs = 10
batch_size = 72
is_training = 0


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


values = dataset.values
values = values.astype('float32')
reframed = series_to_supervised(values, prediction_basesize, 1)
reframed.drop(reframed.columns[range(3 * (prediction_basesize + 1) - 3, 3 * (prediction_basesize + 1) - 1)], axis=1,
              inplace=True)


# split into train and test sets
values = reframed.values
n_train_examples = 50000
train = values[:n_train_examples, :]
test = values[n_train_examples:, :]

# scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
# train=scaler.fit_transform(train)
# test=scaler.fit_transform(test)


# split into input and outputs
train_X_un, train_y_un = train[:, :-1], train[:, -1]
test_X_un, test_y_un = test[:, :-1], test[:, -1]

train_X=preprocessing.scale(train_X_un)
train_y=preprocessing.scale(train_y_un)
test_X=preprocessing.scale(test_X_un)
test_y=preprocessing.scale(test_y_un)


print(train_X)
print(train_y)

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(cell_size, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam', )


# fit network


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.loss = []

    def on_epoch_end(self, batch, logs={}):
        self.loss.append(logs.get('loss'))


history = LossHistory()

early_stopping = EarlyStopping(monitor='loss', patience=3, verbose=0, mode='auto')

if is_training:
    print('Training ------------')
    model.fit(train_X, train_y, epochs=train_apochs, batch_size=batch_size, verbose=1,
              callbacks=[history, early_stopping], shuffle=False)
    # json_string = model.to_json()
    # open('/var/my_model_architecture.json', 'w').write(json_string)
    # model.save_weights('/var/my_model_weights.h5')
    model.save('pandas_result.h5')

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")

else:
    # model = model_from_json(open('my_model_architecture.json').read())
    # model.load_weights('my_model_weights.h5')
    model = load_model('pandas_result.h5')
    print('\nTesting ------------')
    loss = model.evaluate(test_X, test_y, batch_size=batch_size, )
    prediction_data = model.predict(test_X, batch_size=batch_size)

    print('\ntest loss: ', loss)

    # original_data = (test)
    # prediction_data = np.hstack((test_X.reshape(test_X.shape[0],test_X.shape[2]),prediction_data))
    #
    # print(original_data.shape)
    # print(prediction_data.shape)
    # original_data=scaler.inverse_transform(original_data)
    # prediction_data=scaler.inverse_transform(prediction_data)
    #
    # original_data = original_data[:,-1]
    # prediction_data = prediction_data[:,-1]

    original_data=test_y_un
    prediction_data=prediction_data*np.std(test_y_un)+np.mean(test_y_un)


    def accuracy(confidence, test_data, prediction_data):
        count = 0
        for i in range(len(prediction_data)):
            if abs(prediction_data[i] - test_data[i]) <= confidence:
                count = count + 1
        print('%.2f%%' % (count * 100 / len(prediction_data)))


    stat_original = original_data.flatten()
    stat_prediction = prediction_data.flatten()
    accuracy(0.1, stat_original, stat_prediction)
    accuracy(0.01, stat_original, stat_prediction)
    accuracy(0.001, stat_original, stat_prediction)

    plt.figure()
    plt.plot(list(range(len(original_data))), original_data.flatten(), color='b',
             label='Original data')
    plt.plot(list(range(len(prediction_data))), prediction_data.flatten(), color='r',
             label='Prediction data')
    plt.xlabel('Time (second)')
    plt.ylabel('Fiber diameter (mm)')
    plt.show()
