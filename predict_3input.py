from pandas import read_csv
from sklearn import preprocessing
from keras.models import load_model
import numpy as np

dataset = read_csv('Data/cv.csv' )
values=dataset.values

reserve=values[:,-1]
npstd=np.std(reserve)
npmean=np.mean(reserve)

values=values.reshape(1,30)
test_X=preprocessing.scale(values)
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

model = load_model('pandas_result.h5')
prediction_data = model.predict(test_X, batch_size=1)
prediction_data=prediction_data*npstd+npmean
print(prediction_data)