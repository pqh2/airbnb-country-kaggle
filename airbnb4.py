import csv
import gzip
###
import numpy as np
###
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.utils import np_utils
from sklearn import metrics
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler
import pandas
import numpy
import sklearn
from sknn.mlp import *
from sklearn.feature_extraction import DictVectorizer
from dateutil.parser import parse

def get_fields(data, fields):
  extracted = []
  for row in data:
    extract = []
    for field, f in sorted(fields.items()):
      info = f(row[field])
      if type(info) == list:
        extract.extend(info)
      else:
        extract.append(info)
    extracted.append(np.array(extract, dtype=np.float32))
  return extracted

def preprocess_data(X, scaler=None):
  if not scaler:
    scaler = StandardScaler()
    scaler.fit(X)
  X = scaler.transform(X)
  return X, scaler


def shuffle(X, y, seed=1337):
  np.random.seed(seed)
  shuffle = np.arange(y.shape[0])
  np.random.shuffle(shuffle)
  X = X[shuffle]
  y = y[shuffle]
  return X, y


raw_train = pandas.read_csv("/Users/patrickhess/Downloads/train_users_2.csv")

print len(raw_train)
print('Creating training data...')
X = []
Y = []
for it, row in raw_train.iterrows():
    data_point = {
                  'gender': 'unknown' if (str(row['gender']) == 'nan') else row['gender'],
                  'age': 'unknown' if (str(row['age']) == 'nan') else row['age'],
                  'signup_method': 'unknown' if (str(row['signup_method']) == 'nan') else row['signup_method'],
                  'signup_flow': 'unknown' if (str(row['signup_flow']) == 'nan') else row['signup_flow'],
                  'language': 'unknown' if (str(row['language']) == 'nan') else row['language'],
                  'affiliate_channel': 'unknown'  if (str(row['affiliate_channel']) == 'nan') else row['affiliate_channel'],
                  'affiliate_provider': 'unknown'  if (str(row['affiliate_provider']) == 'nan') else row['affiliate_provider'],
                  'first_affiliate_tracked': 'unknown' if (str(row['first_affiliate_tracked']) == 'nan') else row['first_affiliate_tracked'],
                  'signup_app': 'unknown' if (str(row['signup_app']) == 'nan') else row['signup_app'],
                  'month_first_booking': 'unknown' if (str(row['date_first_booking']) == 'nan') else str(parse(row['date_first_booking']).month) 
                  }
    X.append(data_point)
    Y.append({'category': row['country_destination']})

dctX = DictVectorizer(sparse=False)
X = dctX.fit_transform(X)
print('Creating training labels...')
dctY = DictVectorizer(sparse=False)
Y = dctY.fit_transform(Y)

X, Y = shuffle(X, Y)
#X, scaler = preprocess_data(X)


input_dim = X.shape[1]
output_dim = Y.shape[1]
print('Input dimensions: {}'.format(input_dim))


def build_model(input_dim, output_dim, hn=64, dp=0.5, layers=1):
    model = Sequential()
    model.add(Dense(input_dim=input_dim, output_dim=hn, init='glorot_uniform'))
    model.add(PReLU(input_shape=(hn,)))
    model.add(Dropout(dp))
    for i in range(layers):
        model.add(Dense(input_dim=hn, output_dim=hn, init='glorot_uniform'))
        model.add(PReLU(input_shape=(hn,)))
        model.add(BatchNormalization(input_shape=(hn,)))
        model.add(Dropout(dp))
        model.add(Dense(input_dim=hn, output_dim=output_dim, init='glorot_uniform'))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

EPOCHS = 4
BATCHES = 128
HN = 64
RUN_FOLDS = False
nb_folds = 4
kfolds = KFold(len(y), nb_folds)
av_ll = 0.
f = 0

print("Generating submission...")

model = build_model(input_dim, output_dim, HN)
model.fit(X, Y, nb_epoch=EPOCHS, batch_size=BATCHES, verbose=0) 
