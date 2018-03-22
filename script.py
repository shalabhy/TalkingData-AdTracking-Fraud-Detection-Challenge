
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


train_columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
#test_columns  = 
test_columns  = ['ip', 'app', 'device', 'os', 'channel', 'click_time']
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }

# Read the last lines because they are more impacting in training than the starting lines
path = '../input/'
test = pd.read_csv(path+"test.csv", usecols=test_columns, dtype=dtypes)
train = pd.read_csv(path+"train.csv", chunksize = 1000000 , usecols=train_columns, dtype=dtypes)
from sklearn.utils import shuffle
final_train = pd.DataFrame({'ip':[np.nan], 'app':[np.nan], 'device':[np.nan], 'os':[np.nan], 'channel':[np.nan], 'click_time':[np.nan], 'is_attributed':[np.nan]})

def undersampling(df):
    downloaded = df[df["is_attributed"] == 1]
    df = df[df["is_attributed"] == 0]
    sampled = df.sample(downloaded.shape[0]*2)
    undersampled = pd.concat([downloaded,sampled],ignore_index=True)
    return undersampled
    
for i in train:
    train_data = i
    smaller_data = undersampling(train_data)
    smaller_data = shuffle(smaller_data)
    del(train_data)
    final_train = pd.concat([final_train,smaller_data])
    
final_train = final_train.drop([0])

def timeFeatures(df):
    # Make some new features with click_time column
    df['datetime'] = pd.to_datetime(df['click_time'])
    df['day']      = df['datetime'].dt.dayofweek
    df['date']      = df["datetime"].dt.dayofyear
    #df["dteom"]    = df["datetime"].dt.daysinmonth - df["datetime"].dt.day
    df.drop(['click_time', 'datetime'], axis=1, inplace=True)
    return df

final_train = timeFeatures(final_train)

valid = final_train[1300000:]
final_train = final_train[:1300000]

y_train = final_train.is_attributed
y_valid = valid.is_attributed

final_train.drop("is_attributed",axis = 1, inplace = True)
valid.drop("is_attributed",axis = 1, inplace = True)

for col in final_train.columns:
    final_train[col] = final_train[col].astype('int')
    valid[col] = valid[col].astype('int')
    
y_train = y_train.astype('int')
y_valid = y_valid.astype('int')    
import lightgbm as lgb
d_train = lgb.Dataset(final_train, label = y_train)
d_valid = lgb.Dataset(valid, label = y_valid)

params = {
    'learning_rate' : 0.03,'objective' :'binary', 'metric' :'auc', 'verbose' : 2
    }

clf = lgb.train(params,d_train, valid_sets = d_valid,num_boost_round=2000,early_stopping_rounds = 200)    

test = timeFeatures(test)

test = test[final_train.columns]

submission = pd.read_csv("../input/sample_submission.csv")

pred = clf.predict(test)

submission["is_attributed"] = pred

submission.to_csv("undersampling_lgbm.csv", index = False)
