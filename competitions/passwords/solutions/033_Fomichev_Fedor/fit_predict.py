import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import xgboost as xgb

def write2submission(df):
    with open('./sub.csv', 'w') as f:
       f.write('Id,Times' + '\n')
       for _, row in df.iterrows():
           f.write(str(row['Id']) + ',' + str(row['pred']) + '\n')

PATH_TO_TRAIN = './train.csv'
PATH_TO_TEST = './Xtest.csv'

train = pd.read_csv(PATH_TO_TRAIN)
test = pd.read_csv(PATH_TO_TEST)

train_X = train[['has_name', 'key_dist', 'has_alpha', 'has_digit',
                 'has_upper', 'has_lower', 'has_year', 'has_worst',
                 'pass_prob']]

train_y = train['log_y']

params = {'n_estimators': [50] + [x for x in range(100, 200, 5)],
          'max_depth': [x for x in range(3, 15)]}

gbm = xgb.XGBRegressor()

cv = RandomizedSearchCV(gbm, params, scoring='neg_mean_squared_log_error')
cv.fit(train_X, train_y)

test_X = test[['has_name', 'key_dist', 'has_alpha', 'has_digit',
                'has_upper', 'has_lower', 'has_year', 'has_worst',
                'pass_prob']]

predictions = cv.predict(test_X)
test['pred'] = np.exp(predictions)
write2submission(test)

gbm = xgb.XGBRegressor(max_depth=5, n_estimators=150)
gbm.fit(train_X, train_y)
predictions = gbm.predict(test_X)
test['pred'] = np.exp(predictions)
write2submission(test)
