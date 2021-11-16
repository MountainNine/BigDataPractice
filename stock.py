import pandas as pd
import numpy as np
import os
import FinanceDataReader as fdr

from sklearn.linear_model import LinearRegression
from tqdm import tqdm

path = './data/stock'
list_name = 'stock_list.csv'
sample_name = 'sample_submission.csv'

sample_submission = pd.read_csv(os.path.join(path,sample_name))
stock_list = pd.read_csv(os.path.join(path, list_name))
stock_list['종목코드'] = stock_list['종목코드'].apply(lambda x: str(x).zfill(6))

start_date = '20210104'
end_date = '20211105'

start_weekday = pd.to_datetime(start_date).weekday()
max_weeknum = pd.to_datetime(end_date).strftime("%V")
business_days = pd.DataFrame(pd.date_range(start_date,end_date,freq='B'),columns=['Date'])
model = LinearRegression()

for code in tqdm(stock_list['종목코드'].values):
    data = fdr.DataReader(code, start=start_date, end=end_date)[['Close']].reset_index()
    data = pd.merge(business_days, data, how='outer')
    data['weekday'] = data.Date.apply(lambda x: x.weekday())
    data['weeknum'] = data.Date.apply(lambda x : x.strftime('%V'))
    data.Close = data.Close.ffill()
    data = pd.pivot_table(data= data, values='Close', columns='weekday', index='weeknum')

    x = data.iloc[0:-2].to_numpy()
    y = data.iloc[1:-1].to_numpy()
    y_0 = y[:,0]
    y_1 = y[:,1]
    y_2 = y[:,2]
    y_3 = y[:,3]
    y_4 = y[:,4]

    y_values = [y_0, y_1, y_2, y_3, y_4]
    x_public = data.iloc[-2].to_numpy()

    predictions = []
    for y_value in y_values:
        model.fit(x, y_value)
        prediction = model.predict(np.expand_dims(x_public, 0))
        predictions.append(prediction[0])
    sample_submission.loc[:, code] = predictions * 2
    sample_submission.isna().sum().sum()

columns = list(sample_submission.columns[1:])
columns = ['Day'] + [str(x).zfill(6) for x in columns]
sample_submission.columns = columns
sample_submission.to_csv('./data/stock/BASELINE_Linear.csv',index=False)
