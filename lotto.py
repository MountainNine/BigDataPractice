import json

import numpy as np
import pandas as pd
import requests
from fbprophet import Prophet


def crawl_lotto_num():
    basic_url = 'https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo='
    df_lotto_num = pd.DataFrame()
    for x in range(1, 10000):
        req_url = basic_url + str(x)
        json_res = json.loads(requests.get(req_url).text)
        if 'drwNo' in json_res.keys():
            df_response = pd.DataFrame(json_res, index=[x])
            df_sub = df_response[
                ['drwNoDate', 'drwtNo1', 'drwtNo2', 'drwtNo3', 'drwtNo4', 'drwtNo5', 'drwtNo6', 'bnusNo']]
            if x == 1:
                df_lotto_num = df_sub
            else:
                df_lotto_num = df_lotto_num.append(df_sub)
            print(str(x), "crawled")
        else:
            break
    df_lotto_num.to_csv('data/lotto_num.csv', index=False)


df_lotto_num = pd.read_csv("data/lotto_num.csv")
result = []
for c in ['drwtNo1', 'drwtNo2', 'drwtNo3', 'drwtNo4', 'drwtNo5', 'drwtNo6', 'bnusNo']:
    prophet = Prophet(seasonality_mode='multiplicative',
                      yearly_seasonality=True,
                      weekly_seasonality=True,
                      daily_seasonality=False,
                      changepoint_prior_scale=0.5)

    df_lotto_sub = df_lotto_num[['drwNoDate', c]]
    df_lotto_sub.columns = ['ds', 'y']
    prophet.fit(df_lotto_sub)
    future_data = prophet.make_future_dataframe(periods=7, freq='D')
    forecast_data = prophet.predict(future_data)
    if not result:
        ts = pd.to_datetime(str(forecast_data['ds'].tail(1).values[0]))
        result.append(ts.strftime("%Y-%m-%d"))
    result.append(np.round(forecast_data['yhat'].tail(1).values[0]))

print(result)

# 973 : 6 13 22 28 34 41
