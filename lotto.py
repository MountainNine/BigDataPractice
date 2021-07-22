import json

import pandas as pd
import requests


def crawl_lotto_num():
    basic_url = 'https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo='
    df_lotto_num = pd.DataFrame()
    for x in range(1, 10000):
        req_url = basic_url + str(x)
        json_res = json.loads(requests.get(req_url).text)
        if 'drwNo' in json_res.keys():
            df_response = pd.DataFrame(json_res, index=[x])
            df_sub = df_response[['drwNo', 'drwtNo1', 'drwtNo2', 'drwtNo3', 'drwtNo4', 'drwtNo5', 'drwtNo6', 'bnusNo']]
            if x == 1:
                df_lotto_num = df_sub
            else:
                df_lotto_num = df_lotto_num.append(df_sub)
            print(str(x), "crawled")
        else:
            break
    df_lotto_num.to_csv('data/lotto_num.csv', index=False)
