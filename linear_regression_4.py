import pandas as pd

train_x = pd.read_csv('https://raw.githubusercontent.com/Datamanim/mosquito/main/train_x.csv', encoding='euc-kr')
train_y = pd.read_csv('https://raw.githubusercontent.com/Datamanim/mosquito/main/train_y.csv', encoding='euc-kr')
test_x = pd.read_csv('https://raw.githubusercontent.com/Datamanim/mosquito/main/test_x.csv', encoding='euc-kr')
sub = pd.read_csv('https://raw.githubusercontent.com/Datamanim/mosquito/main/sub.csv')
