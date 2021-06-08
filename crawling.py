from bs4 import BeautifulSoup as bs
import requests
import pandas as pd

url = 'https://movie.naver.com/movie/sdb/rank/rmovie.nhn?sel=cur&date=20210503'
request = requests.get(url)
soup = bs(request.text, 'lxml')
meta = soup.select('meta')

lstRanking = soup.select(".list_ranking")[0]
title = [x.text.strip() for x in lstRanking.find_all('div', class_='tit5')]
score = [x.text.strip() for x in lstRanking.find_all('td', class_='point')]

ranking = range(1,len(score)+1)
link = ['https://movie.naver.com' + x.a.get('href') for x in lstRanking.find_all('div', class_='tit5')]
movieTable = pd.DataFrame({'rank': ranking, 'title': title, 'score': score, 'link': link})
movieTable['date'] = '2021-05-03'
print(movieTable.head())