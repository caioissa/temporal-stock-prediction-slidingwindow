from datetime import datetime
import requests
import os

today = int(datetime.now().timestamp())
start = today - 31622763
stock = 'RUN'

r = requests.get('https://query1.finance.yahoo.com/v7/finance/download/{}?period1={}&period2={}&interval=1d&events=history'.format(stock, start, today))

if r.status_code == 200:
    if not os.path.exists('./data'):
        os.mkdir('./data')
    with open('./data/RUN.csv', 'w') as f:
        f.write(r.text)
