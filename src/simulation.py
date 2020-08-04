print('\nStock Client\n')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR, StepLR
import copy
import torch.utils.data as tdata
from train import Model, predict, get_accuracy, get_horizon, get_filename, get_unseendata
import sys

try:
    init_w = float(sys.argv[1])
except:
    print('''      This program requires one parameter:
      1. init_w(float) - Initial Wallet Size
          ''')
    sys.exit(0)

timegap = get_unseendata()
init_stock = 0
horizon = get_horizon()

class Wallet():
    def __init__(self, wallet, stock):
        self.wallet = wallet
        self.stock = stock

    def purchase(self, value, price):
        if value == 1:
            halfwallet = self.wallet/2
            while self.wallet - halfwallet - price > 0:
                self.wallet -= price
                self.stock += 1
        else:
            halfstock = self.stock/2
            while self.stock - halfstock > 0:
                self.wallet += price
                self.stock -= 1

def transaction(yesterday, today):
    #code to decide if buy or sell
    if today > yesterday:
        return 1
    else:
        return -1

wallet = Wallet(init_w, init_stock)

file_name = get_filename()

file = open(file_name, 'r')
fl = file.readlines()[-timegap :]
idate = fl[0].split(',')[0]
fdate = fl[-1].split(',')[0]

print('Working on data from {} to {}'.format(idate, fdate))

my_data = np.genfromtxt(file_name, delimiter=',')[1:,1:-2]

my_data = my_data[-timegap-horizon:]
Xclose = my_data[:,3]

sample_length = len(my_data)
print('Days of simulation: ', timegap)
print('Sliding Window Size: ', horizon)
print('Sample Length: ',sample_length, '\n')
assert timegap == sample_length - horizon

input_dim = horizon*2+1
output_dim = 1

model = Model(input_dim, output_dim)
model.load_state_dict(torch.load('models/model'))
model.eval()

last_prediction = None
for i in range(timegap):
    train = my_data[i:i+horizon]
    day = my_data[i+horizon].reshape(my_data.shape[1],1)
    input_data = np.concatenate((train[:,0], day[0],train[:,3]))
    open_price = day[0]
    prediction = predict(model, input_data)
    if last_prediction != None:
        value = transaction(last_prediction, prediction)
        wallet.purchase(value, open_price)
    last_prediction = prediction
    final_price = day[3][0]

total = wallet.wallet + wallet.stock*final_price
profit = total - init_w
print('Init Wallet: ', init_w)
print('Number of Stocks: {}, \t Stock Value: {}'.format(wallet.stock, final_price))
print('Available in Wallet: ', wallet.wallet[0])
print('Total in Wallet: ', total[0])
print('Profit:', profit[0])
