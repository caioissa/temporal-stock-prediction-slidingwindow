import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR, StepLR
import copy
import torch.utils.data as tdata

horizon = 12
unseendata = 50
file_name = './data/RUN.csv'

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dense = nn.Sequential(
            nn.Linear(input_dim, 2*input_dim + 1),
            nn.ReLU(),
            nn.Linear(2*input_dim + 1, 2*input_dim),
            nn.ReLU(),
            nn.Linear(2*input_dim, output_dim)
        )

    def forward(self, x):
        x = self.dense(x)
        return x

    def clone(self):
        model_copy = type(self)(input_dim,output_dim)
        model_copy.load_state_dict(self.state_dict())
        return model_copy

def predict(model, inputs):
    if (isinstance(inputs, np.ndarray)):
        inputs = torch.FloatTensor(inputs)
    outputs = model(Variable(inputs))
    return outputs.data.numpy()

def get_accuracy(model, inputs, targets):
    if (isinstance(inputs, np.ndarray)):
        inputs = torch.FloatTensor(inputs)

    outputs = model(Variable(inputs))
    outputs = outputs.data.numpy()

    if (isinstance(targets, torch.Tensor)):
        targets = targets.numpy().reshape(len(targets),1)

    accuracy = ((outputs - targets) ** 2).mean(axis=0)

    return accuracy

def get_filename():
    return file_name

def get_horizon():
    return horizon

def get_unseendata():
    return unseendata

if __name__ == '__main__':
    file = open(file_name, 'r')
    fl = file.readlines()[:-unseendata]
    idate = fl[1].split(',')[0]
    fdate = fl[-1].split(',')[0]

    print('Training on data from {} to {}'.format(idate, fdate))

    my_data = np.genfromtxt(file_name, delimiter=',')[1:,1:-2]

    my_data = my_data[:-unseendata]
    #normalization
    row_max = my_data.max(axis=0)
    row_min = my_data.min(axis=0)
    norm_data = (my_data - row_min) / (row_max - row_min)

    #definitions
    Yclose = my_data[horizon+1:,3]
    Xopen = norm_data[:,0]
    Xhigh = norm_data[:,1]
    Xlow = norm_data[:,2]
    Xclose = norm_data[:,3]

    #inicialization
    index = 0
    sample_index = np.arange(index, index+horizon+1)
    sample_input = np.concatenate((Xopen[index:index+horizon+1], Xclose[index:index+horizon]),axis=0)
    sample_output = np.asarray(Xclose[index+horizon+1])
    dataset_index = np.array([sample_index])
    dataset_input = np.array([sample_input])
    dataset_output = np.array([sample_output])

    #creating all samples
    for index in range(1, len(Xopen) - horizon - 1):
        sample_index = np.arange(index, index+horizon+1)
        sample_input = np.concatenate((Xopen[index:index+horizon+1],Xclose[index:index+horizon]),axis=0)
        sample_output = np.asarray(Xclose[index+horizon+1])

        dataset_index = np.insert(arr=dataset_index, obj=len(dataset_index), values=sample_index, axis=0)
        dataset_input = np.insert(arr=dataset_input, obj=len(dataset_input), values=sample_input, axis=0)
        dataset_output = np.insert(arr=dataset_output, obj=len(dataset_output), values=sample_output, axis=0)

    dataset_output = np.reshape(dataset_output, (len(dataset_output), 1))

    num_samples = len(dataset_input)
    print("Total samples: {}".format(num_samples))

    #randomizing samples
    randomize = np.arange(num_samples)
    np.random.shuffle(randomize)
    shuffled_input = dataset_input[randomize]
    shuffled_output = dataset_output[randomize]

    #train test split
    trainSize = int(num_samples*0.7)

    Xtest = shuffled_input[trainSize:]
    Ytest = shuffled_output[trainSize:]
    Xtrain = shuffled_input[:trainSize]
    Ytrain = shuffled_output[:trainSize]
    trainSize = len(Xtrain)
    testSize = len(Xtest)
    assert trainSize + testSize == num_samples

    train = torch.utils.data.TensorDataset(torch.Tensor(Xtrain), torch.Tensor(Ytrain))
    train_loader = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
    test = torch.utils.data.TensorDataset(torch.Tensor(Xtest), torch.Tensor(Ytest))
    test_loader = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)
    #criando modelo
    input_dim = Xtest.shape[1]
    output_dim = 1

    model = Model(input_dim, output_dim)
    print(model)

    learningRate = 0.05

    #Função de perda: MSELoss
    criterion = torch.nn.MSELoss(reduction='mean')

    #Otimizador: Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

    #Scheduler para reduzir lr a cada iteração
    scheduler = StepLR(optimizer, step_size=10, gamma=0.7)

    epochs = 50

    train_losses = []
    test_losses = []
    checkpoint = {'model':copy.deepcopy(model), 'train_loss':10e10, 'test_loss':10e10, 'best_epoch':0}
    Xtest_tensor = torch.FloatTensor(Xtest)
    Ytest_tensor = torch.FloatTensor(Ytest)

    for epoch in range(epochs):
        train_loss = 0
        for inputs,targets in train_loader:
            targets = targets.reshape(len(inputs),1)
            #forward
            output = model(Variable(inputs))

            #calculate loss
            loss = criterion(output, Variable(targets))/len(train_loader)
            train_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        train_loss = train_loss.data.numpy()
        test_loss = get_accuracy(model, Xtest_tensor, Ytest_tensor)[0]

        #Checkpoint
        if test_loss < checkpoint['test_loss']:
            checkpoint['test_loss'] = test_loss
            checkpoint['train_loss'] = train_loss
            checkpoint['best_epoch'] = epoch
            checkpoint['model'] = model.clone()

        train_losses.append(train_loss)
        test_losses.append(test_loss)

    best_model = checkpoint['model']
    inputs = torch.FloatTensor(Xtrain)
    targets = torch.FloatTensor(Ytrain)
    print('Training Loss: ', get_accuracy(best_model, inputs, targets)[0])

    inputs = torch.FloatTensor(Xtest)
    targets = torch.FloatTensor(Ytest)
    print('Test Loss: ', get_accuracy(best_model, inputs, targets)[0])

    predictions = predict(best_model, dataset_input)
    legend1, = plt.plot(predictions, label='predictions')
    legend2, = plt.plot(dataset_output, label='targets')
    plt.legend(['prediction', 'target'])
    plt.show()

    if not os.path.exists('./models'):
        os.mkdir('./models')

    torch.save(model.state_dict(),'models/model')
    print('saved to models/model')
