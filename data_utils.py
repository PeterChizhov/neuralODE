
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy import interpolate


from pathlib import Path
import pickle


import matplotlib.pyplot as plt


def create_hyst_dataset(total_size = 100,test_size=0.25):
    t = np.linspace(-3*np.pi/4, np.pi/4, total_size, endpoint = True)
    x = np.sin(t)
    y = np.cos(t)
    plt.figure(figsize = (7, 5), dpi = 80)
    plt.plot(t, x, color = 'black')
    plt.plot(t, y, color = 'black', label = 'y(x)')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Исходная функция')
    plt.show()
    no_noise_y = np.zeros((len(x), 2))
    no_noise_y[:,0] = x 
    no_noise_y[:,1] = y

    x += 0.04*np.random.normal(0,1,total_size)
    y += 0.04*np.random.normal(0,1,total_size)
    y[0] = x[0]; y[-1] = x[-1]
    plt.figure(figsize = (7, 5), dpi = 80)
    plt.plot(t, x, color = 'black')
    plt.plot(t, y, color = 'black', label = 'y(x)')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Зашумленная функция')
    plt.show()

    label = np.zeros((len(x), 2))
    label[:,0] = x 
    label[:,1] = y
    X = np.zeros((len(t), 2))
    X[:,0] = t
    X[:,1] = 1.
    label_tensor = torch.Tensor(label)
    X_tensor = torch.Tensor(X)
    # hyst_dataset = TensorDataset(X_tensor, label_tensor)
    # t_tensor = torch.tensor(t)
    # hyst_simple_dataset = TensorDataset(t_tensor, label_tensor)

    X_train, X_test, y_train, y_test = train_test_split(X_tensor, label_tensor, test_size=test_size)

    batch_size = len(X_tensor)
    n_iters = 10000
    total_data_size = len(X_tensor)
    num_epochs = n_iters / (total_data_size / batch_size)
    num_epochs = int(num_epochs)
    # num_epochs = 1
    print(f'Total data size = {total_data_size}')
    print(f'Batch size = {batch_size}')
    print(f'Max number of optimixation steps = {n_iters}')
    print(f'Max number epochs = {num_epochs}')



    # plt.figure(figsize = (7, 5), dpi = 80)
    # plt.plot(X_train[:, 0], x, color = 'black')
    # plt.plot(t, y, color = 'black', label = 'y(x)')
    # plt.legend()
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Зашумленная функция')
    # plt.show()


    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    # with open('train_dataset_hyst.pickle', 'wb') as f:
    #     pickle.dump(train_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    # with open('test_dataset_hyst.pickle', 'wb') as f:
    #     pickle.dump(test_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)    


    # Define data loaders for training and testing data in this fold
    # trainloader = torch.utils.data.DataLoader(
    #                   train_dataset, 
    #                   batch_size=batch_size)
    # testloader = torch.utils.data.DataLoader(
    #                   test_dataset,
    #                   batch_size=batch_size)

    plt.figure(figsize=(7, 5), dpi=80)
    sc_1 = plt.scatter(X_train[:, 0], y_train[:, 0], color = 'black', s=5, label = 'Тренировочные данные')
    plt.scatter(X_train[:, 0], y_train[:, 1], color = 'black', s=5)
    sc_2 = plt.scatter(X_test[:, 0], y_test[:, 0], color = 'red', s=5, label = 'Тестовые данные')
    plt.scatter(X_test[:, 0], y_test[:, 1], color = 'red', s=5)
    # plt.plot(X_tensor[:, 0], z_full.view(-1, 2)[:, 0], color = 'green', label = 'Аппроксимация')
    # plt.plot(X_tensor[:, 0], z_full.view(-1, 2)[:, 1], color = 'green')
    # plt.plot(t, np.sin(t), color = 'pink', label = 'Незашумленные данные')
    # plt.plot(t, np.cos(t), color = 'pink')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Разбиение выборки')
    plt.show()

    return train_dataset, test_dataset, X_tensor, label_tensor, no_noise_y


def load_existing_hyst_dataset(total_size = 100,test_size=0.25):
    data_path = Path('..') / 'hyst_dataset' 
    with open(data_path/'train_dataset_hyst.pickle', 'rb') as f:
        train_dataset = pickle.load(f)
    with open(data_path/'test_dataset_hyst.pickle', 'rb') as f:
        test_dataset = pickle.load(f)
    full_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset]) 
    X_full = []
    y_full = []
    for x, y in full_dataset:
        X_full.append(x)
        y_full.append(y)
    # упорядочиваем для визуализации    
    X_full = torch.stack(X_full, 0)
    sorted, indices = torch.sort(X_full[:, 0])
    y_full = torch.stack(y_full, 0)  
    X_full = X_full[indices]
    y_full = y_full[indices]

    t = np.linspace(-3*np.pi/4, np.pi/4, total_size, endpoint = True)
    x = np.sin(t)
    y = np.cos(t)  
    no_noise_y = np.zeros((len(x), 2))
    no_noise_y[:,0] = x 
    no_noise_y[:,1] = y

    return train_dataset, test_dataset, X_full, y_full, no_noise_y   

def create_experimental_dataset():
    data_path = Path('..') / 'data' 
    df1 = pd.read_csv(data_path/"Dataset (1).csv", header = None, sep = ';')
    df1.columns = ['x', 'y']
    df2 = pd.read_csv(data_path/"Dataset (2).csv", header = None, sep = ';')
    df2.columns = ['x', 'y']
    # interpolation and extrapolation
    I = interpolate.interp1d(df1['x'], df1['y'], fill_value='extrapolate')
    y2 = I(df2['x'])
    df2['y2'] = y2
    # min-max normalization
    glob_min = min(min(df2['y']), min(df2['y2']))
    glob_max = max(max(df2['y']), max(df2['y2']))
    df2['y'] = (df2['y'] - glob_min)/(glob_max - glob_min)
    df2['y2'] = (df2['y2'] - glob_min)/(glob_max - glob_min) 

    # plotting true data
    plt.figure(figsize = (7, 5), dpi = 80)
    plt.plot(df2['x'], df2['y'], color = 'blue')
    plt.plot(df2['x'], df2['y2'], color = 'black', label = 'y(x)')
    plt.legend()
    plt.xlabel('V')
    plt.ylabel('I')
    plt.title('Экспериментальные данные')
    plt.show()

    label = np.zeros((len(df2), 2))
    label[:,0] = df2['y'] 
    label[:,1] = df2['y2']
    X = np.zeros((len(df2), 2))
    X[:,0] = df2['x']
    X[:,1] = 1.
    label_tensor = torch.Tensor(label)
    X_tensor = torch.Tensor(X)
    indices = np.arange(0, len(X_tensor))
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X_tensor, label_tensor, indices, test_size=0.25)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)


    plt.figure(figsize=(7, 5), dpi=80)
    sc_1 = plt.scatter(X_train[:, 0], y_train[:, 0], color = 'black', s=5, label = 'Тренировочные данные')
    plt.scatter(X_train[:, 0], y_train[:, 1], color = 'black', s=5)
    sc_2 = plt.scatter(X_test[:, 0], y_test[:, 0], color = 'red', s=5, label = 'Тестовые данные')
    plt.scatter(X_test[:, 0], y_test[:, 1], color = 'red', s=5)
    # plt.plot(X_tensor[:, 0], z_full.view(-1, 2)[:, 0], color = 'green', label = 'Аппроксимация')
    # plt.plot(X_tensor[:, 0], z_full.view(-1, 2)[:, 1], color = 'green')
    # plt.plot(t, np.sin(t), color = 'pink', label = 'Незашумленные данные')
    # plt.plot(t, np.cos(t), color = 'pink')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Разбиение выборки')
    plt.show()
         
    with open(data_path/'train_dataset.pickle', 'wb') as f:
        pickle.dump(train_dataset, f)
    with open(data_path/'test_dataset.pickle', 'wb') as f:
        pickle.dump(test_dataset, f)
    with open(data_path/'indices_train.pickle', 'wb') as f:
        pickle.dump(indices_train, f)
    with open(data_path/'indices_test.pickle', 'wb') as f:
        pickle.dump(indices_test, f)  
    
    with open(data_path/'X_full.pickle', 'wb') as f:
        pickle.dump(X_tensor, f)
    with open(data_path/'y_full.pickle', 'wb') as f:
        pickle.dump(label_tensor, f)    

    with open(data_path/'min_max_scalers.pickle', 'wb') as f:
        pickle.dump((glob_min, glob_max), f)    

    no_noise_y = None
    return train_dataset, test_dataset, X_tensor, label_tensor, (X_train, X_test, y_train, y_test)
        
def unscale_I(data, glob_min, glob_max):
    return data * (glob_max - glob_min) + glob_min

def load_existing_experimental_dataset():
    data_path = Path('..') / 'data' 
    with open(data_path/'train_dataset.pickle', 'rb') as f:
        train_dataset = pickle.load(f)
    with open(data_path/'train_dataset.pickle', 'rb') as f:
        test_dataset = pickle.load(f)

    with open(data_path/'indices_train.pickle', 'rb') as f:
        indices_train = pickle.load(f)
    with open(data_path/'indices_test.pickle', 'rb') as f:
        indices_test = pickle.load(f)

    with open(data_path/'X_full.pickle', 'rb') as f:
        X_full = pickle.load(f)
    with open(data_path/'y_full.pickle', 'rb') as f:
        y_full = pickle.load(f)   
    with open(data_path/'min_max_scalers.pickle', 'rb') as f:
        glob_min, glob_max = pickle.load(f)         
    
    X_train = X_full[indices_train, :]
    y_train = y_full[indices_train, :]
    X_test = X_full[indices_test, :]
    y_test = y_full[indices_test, :]
    no_noise_y = (X_train, X_test, y_train, y_test, glob_min, glob_max)
    return train_dataset, test_dataset, X_full, y_full, no_noise_y


def viz_existing_experimental_dataset():
    data_path = Path('..') / 'data' 
    with open(data_path/'train_dataset.pickle', 'rb') as f:
        train_dataset = pickle.load(f)
    with open(data_path/'train_dataset.pickle', 'rb') as f:
        test_dataset = pickle.load(f)

    with open(data_path/'indices_train.pickle', 'rb') as f:
        indices_train = pickle.load(f)
    with open(data_path/'indices_test.pickle', 'rb') as f:
        indices_test = pickle.load(f)

    with open(data_path/'X_full.pickle', 'rb') as f:
        X_full = pickle.load(f)
    with open(data_path/'y_full.pickle', 'rb') as f:
        y_full = pickle.load(f)

    with open(data_path/'min_max_scalers.pickle', 'rb') as f:
        glob_min, glob_max = pickle.load(f)   

    y_full = unscale_I(y_full, glob_min, glob_max)
    plt.figure(figsize = (7, 5), dpi = 80)
    plt.plot(X_full[:, 0], y_full[:, 0], color = 'black')
    plt.plot(X_full[:, 0], y_full[:, 1], color = 'black')
    plt.legend()
    plt.xlabel('V')
    plt.ylabel('I')
    plt.title('Экспериментальные данные')
    plt.grid()
    plt.show()
    no_noise_y = None
    X_train = X_full[indices_train, :]
    y_train = y_full[indices_train, :]
    X_test = X_full[indices_test, :]
    y_test = y_full[indices_test, :]
    plt.figure(figsize=(7, 5), dpi=80)
    sc_1 = plt.scatter(X_train[:, 0], y_train[:, 0], color = 'black', s=5, label = 'Тренировочные данные')
    plt.scatter(X_train[:, 0], y_train[:, 1], color = 'black', s=5)
    sc_2 = plt.scatter(X_test[:, 0], y_test[:, 0], color = 'red', s=5, label = 'Тестовые данные')
    plt.scatter(X_test[:, 0], y_test[:, 1], color = 'red', s=5)
    # plt.plot(X_tensor[:, 0], z_full.view(-1, 2)[:, 0], color = 'green', label = 'Аппроксимация')
    # plt.plot(X_tensor[:, 0], z_full.view(-1, 2)[:, 1], color = 'green')
    # plt.plot(t, np.sin(t), color = 'pink', label = 'Незашумленные данные')
    # plt.plot(t, np.cos(t), color = 'pink')
    plt.legend()
    plt.xlabel('V')
    plt.ylabel('I')
    plt.title('Разбиение выборки')
    plt.grid()
    plt.show()
    return train_dataset, test_dataset, X_full, y_full, no_noise_y



def create_windowed_experimental_dataset(test_frac, window_size=32):
    data_path = Path('..') / 'data' 
    new_data_path = Path('..') / 'window_data'
    iv_timed_df = pd.read_csv(data_path/"I_V_timed.csv")
    
    # min-max normalization only for I
    glob_min_I = min(iv_timed_df['I'])
    glob_max_I = max(iv_timed_df['I'])
    iv_timed_df['I_scaled'] = (iv_timed_df['I'] - glob_min_I)/(glob_max_I - glob_min_I) 

    iv_arr = iv_timed_df[['V', 'I']].values
    
    windowed_arr = []
    for ind_start in range(len(iv_arr)):
        ind_end = ind_start + window_size
        
        if ind_end > len(iv_arr):
            cycle_slice = iv_arr[ind_start : ind_end]
            cycle_slice = np.append(cycle_slice, iv_arr[0:ind_end -len(iv_arr)], axis=0)
        else:
            cycle_slice = iv_arr[ind_start : ind_end]
        windowed_arr.append(cycle_slice)
    iv_windowed = np.stack(windowed_arr, axis=0)
    print(iv_windowed.shape) 

    target_I = iv_windowed[:,:, 1]


    # plotting true data
    plt.figure(figsize = (7, 5), dpi = 80)
    plt.plot(df2['x'], df2['y'], color = 'blue')
    plt.plot(df2['x'], df2['y2'], color = 'black', label = 'y(x)')
    plt.legend()
    plt.xlabel('V')
    plt.ylabel('I (scaled)')
    plt.title('Экспериментальные данные')
    plt.show()

    label = np.zeros((len(df2), 2))
    label[:,0] = df2['y'] 
    label[:,1] = df2['y2']
    X = np.zeros((len(df2), 2))
    X[:,0] = df2['x']
    X[:,1] = 1.
    label_tensor = torch.Tensor(label)
    X_tensor = torch.Tensor(X)
    indices = np.arange(0, len(X_tensor))
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X_tensor, label_tensor, indices, test_size=0.25)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)


    plt.figure(figsize=(7, 5), dpi=80)
    sc_1 = plt.scatter(X_train[:, 0], y_train[:, 0], color = 'black', s=5, label = 'Тренировочные данные')
    plt.scatter(X_train[:, 0], y_train[:, 1], color = 'black', s=5)
    sc_2 = plt.scatter(X_test[:, 0], y_test[:, 0], color = 'red', s=5, label = 'Тестовые данные')
    plt.scatter(X_test[:, 0], y_test[:, 1], color = 'red', s=5)
    # plt.plot(X_tensor[:, 0], z_full.view(-1, 2)[:, 0], color = 'green', label = 'Аппроксимация')
    # plt.plot(X_tensor[:, 0], z_full.view(-1, 2)[:, 1], color = 'green')
    # plt.plot(t, np.sin(t), color = 'pink', label = 'Незашумленные данные')
    # plt.plot(t, np.cos(t), color = 'pink')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Разбиение выборки')
    plt.show()
         
    with open(data_path/'train_dataset.pickle', 'wb') as f:
        pickle.dump(train_dataset, f)
    with open(data_path/'test_dataset.pickle', 'wb') as f:
        pickle.dump(test_dataset, f)
    with open(data_path/'indices_train.pickle', 'wb') as f:
        pickle.dump(indices_train, f)
    with open(data_path/'indices_test.pickle', 'wb') as f:
        pickle.dump(indices_test, f)  
    
    with open(data_path/'X_full.pickle', 'wb') as f:
        pickle.dump(X_tensor, f)
    with open(data_path/'y_full.pickle', 'wb') as f:
        pickle.dump(label_tensor, f)    

    with open(data_path/'min_max_scalers.pickle', 'wb') as f:
        pickle.dump((glob_min, glob_max), f)    

    no_noise_y = None
    return train_dataset, test_dataset, X_tensor, label_tensor, (X_train, X_test, y_train, y_test)