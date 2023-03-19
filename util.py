import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
import random
import scipy.io
import pandas as pd
class DataLoader(object):
    def __init__(self, xs, ys, xs_interpolation, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            x_interpolation_padding = np.repeat(xs_interpolation[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
            xs_interpolation = np.concatenate([xs_interpolation, x_interpolation_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        self.xs_interpolation = xs_interpolation

    def shuffle(self):

        permutation = np.random.permutation(self.size)
        
        xs, ys, x_i_interpolation = self.xs[permutation], self.ys[permutation], self.xs_interpolation[permutation]
        self.xs = xs
        self.ys = ys
        self.xs_interpolation = x_i_interpolation

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]   #这两个是上面一路预测的输入输出
                x_i_interpolation = self.xs_interpolation[start_ind: end_ind, ...]

                yield (x_i, y_i, x_i_interpolation)
                self.current_ind += 1

        return _wrapper()

class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean



def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_adj(pkl_filename, adjtype):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return sensor_ids, sensor_id_to_ind, adj


def generate_data(data, train_length=12, pred_length=1):
    x_list = []
    y_list = []
    x_interpolation = []
    for i in range(data.shape[0] - train_length - pred_length + 1):
        x_list.append(data[i: i + train_length])
        y_list.append(data[i + train_length: i+train_length+pred_length])  # 这个即可以做预测的标签，也可以做插补的标签
        x_interpolation.append(data[i: i + train_length + pred_length])
    x_list, y_list = np.stack(x_list,axis=0),np.stack(y_list,axis=0)
    x_interpolation = np.stack(x_interpolation, axis=0)
    print(x_list.shape, y_list.shape, x_interpolation.shape)
    return x_list, y_list, x_interpolation

def load_dataset(training_set, val_set, test_set, batch_size, valid_batch_size, test_batch_size, no_nm, n_m, unknow_set, A_s, history=12, predict=1):
    
    # training_set, val_set, test_set shape : T,B,N
    print(training_set.shape)
    data = {}
    data['x_train_pred'], data['y_train'] , data['x_train_interpolation'] = generate_data(training_set, history, predict)
    data['x_val_pred'], data['y_val'] , data['x_val_interpolation'] = generate_data(val_set, history, predict)
    data['x_test_pred'], data['y_test'] , data['x_test_interpolation'] = generate_data(test_set, history, predict)
    
    scaler = StandardScaler(mean=data['x_train_interpolation'][..., 0].mean(), std=data['x_train_interpolation'][..., 0].std())
    for category in ['train', 'val', 'test']:
        data['x_' + category + '_pred'][..., 0] = scaler.transform(data['x_' + category +'_pred'][..., 0])
        data['x_' + category + '_interpolation'][..., 0] = scaler.transform(data['x_' + category +'_interpolation'][..., 0])
        # data['x_' + category][:] = scaler.transform(data['x_' + category][:])


    print(data['x_train_pred'].shape, data['y_train'].shape, data['x_train_interpolation'].shape)
    data['train_loader'] = DataLoader(data['x_train_pred'], data['y_train'], data['x_train_interpolation'], batch_size)
    data['val_loader'] = DataLoader(data['x_val_pred'], data['y_val'], data['x_val_interpolation'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test_pred'], data['y_test'], data['x_test_interpolation'], test_batch_size)   # 生成了上面预测的一路的data

    data['scaler'] = scaler


    return data

def load_adj_new(A, adjtype):
    adj_mx = A
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return adj

def make_graph_inputs_new(args, device):
    # sensor_ids, sensor_id_to_ind, adj_mx = load_adj(args.adjdata, args.adjtype)
    adj_mx = load_adj_new(args.adj_file, args.adjtype)  # 加载邻接矩阵
    print('adj_mx_shape', len(adj_mx))
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    aptinit = None if args.randomadj else supports[0]  # ignored without do_graph_conv and add_apt_adj
    if args.aptonly:
        if not args.addaptadj and args.do_graph_conv: raise ValueError(
            'WARNING: not using adjacency matrix')
        supports = None
    return aptinit, supports

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse

def mask_np(array, null_val):
    if np.isnan(null_val):
        return (~np.isnan(array)).astype('float32')
    else:
        return np.not_equal(array, null_val).astype('float32')

def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = mask_np(y_true, null_val)
        mask /= mask.mean()
        mape = np.abs((y_pred - y_true) / y_true)
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100

def masked_mse_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mse = (y_true - y_pred) ** 2
    return np.mean(np.nan_to_num(mask * mse))

def masked_mae_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mae = np.abs(y_true - y_pred)
    return np.mean(np.nan_to_num(mask * mae))
    
def load_data(dataset, n_u, seed):
    '''Load dataset
    Input: dataset name
    Returns
    -------
    A: adjacency matrix 
    X: processed data
    capacity: only works for NREL, each station's capacity
    '''
    capacities = []
    E_maxvalue = None
    # data_load_path = "/mnt/nfs-storage-node-10/data/kriging/"  + dataset + '/'
    data_load_path = './data/' + dataset + '/'
    if dataset == 'METR-LA':
        A, X = np.load(data_load_path+"adj.npy"), np.load(data_load_path+"data.npy")
        X = X[:,:,0:1]
        print(A.shape)
        print(X.shape)
    elif dataset == 'PEMS-BAY':
        X, A = np.load(data_load_path + "data.npy"), np.load(data_load_path + "adj.npy")
        # X = X.transpose()
        A = A.astype('float32')
        X = X.astype('float32')
        X = np.expand_dims(X,axis=2)
        print(X.shape, A.shape)
        # exit()
    elif "PEMSD7-L" in dataset:
        X = np.load(data_load_path + "flow.npy")
        A = np.load(data_load_path + "adj_spatial.npy")
        X = X[:,:,0:1]
        # X = X.transpose()

        A = A.astype('float32')
        X = X.astype('float32')
        print(A.shape, X.shape)
    elif "BJ-AIR" in dataset:
        X = np.load(data_load_path + "data.npy")
        A = np.load(data_load_path + "adj.npy")

        X = np.expand_dims(X,axis=-1)

        A = A.astype('float32')
        X = X.astype('float32')
        print(A.shape, X.shape)
    else:
        raise NotImplementedError('Please specify datasets from: metr, nrel, ushcn, sedata or pems')
    # exit()
    print(X.shape)
    # X: T,N,F
    train_val_split = int(X.shape[0] * 0.6)
    val_test_split = int(X.shape[0] * 0.8)
    training_set = X[:train_val_split]
    val_set = X[train_val_split:val_test_split]       # split the training and test period
    test_set = X[val_test_split:]
    print(training_set.shape)
    print(val_set.shape)
    print(test_set.shape)
    
    print("--------- load data ok -----------")
    rand = np.random.RandomState(seed) # Fixed random output
    unknow_set = rand.choice(list(range(0,X.shape[1])),n_u,replace=False)
    unknow_set = set(unknow_set)
    full_set = set(range(0,X.shape[1]))
    know_set = full_set - unknow_set


    training_set_s = training_set[:, list(know_set)]   # get the training data in the sample time period
    A_s = A[:, list(know_set)][list(know_set), :]      # get the observed adjacent matrix from the full adjacent matrix,
                                                    # the adjacent matrix are based on pairwise distance, 
                                                    # so we need not to construct it for each batch, we just use index to find the dynamic adjacent matrix  
    return A,X,training_set_s,val_set,test_set,unknow_set,full_set,know_set,A_s