'''
Description: 
Author: voicebeer
Date: 2020-09-08 07:00:34
LastEditTime: 2021-12-22 01:53:49
'''

# For SEED data loading
from torch.utils.data import Dataset, DataLoader
import torch
from torch.autograd import Function
import pickle
import copy
import os
import scipy.io as scio
from torch import nn
# standard package
import numpy as np
import random
random.seed(0)
from scipy.spatial.distance import cdist
import scipy.io

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# DL

dataset_path = {'seed4': 'eeg_feature_smooth', 'seed3': 'ExtractedFeatures'}

'''
Tools
'''


def Pseudo_Label_Weight(x, T1, T2, bata):
    """
    假设这个分段函数定义如下:
    f(x) = {
        0, if x < 0
        x + 1, if 0 <= x < 2
        2x - 1, if x >= 2
    }
    """
    if x < T1:
        return 0
    elif T1 <= x < T2:
        return (x-T1) * bata / (T2-T1)
    elif T2 <= x :
        return bata


def norminx(data):
    '''
    description: norm in x dimension
    param {type}:
        data: array
    return {type} 
    '''
    for i in range(data.shape[0]):
        data[i] = normalization(data[i])
    return data


def norminy(data):
    dataT = data.T
    for i in range(dataT.shape[0]):
        dataT[i] = normalization(dataT[i])
    return dataT.T


def normalization(data):
    '''
    description: 
    param {type} 
    return {type} 
    '''
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

# package the data and label into one class


class CustomDataset(Dataset):
    # initialization: data and label
    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label
    # get the size of data

    def __len__(self):
        return len(self.Data)
    # get the data and label

    def __getitem__(self, index):
        data = torch.Tensor(self.Data[index])
        label = torch.LongTensor([self.Label[index]])
        return data, label, index

# mmd loss and guassian kernel


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                  for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)


def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss


def mmd_rbf_accelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i+1) % batch_size
        t1, t2 = s1+batch_size, s2+batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size)


def mmd_linear(f_of_X, f_of_Y):
    delta = f_of_X - f_of_Y
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss


def CORAL(source, target):
    d = source.data.shape[1]

    # source covariance
    xm = torch.mean(source, 1, keepdim=True) - source
    xc = torch.matmul(torch.transpose(xm, 0, 1), xm)

    # target covariance
    xmt = torch.mean(target, 1, keepdim=True) - target
    xct = torch.matmul(torch.transpose(xmt, 0, 1), xmt)
    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss / (4*d*4)
    return loss


def EntropyLoss(input_):
    mask = input_.ge(0.000001)
    mask_out = torch.masked_select(input_, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(input_.size(0))


def PADA(features, ad_net, grl_layer, weight_ad, use_gpu=True):
    ad_out = ad_net(grl_layer(features))
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(
        np.array([[1]] * batch_size + [[0]] * batch_size)).float()
    if use_gpu:
        dc_target = dc_target.cuda()
        weight_ad = weight_ad.cuda()
    return nn.BCELoss(weight=weight_ad.view(-1))(ad_out.view(-1), dc_target.view(-1))


def get_number_of_label_n_trial(dataset_name):
    '''
    description: get the number of categories, trial number and the corresponding labels
    param {type} 
    return {type}:
        trial: int
        label: int
        label_xxx: list 3*15
    '''
    # global variable
    label_seed4 = [[1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
                   [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2,
                       0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
                   [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]]
    label_seed3 = [[2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0],
                   [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0],
                   [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0]]
    if dataset_name == 'seed3':
        label = 3
        trial = 15
        return trial, label, label_seed3
    elif dataset_name == 'seed4':
        label = 4
        trial = 24
        return trial, label, label_seed4
    else:
        print('Unexcepted dataset name')


def reshape_data(data, label):
    '''
    description: reshape data and initiate corresponding label vectors
    param {type}:
        data: list
        label: list
    return {type}:
        reshape_data: array, x*310
        reshape_label: array, x*1
    '''
    reshape_data = None
    reshape_label = None
    for i in range(len(data)):
        one_data = np.reshape(np.transpose(
            data[i], (1, 2, 0)), (-1, 310), order='F') #先转置把“行”移到最后,后重塑,意思是将62和5合并为310
        one_label = np.full((one_data.shape[0], 1), label[i])
        if reshape_data is not None:
            reshape_data = np.vstack((reshape_data, one_data))      #行叠加在一起
            reshape_label = np.vstack((reshape_label, one_label))
        else:
            reshape_data = one_data
            reshape_label = one_label
    return reshape_data, reshape_label


def get_data_label_frommat(mat_path, dataset_name, session_id):
    '''
    description: load data from mat path and reshape to 851*310
    param {type}:
        mat_path: String
        session_id: int
    return {type}: 
        one_sub_data, one_sub_label: array (851*310, 851*1)
    '''
    _, _, labels = get_number_of_label_n_trial(dataset_name) # _(通常用下划线表示)可能是一个占位符，表示该值不会被使用或忽略。
    mat_data = scio.loadmat(mat_path)
    mat_de_data = {key: value for key,                  
                   value in mat_data.items() if key.startswith('de_LDS')}   #将de特征提取出来放入矩阵
    mat_de_data = list(mat_de_data.values())
    one_sub_data, one_sub_label = reshape_data(mat_de_data, labels[session_id])
    return one_sub_data, one_sub_label


def sample_by_value(list, value, number):
    '''
    @Description: sample the given list randomly with given value
    @param {type}: 
        list: list
        value: int {0,1,2,3}
        number: number of sampling
    @return: 
        result_index: list
    '''
    result_index = []
    index_for_value = [i for (i, v) in enumerate(list) if v == value]
    result_index.extend(random.sample(index_for_value, number))
    return result_index


'''
For loading data
'''


def get_allmats_name(dataset_name):
    '''
    description: get the names of all the .mat files
    param {type}
    return {type}:
        allmats: list (3*15)
    '''
    path = dataset_path[dataset_name]
    sessions = os.listdir(path)                            #获取文件名                 
    sessions.sort()                                        #.sort()升序
    allmats = []
    for session in sessions:
        if session != '.DS_Store':
            mats = os.listdir(path + '/' + session)
            mats.sort()
            mats_list = []
            for mat in mats:
                mats_list.append(mat)
            allmats.append(mats_list)
    return path, allmats


def load_data(dataset_name):
    '''
    description: get all the data from one dataset
    param {type} 
    return {type}:
        data: list 3(sessions) * 15(subjects), each data is x * 310
        label: list 3*15, x*1
    '''

    if dataset_name == 'seed3':
        path, allmats = get_allmats_name(dataset_name)   #allmats为3*15的文件名列表
        data = [([0] * 15) for i in range(3)]            #3*15
        label = [([0] * 15) for i in range(3)]   
        #data = np.zeros((3, 15, 851,310))
        #label = np.zeros((3, 15, 851,1))
        for i in range(len(allmats)):                    #循环遍历整个allmats
            for j in range(len(allmats[0])):
                mat_path = path + '/' + str(i+1) + '/' + allmats[i][j]
                one_data, one_label = get_data_label_frommat(
                    mat_path, dataset_name, i)
                data[i][j] = one_data.copy()
                label[i][j] = one_label.copy()
        return np.array(data), np.array(label)
    elif dataset_name == 'seed4':
        path, allmats = get_allmats_name(dataset_name)   #allmats为3*15的文件名列表
        data = [([0] * 15) for i in range(3)]            #3*15
        label = [([0] * 15) for i in range(3)]   
        for i in range(len(allmats)):                    #循环遍历整个allmats
            for j in range(len(allmats[0])):
                mat_path = path + '/' + str(i+1) + '/' + allmats[i][j]
                one_data, one_label = get_data_label_frommat(
                    mat_path, dataset_name, i)
                one_data = one_data[:822]
                one_label =one_label[:822]
                data[i][j] = one_data.copy()
                label[i][j] = one_label.copy()
        return np.array(data), np.array(label)

# def load_deap():
#     '''
#     description:
#     param {type}
#     return {type}
#     '''
#     path = 'deap'
#     dats = os.listdir(path)
#     dats.sort()

#     for i in range(1, len(dats)):
#         temp_dat_file = pickle.load(open((path+"/"+dats[i]), 'rb'), encoding='iso-8859-1')
#         temp_data, temp_label = temp_dat_file['data'], temp_dat_file['labels']
#         np.vstack((data, temp_data))
    # np.vstack((label, temp_label))
    # print(data.shape, label.shape)
    # for i in range()
    # x = pickle.load(open('deap/s01.dat', 'rb'), encoding='iso-8859-1')

    # return x

# print(load_deap()['data'].shape)
# load_deap()

# def initial_cd_ud(data, label, cd_count=16, dataset_name):
#     cd_list, ud_list = [], []
#     number_trial, number_label, _ = get_number_of_label_n_trial(dataset_name)
#     for i in range(number_label):
#         cd_list.extend(sample_by_value(label, i, int(cd_count/number_label)))
#     ud_list.extend([i for i in range(number_trial) if i not in cd_list])
#     cd_label_list = copy.deepcopy(cd_list)
#     ud_label_list = copy.deepcopy(ud_list)
#     for i in range(len(cd_list)):
#         cd_list[i] =


def pick_one_data(dataset_name, session_id=1, cd_count=4, sub_id=0):
    '''
    @Description: pick one data from session 2 (or from other sessions), 
    @param {type}:
        session_id: int
        cd_count: int (to indicate the number of calibration data)
    @return: 
        832 for session 1, 851 for session 0
        cd_data: array (x*310, x is determined by cd_count)
        ud_data: array ((832-x)*310, the rest of that sub data)
        cd_label: array (x*1)
        ud_label: array ((832-x)*1)              
    '''
    path, allmats = get_allmats_name(dataset_name)
    mat_path = path + "/" + str(session_id+1) + \
        "/" + allmats[session_id][sub_id]
    mat_data = scio.loadmat(mat_path)
    mat_de_data = {key: value for key,
                   value in mat_data.items() if key.startswith('de_LDS')}
    mat_de_data = list(mat_de_data.values())  # 24 * 62 * x * 5
    cd_list = []
    ud_list = []
    number_trial, number_label, labels = get_number_of_label_n_trial(
        dataset_name)
    session_label_one_data = labels[session_id]
    for i in range(number_label):
        # 根据给定的label值从label链表中拿到全部的index后根据数量随机采样
        cd_list.extend(sample_by_value(
            session_label_one_data, i, int(cd_count/number_label)))
    ud_list.extend([i for i in range(number_trial) if i not in cd_list])
    cd_label_list = copy.deepcopy(cd_list)
    ud_label_list = copy.deepcopy(ud_list)
    for i in range(len(cd_list)):
        cd_list[i] = mat_de_data[cd_list[i]]
        cd_label_list[i] = labels[session_id][cd_label_list[i]]
    for i in range(len(ud_list)):
        ud_list[i] = mat_de_data[ud_list[i]]
        ud_label_list[i] = labels[session_id][ud_label_list[i]]

    # reshape
    cd_data, cd_label = reshape_data(cd_list, cd_label_list)
    ud_data, ud_label = reshape_data(ud_list, ud_label_list)

    return cd_data, cd_label, ud_data, ud_label


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


def obtain_pseudo_label(loader, model, number_of_source, c=None):
    start_test = True
    with torch.no_grad():
        all_output = []
        all_fea = []
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            #idx = data[2]
            inputs = inputs.cuda()
            #feas = netB(netF(inputs))
            #outputs = netC(feas)
            feas, outputs = model(inputs, number_of_source)
            if start_test:
                for i in range(number_of_source):
                    all_fea.append(feas[i].float().cpu())
                    all_output.append(outputs[i].float().cpu())
                all_label = labels.float()
                start_test = False
            else:
                for i in range(number_of_source):
                    all_fea[i] = torch.cat((all_fea[i], feas[i].float().cpu()), 0)
                    all_output[i] = torch.cat((all_output[i], outputs[i].float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    pred_label = []
    for i in range(number_of_source):
        all_output[i] = nn.Softmax(dim=1)(all_output[i])
        _, predict = torch.max(all_output[i], 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == torch.squeeze(all_label)).item() / float(all_label.size(0))

        all_fea[i] = torch.cat((all_fea[i], torch.ones(all_fea[i].size(0), 1)), 1)
        all_fea[i] = (all_fea[i].t() / torch.norm(all_fea[i], p=2, dim=1)).t()
        all_fea[i] = all_fea[i].float().cpu().numpy()

        K = all_output[i].size(1)
        aff = all_output[i].float().cpu().numpy()
        initc = aff.transpose().dot(all_fea[i])
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea[i], initc, 'cosine')
        pred_label_i = dd.argmin(axis=1)
        acc = np.sum(pred_label_i == torch.squeeze(all_label).float().numpy()) / len(all_fea[i])

        for round in range(1):
            aff = np.eye(K)[pred_label_i]
            initc = aff.transpose().dot(all_fea[i])
            initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
            dd = cdist(all_fea[i], initc, 'cosine')
            pred_label_i = dd.argmin(axis=1)
            acc = np.sum(pred_label_i == torch.squeeze(all_label).float().numpy()) / len(all_fea[i])

        log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy*100, acc*100)
        #args.out_file.write(log_str + '\n')
        #args.out_file.flush()
        print(log_str+'\n')
        pred_label_i = pred_label_i.astype('int')
        pred_label_i = torch.Tensor(pred_label_i).cuda()
        pred_label.append(pred_label_i)
        #pred_label_i = torch.from_numpy(pred_label_i).cuda()
        #pred_label.append(DataLoader(dataset=PseudolabelDataset(pred_label_i),
        #                                                    batch_size=256,
        #                                                    shuffle=True,
        #                                                    drop_last=True))
    return pred_label


def load_trained_DEAP():
    samples_path_list = './processed_DEAP'
    train_data = None
    label = None
    for path in os.listdir(samples_path_list):
        file_path = os.path.join(samples_path_list, path)
        data = scipy.io.loadmat(file_path)
        #base_DE = data['base_data']
        decomposed_DE = data['data']
        train_sample = None
        train_label = np.zeros(0)
        # flag = 0
        for i in range(len(data['data'])):
            data_temp = decomposed_DE[i]
            label_temp = data['arousal_labels'][i]
            # data_temp = data['ww_eeg' + i]
            # data_temp = scaler.fit_transform(data_temp.transpose(1, 2, 0).reshape(-1, 62)).reshape(-1, 5, 62).transpose(0, 2, 1)
            # data_temp = data_temp.reshape(-1, 32, 5)
            # label_temp = label.reshape(-1)
            #data_seq, label_seq = get_seq_data(data_temp, label_temp)

            if train_sample is None:
                train_sample = data_temp
                train_label = label_temp
            else:
                train_sample = np.vstack((train_sample, data_temp))  # 行堆叠(第一个维度)
                train_label = np.append(train_label, label_temp)
            # flag += 1
        train_sample = np.expand_dims(train_sample, axis=0)
        train_label = np.expand_dims(train_label, axis=0)
        if train_data is None:
            train_data = train_sample
            label = train_label
        else:
            train_data = np.vstack((train_data, train_sample))
            label = np.vstack((label, train_label))

    #train_data = torch.Tensor(train_data).permute(0, 1, 3, 2)
    #label = torch.LongTensor(label)  # y就是标签
    # train_sample = shuffle(train_sample, random_state=2022)   #random_state=2022为随机数种子
    # train_label = shuffle(train_label, random_state=2022)
    # normalize tensor
    #train_sample = normalize(train_sample)
    #pass
    #train_data = train_data.reshape(32,4800,-1)
    train_data = np.expand_dims(train_data, axis=0)
    label = np.expand_dims(label, axis=0)
    train_data = train_data.reshape(1,32,2400,-1)
    #label = label.reshape(1,32,2400,-1)
    return train_data, label


def normalize(features, select_dim=0):
    features_min, _ = torch.min(features, dim=select_dim)
    features_max, _ = torch.max(features, dim=select_dim)
    features_min = features_min.unsqueeze(select_dim)
    features_max = features_max.unsqueeze(select_dim)
    return (features - features_min)/(features_max - features_min)


def load_trained_DREAMER():
    samples_path_list = './processed_DREAMER'
    train_data = None
    label = None
    for path in os.listdir(samples_path_list):
        file_path = os.path.join(samples_path_list, path)
        data = scipy.io.loadmat(file_path)
        decomposed_DE = data['data']
        index = list(map(int, data['index'][0]))
        flag = 0
        train_sample = None
        train_label = np.zeros(0)
        for i in range(len(index)):
            data_temp = decomposed_DE[flag: flag + index[i]]
            label_temp = data['valence_labels'][0][flag: flag + index[i]]
            # label_temp = data['arousal_labels'][0][0: index[i]]
            # label_temp = data['dominance_labels'][0][0: index[i]]
            flag = flag + index[i]

            #data_seq, label_seq = get_seq_data(data_temp, label_temp)

            if train_sample is None:
                train_sample = data_temp
                train_label = label_temp
            else:
                train_sample = np.vstack((train_sample, data_temp))  # 行堆叠(第一个维度)
                train_label = np.append(train_label, label_temp)
            # flag += 1
        train_sample = np.expand_dims(train_sample, axis=0)
        train_label = np.expand_dims(train_label, axis=0)
        if train_data is None:
            train_data = train_sample
            label = train_label
        else:
            train_data = np.vstack((train_data, train_sample))
            label = np.vstack((label, train_label))
    #train_sample = torch.Tensor(train_sample)
    #train_label = torch.LongTensor(train_label)  # y就是标签
    ## train_sample = shuffle(train_sample, random_state=2022)   #random_state=2022为随机数种子
    ## train_label = shuffle(train_label, random_state=2022)
    ## normalize tensor
    #train_sample = normalize(train_sample)
    train_data = np.expand_dims(train_data, axis=0)
    label = np.expand_dims(label, axis=0)
    train_data = train_data.reshape(1,23,3728,-1)
    #label = label.reshape(1,23,3728,-1)
    return train_data, label


def print_confusion_matrix(ground_truth,predictions,data_name,x,y,z):
    plt.switch_backend('agg')
    if data_name == 'seed3':
        #classes=[0,1,2]
        classes=['S','N','H']
    elif data_name == 'seed4':
        #classes=[0,1,2,3]
        classes=['N','S','F','H']
    save_path = './confusion_matrix_SEED4/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    ground_truth = ground_truth.data.squeeze()
    ground_truth,predictions=ground_truth.cpu(),predictions.cpu()
    cm = confusion_matrix(ground_truth, predictions)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.set_printoptions(formatter={'float':'{:0.3f}'.format})
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=classes)
    cmap = plt.cm.Blues
    fig, ax = plt.subplots()
    #disp.set(font_scale=1.5)
    disp.plot(ax=ax, cmap=cmap)
    plt.rcParams.update({'font.size': 16})
    #disp.plot(cmap=cmap)
    if z == 14: 
        image_name = 'sub'+ str(x) + str(y) + '.pdf'
        disp.ax_.set_title('Cross-subject on SEED-IV')
    elif z == 2:
        image_name = 'ses'+ str(x) + str(y) + '.pdf'
        disp.ax_.set_title('Cross-session on SEED-IV')
    #disp.ax_.set_xlabel('Predicted label', fontsize=10)  # 设置x轴标签的字体大小
    #disp.ax_.set_ylabel('True label', fontsize=10)  # 设置y轴标签的字体大小
    #disp.ax_.set_title('Confusion Matrix')  # 设置标题的字体大小
    #disp.ax_.tick_params(axis='both', which='major', labelsize=10)  # 设置刻度标签字体大小
    fig.savefig(os.path.join(save_path, image_name), overwrite=True, bbox_inches="tight")
    plt.close(fig)
    #plt.show()


def t_SNE(model, source_data, target_data, x,y,N):
    save_path = './t_SNE/'
    if N == 14: 
        image_name = 'sub'+ str(x) + str(y) + '.pdf'
    elif N == 2:
        image_name = 'ses'+ str(x) + str(y) + '.pdf'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with torch.no_grad():
        source_data_samples = []
        source_label = []
        source_features = []
        for i in range(len(source_data)):
            for j, (data, label, _) in enumerate(source_data[i]):
                #_, preds = self.model(data, len(self.source_loaders))
                data_100 = data[:100,:].cuda()

                feature, _ = model(data_100,N)
                feature = feature[i]
                source_features.append(feature)
                source_data_samples.append(data)
                source_label.append(label)
                if j == 0:  # 取出10个批次后停止
                    break
        for j, (data, target_label, _) in enumerate(target_data):
            target_data_sample = data[:100,:].cuda()
            target_features, _ = model(target_data_sample, N)
            if j == 0:  # 取出10个批次后停止
                break
        tsne = TSNE(init='pca', random_state=0)
        for i in range(len(source_features)):
            trans_data = tsne.fit_transform(source_features[i].cpu().data.numpy())
            trans_data = norm_tsne(trans_data)
            plt.scatter(trans_data[:,0], trans_data[:,1], marker=".")
            #trans_t_data = tsne.fit_transform(target_features[i].cpu().data.numpy())
            #trans_t_data = norm_tsne(trans_t_data)
            #plt.scatter(trans_t_data[:,0], trans_t_data[:,1], marker="x", c='black', alpha=0.3)
        trans_t_data = tsne.fit_transform(target_features[0].cpu().data.numpy())
        trans_t_data = norm_tsne(trans_t_data)
        plt.scatter(trans_t_data[:,0], trans_t_data[:,1], marker="x", c='black', alpha=0.5)
        plt.savefig(os.path.join(save_path, image_name), overwrite=True)
        plt.close()
        #plt.show()
        

def norm_tsne(data):
    min, max = np.min(data, 0), np.max(data, 0)
    return data / (max-min)


def sample_from_matrix(data, number):
    number_of_rows = data.shape[0]
    indices = np.random.choice(number_of_rows, number)
    return data[indices, :]


def t_SNE_NEW(model, source_data, target_data, x,y,N):
    save_path = './t_SNE_SEED3/'
    if N == 14: 
        image_name = 'sub'+ str(x) + str(y) + '.pdf'
    elif N == 2:
        image_name = 'ses'+ str(x) + str(y) + '.pdf'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    colors = ['b', 'g', 'r']
    with torch.no_grad():
        source_data_samples = []
        source_label = []
        source_features = []
        for i in range(len(source_data)):
            for j, (data, label, _) in enumerate(source_data[i]):
                #_, preds = self.model(data, len(self.source_loaders))
                data_100 = data[:100,:].cuda()

                feature, _ = model(data_100,N)
                feature = feature[i]
                source_features.append(feature)
                source_data_samples.append(data)
                source_label.append(label)
                if j == 0:  # 取出10个批次后停止
                    break
        for j, (data, target_label, _) in enumerate(target_data):
            target_data_sample = data[:100,:].cuda()
            target_features, _ = model(target_data_sample, N)
            if j == 0:  # 取出10个批次后停止
                break
        target_label = [target_label] * N
        #target_label = np.vstack(target_label)
        tsne = TSNE(init='pca', random_state=0)
        for i in range(len(source_features)):
            source_data = source_features[i].cpu().data.numpy()
            source_label = int(source_label[i].cpu().data.numpy().item())  # 假设每个source_features[i]都对应一个label

            target_data = target_features[i].cpu().data.numpy()
            target_label = target_label[i].cpu().data.numpy()  # 假设每个target_features[i]都对应一个label

            trans_data = tsne.fit_transform(source_data)
            trans_data = norm_tsne(trans_data)
            plt.scatter(trans_data[:, 0], trans_data[:, 1], marker=".", c=colors[source_label])

            trans_t_data = tsne.fit_transform(target_data)
            trans_t_data = norm_tsne(trans_t_data)
            plt.scatter(trans_t_data[:, 0], trans_t_data[:, 1], marker="x", c='black', alpha=0.3)
        class_labels = list(range(len(colors)))
        for label in class_labels:
            plt.scatter([], [], c=colors[label], label=f"Class {label}")
        plt.savefig(os.path.join(save_path, image_name), overwrite=True)
        plt.close()

if __name__ == "__main__":
    pass