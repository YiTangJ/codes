import numpy as np
import scipy.sparse as sp
from random import shuffle
import torch
from tqdm import tqdm
import os

def load_data_twse(path="../data/citeseer/", dataset="citeseer",datatype='restaurant',train='train'):
    '''
        从train.py传过来的
        dataset = 'example'
        path = '../data/example/'
    '''
    # print('Loading {} dataset...'.format(dataset))
    features_block = False  # concatenate the feature spaces or not
    '''
        features_block = True：采用GCN的简单方式是直接将不同类型节点t的特征空间连接在一起，构建一个新的大特征空间————忽略了不同信息类型的异构性
        features_block = False：用本文提出的异构图卷积算法，考虑不同的信息类型
    '''

    # type_list = ['word', 'sentiment','text_ws']
    type_list = ['text_ws','word', 'sentiment','entity']
    type_have_label = 'text_ws'

    features_list = []
    idx_map_list = []
    '''
    idx2type : {'text': set(), 'topic': set(), 'entity': set()}
    '''
    idx2type = {t: set() for t in type_list}

    # '''ws加载text的标签'''
    # print('Loading {} label...'.format(type_have_label))
    # indexes, features, labels = [], [], []
    # with open("{}{}_{}.content.{}_{}".format(path, dataset, datatype, type_have_label, train)) as f:
    #     for line in tqdm(f):
    #         cache = line.strip().split('\t')
    #         labels.append(np.array([cache[-1]], dtype=str))
    #     labels = np.stack(labels)
    #     labels = encode_onehot(labels)
    #     Labels = torch.LongTensor(labels)
    #     print("label matrix shape: {}".format(Labels.shape))  # [40,8]

    for type_name in type_list:
        # print('Loading {} content...'.format(type_name))
        # print('path:', path)
        # print('dataset:', dataset)
        # print('type_name:', type_name)
        indexes, features, labels = [], [], []
        with open("{}{}_{}.content.{}_{}".format(path, dataset,datatype, type_name,train)) as f:
            for line in tqdm(f):
                cache = line.strip().split('\t')
                indexes.append(np.array(cache[0], dtype=int))
                features.append(np.array(cache[1:-1], dtype=np.float32))
                labels.append(np.array([cache[-1]], dtype=str) )
            features = np.stack(features)
            features = normalize(features)
            if not features_block:
                '''
                    直接连接不同类型的特征空间——不是本文的异构图卷积
                    模型叫GCN-HIN
                '''
                features = torch.FloatTensor(np.array(features))
                features = dense_tensor_to_sparse(features)

            features_list.append(features)

        '''
            如果是text类型
        '''
        if type_name == type_have_label:
            labels = np.stack(labels)
            '''
                干什么用的？？？？
            '''
            labels = encode_onehot(labels)
            Labels = torch.LongTensor(labels)
            # print("label matrix shape: {}".format(Labels.shape)) # [40,8]

        idx = np.stack(indexes)
        # idx:[ 98  27 131  31 137  59  29 121  86  53  81  22  83  97  32 104  12  79,  38 105   1 116  35  17   2  24  60  30 141 128  26  90 119 114 122  51,  74  82  88  67]
        for i in idx:
            idx2type[type_name].add(i)
        # idx2type:{'text': {128, 1, 2, 131, 137, 12, 141, 17, 22, 24, 26, 27, 29, 30, 31, 32, 35, 38, 51, 53, 59, 60, 67, 74, 79, 81, 82, 83, 86, 88, 90, 97, 98, 104, 105, 114, 116, 119, 121, 122}, 'topic': set(), 'entity': set()}
        idx_map = {j: i for i, j in enumerate(idx)}
        # idx_map:{98: 0, 27: 1, 131: 2, 31: 3, 137: 4, 59: 5, 29: 6, 121: 7, 86: 8, 53: 9, 81: 10, 22: 11, 83: 12, 97: 13, 32: 14, 104: 15, 12: 16, 79: 17, 38: 18, 105: 19, 1: 20, 116: 21, 35: 22, 17: 23, 2: 24, 24: 25, 60: 26, 30: 27, 141: 28, 128: 29, 26: 30, 90: 31, 119: 32, 114: 33, 122: 34, 51: 35, 74: 36, 82: 37, 88: 38, 67: 39}
        idx_map_list.append(idx_map)
        # idx_map_list:[{98: 0, 27: 1, 131: 2, 31: 3, 137: 4, 59: 5, 29: 6, 121: 7, 86: 8, 53: 9, 81: 10, 22: 11, 83: 12, 97: 13, 32: 14, 104: 15, 12: 16, 79: 17, 38: 18, 105: 19, 1: 20, 116: 21, 35: 22, 17: 23, 2: 24, 24: 25, 60: 26, 30: 27, 141: 28, 128: 29, 26: 30, 90: 31, 119: 32, 114: 33, 122: 34, 51: 35, 74: 36, 82: 37, 88: 38, 67: 39}, {69: 0, 108: 1, 33: 2, 65: 3, 80: 4, 92: 5, 25: 6, 58: 7, 109: 8, 132: 9, 64: 10, 15: 11}, {110: 0, 16: 1, 45: 2, 68: 3, 52: 4, 115: 5, 140: 6, 13: 7, 103: 8, 41: 9, 66: 10, 57: 11, 124: 12, 139: 13, 19: 14, 136: 15, 3: 16, 130: 17, 46: 18, 100: 19, 4: 20, 8: 21, 37: 22, 123: 23, 76: 24, 142: 25, 138: 26, 20: 27, 133: 28, 85: 29, 49: 30, 107: 31, 120: 32, 39: 33, 89: 34, 36: 35, 42: 36, 6: 37, 71: 38, 101: 39, 126: 40, 21: 41, 63: 42, 5: 43, 77: 44, 112: 45, 43: 46, 113: 47, 70: 48, 11: 49, 111: 50, 40: 51, 56: 52, 34: 53, 73: 54, 106: 55, 18: 56, 23: 57, 75: 58, 135: 59, 54: 60, 0: 61, 78: 62, 55: 63, 134: 64, 44: 65, 102: 66, 95: 67, 125: 68, 87: 69, 96: 70, 72: 71, 47: 72, 62: 73, 28: 74, 84: 75, 10: 76, 117: 77, 118: 78, 7: 79, 61: 80, 48: 81, 127: 82, 129: 83, 99: 84, 94: 85, 14: 86, 93: 87, 9: 88, 144: 89, 91: 90, 50: 91, 143: 92}]
        # print('done.')

    len_list = [len(idx2type[t]) for t in type_list] # [40, 12, 93]对应text、topic、entity行数
    type2len = {t: len(idx2type[t]) for t in type_list} # {'text': 40, 'topic': 12, 'entity': 93}
    len_all = sum(len_list) # 145=40+12+93
    if features_block:
        '''
            本文的异构图卷积方法
            我们提出了异构图卷积算法，该算法考虑了不同类型信息的差异，并将它们用各自的变换矩阵投影到一个隐式公共空间中
        '''
        flen = [i.shape[1] for i in features_list]
        features = sp.lil_matrix(np.zeros((len_all, sum(flen))), dtype=np.float32)
        bias = 0
        for i_l in range(len(len_list)):
            features[bias:bias+len_list[i_l], :flen[i_l]] = features_list[i_l]
            features_list[i_l] = features[bias:bias+len_list[i_l], :]
            bias += len_list[i_l]
        for fi in range(len(features_list)):
            features_list[fi] = torch.FloatTensor(np.array(features_list[fi].todense()))
            features_list[fi] = dense_tensor_to_sparse(features_list[fi])

    '''
        没看懂？？？？？
    '''
    # print('Building graph...')
    # adj_list:[[None, None, None], [None, None, None], [None, None, None]]
    adj_list = [[None for _ in range(len(type_list))] for __ in range(len(type_list))]
    # build graph
    '''
        "{}{}.cites".format(path, dataset)
        是../data/example/example.cites
    '''
    # 读取../data/example/example.cites文件
    # shape[531,2]
    edges_unordered = np.genfromtxt("{}{}_{}.cites_ws_{}".format(path, dataset,datatype,train),dtype=np.int32)
    # adj_all:[145,145] lil_matrix:[list([]) list([]) list([]) list([]) list([]) list([]) list([]) list([]), list([]) list([]) list([]) list([]) list([]) list([]) list([]) list([]), list([]) list([]) list([]) list([]) list([]) list([]) list([]) list([]), list([]) list([]) list([]) list([])
    adj_all = sp.lil_matrix(np.zeros((len_all, len_all)), dtype=np.float32)

    for i1 in range(len(type_list)):
        for i2 in range(len(type_list)):
            t1, t2 = type_list[i1], type_list[i2]
            if i1 == i2:
                edges = []
                for edge in edges_unordered:
                    if (edge[0] in idx2type[t1] and edge[1] in idx2type[t2]):
                        edges.append([idx_map_list[i1].get(edge[0]), idx_map_list[i2].get(edge[1])]) # text-text加了40条边;topic-topic加了12条；entity-entity边-387条
                edges = np.array(edges) # text-text[40,2]
                if len(edges) > 0:
                    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), # np.ones(n)生成array：np.ones(5)---array([ 1.,  1.,  1.,  1.,  1.])
                                        shape=(type2len[t1], type2len[t2]), dtype=np.float32) # text-text-shape[40,40] 生成稀疏矩阵；topic-topic[12,12]；entity-entity[93,93]
                else:
                    adj = sp.coo_matrix((type2len[t1], type2len[t2]), dtype=np.float32)
                adj_all[sum(len_list[:i1]): sum(len_list[:i1 + 1]), # [145,145]:[list([1.0]) list([1.0]) list([1.0]) list([1.0]) list([1.0]) list([1.0]), list([1.0]) list([1.0]) list([1.0]) list([1.0]) list([1.0]) list([1.0]), list([1.0]) list([1.0]) list([1.0]) list([1.0]) list([1.0]) list([1.0]), list([1.0]) list([1.0]) list([1.0]) list([1.0]) list([1.0]) list([1.0]), list([1.0]) list([1.0]) list([1.0]) list([1.0]) list([1.0]) list([1.0]), list([1.0]) list([1.0]) list([1.0]) list([1.0]) list([1.0]) list([1.0]), list([1.0]) list([1.0]) list([1.0]) list([1.0]) list([]) list([]), list([]) list([]) list([]) list([]) list([]) list([]) list([]) list([]), list([]) list([]) list([]) list([]) list([]) list([]) list([]) list([]), list([]) list([]) list([]) list([]) list([]) list([]) list([]) list([]), list([]) list([]) list([]) list([]) list([]) list([]) list([]) list([]), list([]) list([]) list([]) list([]) list([]) list([]) list([]) list([]), list([]) list([]) list([]) list([]) list([]) list([]) list([]) list([]), list([]) list([]) list([]) list([]) list([]) list([]) li...
                        sum(len_list[:i2]): sum(len_list[:i2 + 1])] = adj.tolil() # 最后的值[list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), li...

            elif i1 < i2:
                edges = []
                for edge in edges_unordered: # 生成text-topic边-80条, 生成text-entity边-12条, 生成topic-entity边-0条
                    if (edge[0] in idx2type[t1] and edge[1] in idx2type[t2]):
                        edges.append([idx_map_list[i1].get(edge[0]), idx_map_list[i2].get(edge[1])])
                    elif (edge[1] in idx2type[t1] and edge[0] in idx2type[t2]):
                        edges.append([idx_map_list[i1].get(edge[1]), idx_map_list[i2].get(edge[0])])
                edges = np.array(edges)
                if len(edges) > 0:
                    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), # text-topic[40,12]，text-entity[40,93]
                                        shape=(type2len[t1], type2len[t2]), dtype=np.float32)
                else:
                    adj = sp.coo_matrix((type2len[t1], type2len[t2]), dtype=np.float32)

                adj_all[
                    sum(len_list[:i1]): sum(len_list[:i1 + 1]),
                    sum(len_list[:i2]): sum(len_list[:i2 + 1])] = adj.tolil()
                adj_all[
                    sum(len_list[:i2]): sum(len_list[:i2 + 1]),
                    sum(len_list[:i1]): sum(len_list[:i1 + 1])] = adj.T.tolil() #[145,145]:[list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]) list([1.0, text-entity:[145,145]

    adj_all = adj_all + adj_all.T.multiply(adj_all.T > adj_all) - adj_all.multiply(adj_all.T > adj_all)
    adj_all = normalize_adj(adj_all + sp.eye(adj_all.shape[0]))

    for i1 in range(len(type_list)):
        for i2 in range(len(type_list)): #[3,3]
            adj_list[i1][i2] = sparse_mx_to_torch_sparse_tensor(
                adj_all[sum(len_list[:i1]): sum(len_list[:i1 + 1]),
                        sum(len_list[:i2]): sum(len_list[:i2 + 1])]
            )

    # print("Num of edges: {}".format(len( adj_all.nonzero()[0] ))) # 917
    # 生成text的idx_map_list__train
    text_idx_map_list_train = []
    text_indexes_train = []
    with open("{}{}_{}.content.text_ws_train".format(path, dataset,datatype)) as f:
        for line in tqdm(f):
            cache = line.strip().split('\t')
            text_indexes_train.append(np.array(cache[0], dtype=int))
    text_idx = np.stack(text_indexes_train)  # idx里面是text类型中的节点在mapindex中的索引
    text_idx_map = {j: i for i, j in enumerate(text_idx)}  # 里面是{text中的节点在mapindex中的索引：idx_map中的索引}
    text_idx_map_list_train.append(text_idx_map)  # 里面一共有text节点类型个{}=[{}],{}={text类型中的节点的点在mapindex中的索引：idx_map中的索引}
    # print('text_idx done.')
    # 生成text的idx_map_list___test
    text_idx_map_list_test = []
    text_indexes_test = []
    with open("{}{}_{}.content.text_ws_test".format(path, dataset, datatype)) as f:
        for line in tqdm(f):
            cache = line.strip().split('\t')
            text_indexes_test.append(np.array(cache[0], dtype=int))
    text_idx = np.stack(text_indexes_test)  # idx里面是text类型中的节点在mapindex中的索引
    text_idx_map = {j: i for i, j in enumerate(text_idx)}  # 里面是{text中的节点在mapindex中的索引：idx_map中的索引}
    text_idx_map_list_test.append(text_idx_map)  # 里面一共有text节点类型个{}=[{}],{}={text类型中的节点的点在mapindex中的索引：idx_map中的索引}
    # print('text_idx done.')

    idx_train, idx_test = load_divide_idx(path, text_idx_map_list_train[0],text_idx_map_list_test[0],Hete='ws')
    # idx_train, idx_test = load_divide_idx(path, idx_map_list[0],Hete='ws')
    # return adj_list, features_list, Labels, idx_train, idx_test, idx_map_list[0]
    return adj_list, features_list, Labels, idx_train, idx_test, text_idx_map_list_train[0], text_idx_map_list_test[0]
'''
    model文件夹下的data文件夹中读取
    ../data/citeseer/citeseer/读取数据entity、text、topic
'''
def load_data_ws(path="../data/citeseer/", dataset="citeseer",datatype='restaurant',train='train'):
    '''
        从train.py传过来的
        dataset = 'example'
        path = '../data/example/'
    '''
    # print('Loading {} dataset...'.format(dataset))
    features_block = False  # concatenate the feature spaces or not
    '''
        features_block = True：采用GCN的简单方式是直接将不同类型节点t的特征空间连接在一起，构建一个新的大特征空间————忽略了不同信息类型的异构性
        features_block = False：用本文提出的异构图卷积算法，考虑不同的信息类型
    '''

    # type_list = ['word', 'sentiment','text_ws']
    type_list = ['text_ws','word', 'sentiment']
    type_have_label = 'text_ws'

    features_list = []
    idx_map_list = []
    '''
    idx2type : {'text': set(), 'topic': set(), 'entity': set()}
    '''
    idx2type = {t: set() for t in type_list}

    # '''ws加载text的标签'''
    # print('Loading {} label...'.format(type_have_label))
    # indexes, features, labels = [], [], []
    # with open("{}{}_{}.content.{}_{}".format(path, dataset, datatype, type_have_label, train)) as f:
    #     for line in tqdm(f):
    #         cache = line.strip().split('\t')
    #         labels.append(np.array([cache[-1]], dtype=str))
    #     labels = np.stack(labels)
    #     labels = encode_onehot(labels)
    #     Labels = torch.LongTensor(labels)
    #     print("label matrix shape: {}".format(Labels.shape))  # [40,8]

    for type_name in type_list:
        # print('Loading {} content...'.format(type_name))
        # print('path:', path)
        # print('dataset:', dataset)
        # print('type_name:', type_name)
        indexes, features, labels = [], [], []
        with open("{}{}_{}.content.{}_{}".format(path, dataset,datatype, type_name,train)) as f:
            for line in tqdm(f):
                cache = line.strip().split('\t')
                indexes.append(np.array(cache[0], dtype=int))
                features.append(np.array(cache[1:-1], dtype=np.float32))
                labels.append(np.array([cache[-1]], dtype=str) )
            features = np.stack(features)
            features = normalize(features)
            if not features_block:
                '''
                    直接连接不同类型的特征空间——不是本文的异构图卷积
                    模型叫GCN-HIN
                '''
                features = torch.FloatTensor(np.array(features))
                features = dense_tensor_to_sparse(features)

            features_list.append(features)

        '''
            如果是text类型
        '''
        if type_name == type_have_label:
            labels = np.stack(labels)
            '''
                干什么用的？？？？
            '''
            labels = encode_onehot(labels)
            Labels = torch.LongTensor(labels)
            # print("label matrix shape: {}".format(Labels.shape)) # [40,8]

        idx = np.stack(indexes)
        # idx:[ 98  27 131  31 137  59  29 121  86  53  81  22  83  97  32 104  12  79,  38 105   1 116  35  17   2  24  60  30 141 128  26  90 119 114 122  51,  74  82  88  67]
        for i in idx:
            idx2type[type_name].add(i)
        # idx2type:{'text': {128, 1, 2, 131, 137, 12, 141, 17, 22, 24, 26, 27, 29, 30, 31, 32, 35, 38, 51, 53, 59, 60, 67, 74, 79, 81, 82, 83, 86, 88, 90, 97, 98, 104, 105, 114, 116, 119, 121, 122}, 'topic': set(), 'entity': set()}
        idx_map = {j: i for i, j in enumerate(idx)}
        # idx_map:{98: 0, 27: 1, 131: 2, 31: 3, 137: 4, 59: 5, 29: 6, 121: 7, 86: 8, 53: 9, 81: 10, 22: 11, 83: 12, 97: 13, 32: 14, 104: 15, 12: 16, 79: 17, 38: 18, 105: 19, 1: 20, 116: 21, 35: 22, 17: 23, 2: 24, 24: 25, 60: 26, 30: 27, 141: 28, 128: 29, 26: 30, 90: 31, 119: 32, 114: 33, 122: 34, 51: 35, 74: 36, 82: 37, 88: 38, 67: 39}
        idx_map_list.append(idx_map)
        # idx_map_list:[{98: 0, 27: 1, 131: 2, 31: 3, 137: 4, 59: 5, 29: 6, 121: 7, 86: 8, 53: 9, 81: 10, 22: 11, 83: 12, 97: 13, 32: 14, 104: 15, 12: 16, 79: 17, 38: 18, 105: 19, 1: 20, 116: 21, 35: 22, 17: 23, 2: 24, 24: 25, 60: 26, 30: 27, 141: 28, 128: 29, 26: 30, 90: 31, 119: 32, 114: 33, 122: 34, 51: 35, 74: 36, 82: 37, 88: 38, 67: 39}, {69: 0, 108: 1, 33: 2, 65: 3, 80: 4, 92: 5, 25: 6, 58: 7, 109: 8, 132: 9, 64: 10, 15: 11}, {110: 0, 16: 1, 45: 2, 68: 3, 52: 4, 115: 5, 140: 6, 13: 7, 103: 8, 41: 9, 66: 10, 57: 11, 124: 12, 139: 13, 19: 14, 136: 15, 3: 16, 130: 17, 46: 18, 100: 19, 4: 20, 8: 21, 37: 22, 123: 23, 76: 24, 142: 25, 138: 26, 20: 27, 133: 28, 85: 29, 49: 30, 107: 31, 120: 32, 39: 33, 89: 34, 36: 35, 42: 36, 6: 37, 71: 38, 101: 39, 126: 40, 21: 41, 63: 42, 5: 43, 77: 44, 112: 45, 43: 46, 113: 47, 70: 48, 11: 49, 111: 50, 40: 51, 56: 52, 34: 53, 73: 54, 106: 55, 18: 56, 23: 57, 75: 58, 135: 59, 54: 60, 0: 61, 78: 62, 55: 63, 134: 64, 44: 65, 102: 66, 95: 67, 125: 68, 87: 69, 96: 70, 72: 71, 47: 72, 62: 73, 28: 74, 84: 75, 10: 76, 117: 77, 118: 78, 7: 79, 61: 80, 48: 81, 127: 82, 129: 83, 99: 84, 94: 85, 14: 86, 93: 87, 9: 88, 144: 89, 91: 90, 50: 91, 143: 92}]
        # print('done.')

    len_list = [len(idx2type[t]) for t in type_list] # [40, 12, 93]对应text、topic、entity行数
    type2len = {t: len(idx2type[t]) for t in type_list} # {'text': 40, 'topic': 12, 'entity': 93}
    len_all = sum(len_list) # 145=40+12+93
    if features_block:
        '''
            本文的异构图卷积方法
            我们提出了异构图卷积算法，该算法考虑了不同类型信息的差异，并将它们用各自的变换矩阵投影到一个隐式公共空间中
        '''
        flen = [i.shape[1] for i in features_list]
        features = sp.lil_matrix(np.zeros((len_all, sum(flen))), dtype=np.float32)
        bias = 0
        for i_l in range(len(len_list)):
            features[bias:bias+len_list[i_l], :flen[i_l]] = features_list[i_l]
            features_list[i_l] = features[bias:bias+len_list[i_l], :]
            bias += len_list[i_l]
        for fi in range(len(features_list)):
            features_list[fi] = torch.FloatTensor(np.array(features_list[fi].todense()))
            features_list[fi] = dense_tensor_to_sparse(features_list[fi])

    '''
        没看懂？？？？？
    '''
    # print('Building graph...')
    # adj_list:[[None, None, None], [None, None, None], [None, None, None]]
    adj_list = [[None for _ in range(len(type_list))] for __ in range(len(type_list))]
    # build graph
    '''
        "{}{}.cites".format(path, dataset)
        是../data/example/example.cites
    '''
    # 读取../data/example/example.cites文件
    # shape[531,2]
    edges_unordered = np.genfromtxt("{}{}_{}.cites_ws_{}".format(path, dataset,datatype,train),dtype=np.int32)
    # adj_all:[145,145] lil_matrix:[list([]) list([]) list([]) list([]) list([]) list([]) list([]) list([]), list([]) list([]) list([]) list([]) list([]) list([]) list([]) list([]), list([]) list([]) list([]) list([]) list([]) list([]) list([]) list([]), list([]) list([]) list([]) list([])
    adj_all = sp.lil_matrix(np.zeros((len_all, len_all)), dtype=np.float32)

    for i1 in range(len(type_list)):
        for i2 in range(len(type_list)):
            t1, t2 = type_list[i1], type_list[i2]
            if i1 == i2:
                edges = []
                for edge in edges_unordered:
                    if (edge[0] in idx2type[t1] and edge[1] in idx2type[t2]):
                        edges.append([idx_map_list[i1].get(edge[0]), idx_map_list[i2].get(edge[1])]) # text-text加了40条边;topic-topic加了12条；entity-entity边-387条
                edges = np.array(edges) # text-text[40,2]
                if len(edges) > 0:
                    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), # np.ones(n)生成array：np.ones(5)---array([ 1.,  1.,  1.,  1.,  1.])
                                        shape=(type2len[t1], type2len[t2]), dtype=np.float32) # text-text-shape[40,40] 生成稀疏矩阵；topic-topic[12,12]；entity-entity[93,93]
                else:
                    adj = sp.coo_matrix((type2len[t1], type2len[t2]), dtype=np.float32)
                adj_all[sum(len_list[:i1]): sum(len_list[:i1 + 1]), # [145,145]:[list([1.0]) list([1.0]) list([1.0]) list([1.0]) list([1.0]) list([1.0]), list([1.0]) list([1.0]) list([1.0]) list([1.0]) list([1.0]) list([1.0]), list([1.0]) list([1.0]) list([1.0]) list([1.0]) list([1.0]) list([1.0]), list([1.0]) list([1.0]) list([1.0]) list([1.0]) list([1.0]) list([1.0]), list([1.0]) list([1.0]) list([1.0]) list([1.0]) list([1.0]) list([1.0]), list([1.0]) list([1.0]) list([1.0]) list([1.0]) list([1.0]) list([1.0]), list([1.0]) list([1.0]) list([1.0]) list([1.0]) list([]) list([]), list([]) list([]) list([]) list([]) list([]) list([]) list([]) list([]), list([]) list([]) list([]) list([]) list([]) list([]) list([]) list([]), list([]) list([]) list([]) list([]) list([]) list([]) list([]) list([]), list([]) list([]) list([]) list([]) list([]) list([]) list([]) list([]), list([]) list([]) list([]) list([]) list([]) list([]) list([]) list([]), list([]) list([]) list([]) list([]) list([]) list([]) list([]) list([]), list([]) list([]) list([]) list([]) list([]) list([]) li...
                        sum(len_list[:i2]): sum(len_list[:i2 + 1])] = adj.tolil() # 最后的值[list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), li...

            elif i1 < i2:
                edges = []
                for edge in edges_unordered: # 生成text-topic边-80条, 生成text-entity边-12条, 生成topic-entity边-0条
                    if (edge[0] in idx2type[t1] and edge[1] in idx2type[t2]):
                        edges.append([idx_map_list[i1].get(edge[0]), idx_map_list[i2].get(edge[1])])
                    elif (edge[1] in idx2type[t1] and edge[0] in idx2type[t2]):
                        edges.append([idx_map_list[i1].get(edge[1]), idx_map_list[i2].get(edge[0])])
                edges = np.array(edges)
                if len(edges) > 0:
                    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), # text-topic[40,12]，text-entity[40,93]
                                        shape=(type2len[t1], type2len[t2]), dtype=np.float32)
                else:
                    adj = sp.coo_matrix((type2len[t1], type2len[t2]), dtype=np.float32)

                adj_all[
                    sum(len_list[:i1]): sum(len_list[:i1 + 1]),
                    sum(len_list[:i2]): sum(len_list[:i2 + 1])] = adj.tolil()
                adj_all[
                    sum(len_list[:i2]): sum(len_list[:i2 + 1]),
                    sum(len_list[:i1]): sum(len_list[:i1 + 1])] = adj.T.tolil() #[145,145]:[list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]) list([1.0, text-entity:[145,145]

    adj_all = adj_all + adj_all.T.multiply(adj_all.T > adj_all) - adj_all.multiply(adj_all.T > adj_all)
    adj_all = normalize_adj(adj_all + sp.eye(adj_all.shape[0]))

    for i1 in range(len(type_list)):
        for i2 in range(len(type_list)): #[3,3]
            adj_list[i1][i2] = sparse_mx_to_torch_sparse_tensor(
                adj_all[sum(len_list[:i1]): sum(len_list[:i1 + 1]),
                        sum(len_list[:i2]): sum(len_list[:i2 + 1])]
            )

    # print("Num of edges: {}".format(len( adj_all.nonzero()[0] ))) # 917
    # 生成text的idx_map_list__train
    text_idx_map_list_train = []
    text_indexes_train = []
    with open("{}{}_{}.content.text_ws_train".format(path, dataset,datatype)) as f:
        for line in tqdm(f):
            cache = line.strip().split('\t')
            text_indexes_train.append(np.array(cache[0], dtype=int))
    text_idx = np.stack(text_indexes_train)  # idx里面是text类型中的节点在mapindex中的索引
    text_idx_map = {j: i for i, j in enumerate(text_idx)}  # 里面是{text中的节点在mapindex中的索引：idx_map中的索引}
    text_idx_map_list_train.append(text_idx_map)  # 里面一共有text节点类型个{}=[{}],{}={text类型中的节点的点在mapindex中的索引：idx_map中的索引}
    # print('text_idx done.')
    # 生成text的idx_map_list___test
    text_idx_map_list_test = []
    text_indexes_test = []
    with open("{}{}_{}.content.text_ws_test".format(path, dataset, datatype)) as f:
        for line in tqdm(f):
            cache = line.strip().split('\t')
            text_indexes_test.append(np.array(cache[0], dtype=int))
    text_idx = np.stack(text_indexes_test)  # idx里面是text类型中的节点在mapindex中的索引
    text_idx_map = {j: i for i, j in enumerate(text_idx)}  # 里面是{text中的节点在mapindex中的索引：idx_map中的索引}
    text_idx_map_list_test.append(text_idx_map)  # 里面一共有text节点类型个{}=[{}],{}={text类型中的节点的点在mapindex中的索引：idx_map中的索引}
    # print('text_idx done.')

    idx_train, idx_test = load_divide_idx(path, text_idx_map_list_train[0],text_idx_map_list_test[0],Hete='ws')
    # idx_train, idx_test = load_divide_idx(path, idx_map_list[0],Hete='ws')
    # return adj_list, features_list, Labels, idx_train, idx_test, idx_map_list[0]
    return adj_list, features_list, Labels, idx_train, idx_test, text_idx_map_list_train[0], text_idx_map_list_test[0]


def load_data_et(path="../data/citeseer/", dataset="citeseer", datatype='restaurant', train='train'):
    '''
        从train.py传过来的
        dataset = 'example'
        path = '../data/example/'
    '''
    # print('Loading {} dataset...'.format(dataset))
    features_block = False  # concatenate the feature spaces or not
    '''
        features_block = True：采用GCN的简单方式是直接将不同类型节点t的特征空间连接在一起，构建一个新的大特征空间————忽略了不同信息类型的异构性
        features_block = False：用本文提出的异构图卷积算法，考虑不同的信息类型
    '''

    type_list = ['text', 'entity']
    type_have_label = 'text'

    features_list = []
    idx_map_list = []
    '''
    idx2type : {'text': set(), 'topic': set(), 'entity': set()}
    '''
    idx2type = {t: set() for t in type_list}

    for type_name in type_list:
        # print('Loading {} content...'.format(type_name))
        # print('path:', path)
        # print('dataset:', dataset)
        # print('type_name:', type_name)
        indexes, features, labels = [], [], []
        with open("{}{}_{}.content.{}_{}".format(path, dataset, datatype, type_name, train)) as f:
            for line in tqdm(f):
                cache = line.strip().split('\t')
                indexes.append(np.array(cache[0], dtype=int))
                features.append(np.array(cache[1:-1], dtype=np.float32))
                labels.append(np.array([cache[-1]], dtype=str))
            features = np.stack(features)
            features = normalize(features)
            if not features_block:
                '''
                    直接连接不同类型的特征空间——不是本文的异构图卷积
                    模型叫GCN-HIN
                '''
                features = torch.FloatTensor(np.array(features))
                features = dense_tensor_to_sparse(features)

            features_list.append(features)

        '''
            如果是text类型
        '''
        if type_name == type_have_label:
            labels = np.stack(labels)
            '''
                干什么用的？？？？
            '''
            labels = encode_onehot(labels)
            Labels = torch.LongTensor(labels)
            # print("label matrix shape: {}".format(Labels.shape))  # [40,8]

        idx = np.stack(indexes)
        # idx:[ 98  27 131  31 137  59  29 121  86  53  81  22  83  97  32 104  12  79,  38 105   1 116  35  17   2  24  60  30 141 128  26  90 119 114 122  51,  74  82  88  67]
        for i in idx:
            idx2type[type_name].add(i)
        # idx2type:{'text': {128, 1, 2, 131, 137, 12, 141, 17, 22, 24, 26, 27, 29, 30, 31, 32, 35, 38, 51, 53, 59, 60, 67, 74, 79, 81, 82, 83, 86, 88, 90, 97, 98, 104, 105, 114, 116, 119, 121, 122}, 'topic': set(), 'entity': set()}
        idx_map = {j: i for i, j in enumerate(idx)}
        # idx_map:{98: 0, 27: 1, 131: 2, 31: 3, 137: 4, 59: 5, 29: 6, 121: 7, 86: 8, 53: 9, 81: 10, 22: 11, 83: 12, 97: 13, 32: 14, 104: 15, 12: 16, 79: 17, 38: 18, 105: 19, 1: 20, 116: 21, 35: 22, 17: 23, 2: 24, 24: 25, 60: 26, 30: 27, 141: 28, 128: 29, 26: 30, 90: 31, 119: 32, 114: 33, 122: 34, 51: 35, 74: 36, 82: 37, 88: 38, 67: 39}
        idx_map_list.append(idx_map)
        # idx_map_list:[{98: 0, 27: 1, 131: 2, 31: 3, 137: 4, 59: 5, 29: 6, 121: 7, 86: 8, 53: 9, 81: 10, 22: 11, 83: 12, 97: 13, 32: 14, 104: 15, 12: 16, 79: 17, 38: 18, 105: 19, 1: 20, 116: 21, 35: 22, 17: 23, 2: 24, 24: 25, 60: 26, 30: 27, 141: 28, 128: 29, 26: 30, 90: 31, 119: 32, 114: 33, 122: 34, 51: 35, 74: 36, 82: 37, 88: 38, 67: 39}, {69: 0, 108: 1, 33: 2, 65: 3, 80: 4, 92: 5, 25: 6, 58: 7, 109: 8, 132: 9, 64: 10, 15: 11}, {110: 0, 16: 1, 45: 2, 68: 3, 52: 4, 115: 5, 140: 6, 13: 7, 103: 8, 41: 9, 66: 10, 57: 11, 124: 12, 139: 13, 19: 14, 136: 15, 3: 16, 130: 17, 46: 18, 100: 19, 4: 20, 8: 21, 37: 22, 123: 23, 76: 24, 142: 25, 138: 26, 20: 27, 133: 28, 85: 29, 49: 30, 107: 31, 120: 32, 39: 33, 89: 34, 36: 35, 42: 36, 6: 37, 71: 38, 101: 39, 126: 40, 21: 41, 63: 42, 5: 43, 77: 44, 112: 45, 43: 46, 113: 47, 70: 48, 11: 49, 111: 50, 40: 51, 56: 52, 34: 53, 73: 54, 106: 55, 18: 56, 23: 57, 75: 58, 135: 59, 54: 60, 0: 61, 78: 62, 55: 63, 134: 64, 44: 65, 102: 66, 95: 67, 125: 68, 87: 69, 96: 70, 72: 71, 47: 72, 62: 73, 28: 74, 84: 75, 10: 76, 117: 77, 118: 78, 7: 79, 61: 80, 48: 81, 127: 82, 129: 83, 99: 84, 94: 85, 14: 86, 93: 87, 9: 88, 144: 89, 91: 90, 50: 91, 143: 92}]
        # print('done.')

    len_list = [len(idx2type[t]) for t in type_list]  # [40, 12, 93]对应text、topic、entity行数
    type2len = {t: len(idx2type[t]) for t in type_list}  # {'text': 40, 'topic': 12, 'entity': 93}
    len_all = sum(len_list)  # 145=40+12+93
    if features_block:
        '''
            本文的异构图卷积方法
            我们提出了异构图卷积算法，该算法考虑了不同类型信息的差异，并将它们用各自的变换矩阵投影到一个隐式公共空间中
        '''
        flen = [i.shape[1] for i in features_list]
        features = sp.lil_matrix(np.zeros((len_all, sum(flen))), dtype=np.float32)
        bias = 0
        for i_l in range(len(len_list)):
            features[bias:bias + len_list[i_l], :flen[i_l]] = features_list[i_l]
            features_list[i_l] = features[bias:bias + len_list[i_l], :]
            bias += len_list[i_l]
        for fi in range(len(features_list)):
            features_list[fi] = torch.FloatTensor(np.array(features_list[fi].todense()))
            features_list[fi] = dense_tensor_to_sparse(features_list[fi])

    '''
        没看懂？？？？？
    '''
    print('Building graph...')
    # adj_list:[[None, None, None], [None, None, None], [None, None, None]]
    adj_list = [[None for _ in range(len(type_list))] for __ in range(len(type_list))]
    # build graph
    '''
        "{}{}.cites".format(path, dataset)
        是../data/example/example.cites
    '''
    # 读取../data/example/example.cites文件
    # shape[531,2]
    edges_unordered = np.genfromtxt("{}{}_{}.cites_et_{}".format(path, dataset,datatype, train),
                                    dtype=np.int32)
    # adj_all:[145,145] lil_matrix:[list([]) list([]) list([]) list([]) list([]) list([]) list([]) list([]), list([]) list([]) list([]) list([]) list([]) list([]) list([]) list([]), list([]) list([]) list([]) list([]) list([]) list([]) list([]) list([]), list([]) list([]) list([]) list([])
    adj_all = sp.lil_matrix(np.zeros((len_all, len_all)), dtype=np.float32)

    for i1 in range(len(type_list)):
        for i2 in range(len(type_list)):
            t1, t2 = type_list[i1], type_list[i2]
            if i1 == i2:
                edges = []
                for edge in edges_unordered:
                    if (edge[0] in idx2type[t1] and edge[1] in idx2type[t2]):
                        edges.append([idx_map_list[i1].get(edge[0]), idx_map_list[i2].get(
                            edge[1])])  # text-text加了40条边;topic-topic加了12条；entity-entity边-387条
                edges = np.array(edges)  # text-text[40,2]
                if len(edges) > 0:
                    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                        # np.ones(n)生成array：np.ones(5)---array([ 1.,  1.,  1.,  1.,  1.])
                                        shape=(type2len[t1], type2len[t2]),
                                        dtype=np.float32)  # text-text-shape[40,40] 生成稀疏矩阵；topic-topic[12,12]；entity-entity[93,93]
                else:
                    adj = sp.coo_matrix((type2len[t1], type2len[t2]), dtype=np.float32)
                adj_all[sum(len_list[:i1]): sum(len_list[:i1 + 1]),
                # [145,145]:[list([1.0]) list([1.0]) list([1.0]) list([1.0]) list([1.0]) list([1.0]), list([1.0]) list([1.0]) list([1.0]) list([1.0]) list([1.0]) list([1.0]), list([1.0]) list([1.0]) list([1.0]) list([1.0]) list([1.0]) list([1.0]), list([1.0]) list([1.0]) list([1.0]) list([1.0]) list([1.0]) list([1.0]), list([1.0]) list([1.0]) list([1.0]) list([1.0]) list([1.0]) list([1.0]), list([1.0]) list([1.0]) list([1.0]) list([1.0]) list([1.0]) list([1.0]), list([1.0]) list([1.0]) list([1.0]) list([1.0]) list([]) list([]), list([]) list([]) list([]) list([]) list([]) list([]) list([]) list([]), list([]) list([]) list([]) list([]) list([]) list([]) list([]) list([]), list([]) list([]) list([]) list([]) list([]) list([]) list([]) list([]), list([]) list([]) list([]) list([]) list([]) list([]) list([]) list([]), list([]) list([]) list([]) list([]) list([]) list([]) list([]) list([]), list([]) list([]) list([]) list([]) list([]) list([]) list([]) list([]), list([]) list([]) list([]) list([]) list([]) list([]) li...
                sum(len_list[:i2]): sum(len_list[
                                        :i2 + 1])] = adj.tolil()  # 最后的值[list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), li...

            elif i1 < i2:
                edges = []
                for edge in edges_unordered:  # 生成text-topic边-80条, 生成text-entity边-12条, 生成topic-entity边-0条
                    if (edge[0] in idx2type[t1] and edge[1] in idx2type[t2]):
                        edges.append([idx_map_list[i1].get(edge[0]), idx_map_list[i2].get(edge[1])])
                    elif (edge[1] in idx2type[t1] and edge[0] in idx2type[t2]):
                        edges.append([idx_map_list[i1].get(edge[1]), idx_map_list[i2].get(edge[0])])
                edges = np.array(edges)
                if len(edges) > 0:
                    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                        # text-topic[40,12]，text-entity[40,93]
                                        shape=(type2len[t1], type2len[t2]), dtype=np.float32)
                else:
                    adj = sp.coo_matrix((type2len[t1], type2len[t2]), dtype=np.float32)

                adj_all[
                sum(len_list[:i1]): sum(len_list[:i1 + 1]),
                sum(len_list[:i2]): sum(len_list[:i2 + 1])] = adj.tolil()
                adj_all[
                sum(len_list[:i2]): sum(len_list[:i2 + 1]),
                sum(len_list[:i1]): sum(len_list[
                                        :i1 + 1])] = adj.T.tolil()  # [145,145]:[list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]), list([1.0, 1.0, 1.0]) list([1.0, 1.0, 1.0]) list([1.0, text-entity:[145,145]

    adj_all = adj_all + adj_all.T.multiply(adj_all.T > adj_all) - adj_all.multiply(adj_all.T > adj_all)
    adj_all = normalize_adj(adj_all + sp.eye(adj_all.shape[0]))

    for i1 in range(len(type_list)):
        for i2 in range(len(type_list)):  # [3,3]
            adj_list[i1][i2] = sparse_mx_to_torch_sparse_tensor(
                adj_all[sum(len_list[:i1]): sum(len_list[:i1 + 1]),
                sum(len_list[:i2]): sum(len_list[:i2 + 1])]
            )

    print("Num of edges: {}".format(len(adj_all.nonzero()[0])))  # 917
    # 生成text的idx_map_list__train
    text_idx_map_list_train = []
    text_indexes_train = []
    with open("{}{}_{}.content.text_train".format(path, dataset, datatype)) as f:
        for line in tqdm(f):
            cache = line.strip().split('\t')
            text_indexes_train.append(np.array(cache[0], dtype=int))
    text_idx = np.stack(text_indexes_train)  # idx里面是text类型中的节点在mapindex中的索引
    text_idx_map = {j: i for i, j in enumerate(text_idx)}  # 里面是{text中的节点在mapindex中的索引：idx_map中的索引}
    text_idx_map_list_train.append(text_idx_map)  # 里面一共有text节点类型个{}=[{}],{}={text类型中的节点的点在mapindex中的索引：idx_map中的索引}
    # print('text_idx done.')
    # 生成text的idx_map_list___test
    text_idx_map_list_test = []
    text_indexes_test = []
    with open("{}{}_{}.content.text_test".format(path, dataset, datatype)) as f:
        for line in tqdm(f):
            cache = line.strip().split('\t')
            text_indexes_test.append(np.array(cache[0], dtype=int))
    text_idx = np.stack(text_indexes_test)  # idx里面是text类型中的节点在mapindex中的索引
    text_idx_map = {j: i for i, j in enumerate(text_idx)}  # 里面是{text中的节点在mapindex中的索引：idx_map中的索引}
    text_idx_map_list_test.append(text_idx_map)  # 里面一共有text节点类型个{}=[{}],{}={text类型中的节点的点在mapindex中的索引：idx_map中的索引}
    # print('text_idx done.')
    # idx_train, idx_test = load_divide_idx(path, idx_map_list[0], Hete='et')
    idx_train, idx_test = load_divide_idx(path, text_idx_map_list_train[0],text_idx_map_list_test[0], Hete='et')
    # return adj_list, features_list, Labels, idx_train, idx_test, idx_map_list[0]
    return adj_list, features_list, Labels, idx_train, idx_test, text_idx_map_list_train[0],text_idx_map_list_test[0]

'''
    ？？？
    是多粒度？？
    最后，我们需要捕捉不同信息的重要性，以解决在多个粒度层次上的稀疏性，并减少噪声信息的权重，以实现更准确的分类结果
'''
def multi_label(labels):
    def myfunction(x):
        return list(map(int, x[0].split()))
    return np.apply_along_axis(myfunction, axis=1, arr=labels)


def encode_onehot(labels):
    '''
        text的labels:[40*1]
    '''
    classes = set(labels.T[0]) # classes就是相当于给labels里40个标签去重后，剩下的8个label类别
    # classes：{'education-science', 'culture-arts-entertainment', 'health', 'politics-society', 'computers', 'sports', 'engineering', 'business'}

    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    #classed_dict： {'education-science': array([1., 0., 0., 0., 0., 0., 0., 0.]), 'culture-arts-entertainment': array([0., 1., 0., 0., 0., 0., 0., 0.]), 'health': array([0., 0., 1., 0., 0., 0., 0., 0.]), 'politics-society': array([0., 0., 0., 1., 0., 0., 0., 0.]), 'computers': array([0., 0., 0., 0., 1., 0., 0., 0.]), 'sports': array([0., 0., 0., 0., 0., 1., 0., 0.]), 'engineering': array([0., 0., 0., 0., 0., 0., 1., 0.]), 'business': array([0., 0., 0., 0., 0., 0., 0., 1.])}
    labels_onehot = np.array(list(map(classes_dict.get, labels.T[0])),
                             dtype=np.int32)
    return labels_onehot

def load_divide_idx(path, idx_map_train,idx_map_test,Hete):
    idx_train = []
    idx_test = []
    with open(path+'train_{}_train.map'.format(Hete), 'r') as f:
        for line in f:
            idx_train.append( idx_map_train.get(int(line.strip('\n'))) ) # idx_train:[18, 25, 6, 13, 7, 38, 1, 16, 0, 15, 5, 11, 9, 8, 12, 37]
    with open(path+'test_{}_test.map'.format(Hete), 'r') as f:
        for line in f:
            idx_test.append( idx_map_test.get(int(line.strip('\n'))) ) # idx_test:[36, 10, 21, 33, 23, 14, 32, 27]

    # print("train, test: ", len(idx_train), len(idx_test)) # 16 16 8
    # print('idx_train:',idx_train)
    # print('idx_test:',idx_test)
    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)
    return idx_train, idx_test


def resample(train, val, test : torch.LongTensor, path, idx_map, rewrite=True):
    if os.path.exists(path+'train_inductive.map'):
        rewrite = False
        filenames = ['train', 'unlabeled', 'vali', 'test']
        ans = []
        for file in filenames:
            with open(path+file+'_inductive.map', 'r') as f:
                cache = []
                for line in f:
                    cache.append(idx_map.get(int(line)))
            ans.append(torch.LongTensor(cache))
        return ans

    idx_train = train
    idx_test = val
    cache = list(test.numpy())
    shuffle(cache)
    idx_val = cache[: idx_train.shape[0]]
    idx_unlabeled = cache[idx_train.shape[0]: ]
    idx_val = torch.LongTensor(idx_val)
    idx_unlabeled = torch.LongTensor(idx_unlabeled)

    print("\n\ttrain: ", idx_train.shape[0],
          "\n\tunlabeled: ", idx_unlabeled.shape[0],
          "\n\tvali: ", idx_val.shape[0],
          "\n\ttest: ", idx_test.shape[0])
    if rewrite:
        idx_map_reverse = dict(map(lambda t: (t[1], t[0]), idx_map.items()))
        filenames = ['train', 'unlabeled', 'vali', 'test']
        ans = [idx_train, idx_unlabeled, idx_val, idx_test]
        for i in range(4):
            with open(path+filenames[i]+'_inductive.map', 'w') as f:
                f.write("\n".join(map(str, map(idx_map_reverse.get, ans[i].numpy()))))

    return idx_train, idx_unlabeled, idx_val, idx_test


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(mx):
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    if len(sparse_mx.nonzero()[0]) == 0:
        # 空矩阵
        r, c = sparse_mx.shape
        return torch.sparse.FloatTensor(r, c)
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def dense_tensor_to_sparse(dense_mx):
    return sparse_mx_to_torch_sparse_tensor( sp.coo.coo_matrix(dense_mx) )

'''
dirs:['log/','model/','embeddings/']
'''
def makedirs(dirs: list):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
    return