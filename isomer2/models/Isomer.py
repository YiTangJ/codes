# -*- coding: utf-8 -*-
'''
这个模型没有原论文中的提到的aspect-base attention，被注释掉了
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM
import copy
import numpy as np
from functools import reduce
from transformers.modeling_bert import BertLayer, BertLayerNorm, BertcoLayer
# from models.weight_distribution_with_similar import weight_distribution,weight_distribution_useEucliDist_sim,weight_distribution_useManhadunDist_sim,weight_distribution_useChebyshev_sim,weight_distribution_useYac_dist_sim,weight_distribution_useHan_dist_sim,weight_distribution_useSe_euc_dist_sim,weight_distribution_useCorrcoef_dist_sim
from models.kg_att9 import KG_att_text_entity2,KG_att_text_entity2_noConceptNet,KG_att_text_entity2_noSenticNet
# from models.kg_att9_rest16 import KG_att_text_entity2,KG_att_text_entity2_noConceptNet,KG_att_text_entity2_noSenticNet
# from models.kg_att9_rest14 import KG_att_text_entity2,KG_att_text_entity2_noConceptNet,KG_att_text_entity2_noSenticNet
# from models.kg_att9_twitter import KG_att_text_entity2,KG_att_text_entity2_noConceptNet,KG_att_text_entity2_noSenticNet
# from models.kg_att9_lap14 import KG_att_text_entity2, KG_att_text_entity2_noConceptNet, KG_att_text_entity2_noSenticNet


class Config:
    num_attention_heads = 1
    layer_norm_eps = 1e-20
    hidden_size = 200
    hidden_dropout_prob = 0.3
    intermediate_size = 200
    output_attentions = False
    attention_probs_dropout_prob = 0.3
    hidden_act = 'gelu'


config = Config()


def diji(adj):
    adj = np.array(adj)
    adjs = copy.deepcopy(adj)
    adjs[adjs == 0] = 1000
    length = adj.shape[0]
    for u in range(length):
        for i in range(length):
            for j in range(length):
                if adjs[i, u] + adjs[u, j] < adjs[i, j]:
                    adjs[i, j] = adjs[i, u] + adjs[u, j]
    adjs = (1 / adjs) ** 1
    #    print(adjs)
    adjss = adjs.sum(-1, keepdims=True)
    #    print(adjss)
    adjs = adjs / adjss
    return adjs


class selfalignment(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, bias=True):
        super(selfalignment, self).__init__()
        self.in_features = in_features
        self.dropout = nn.Dropout(0.1)
        #        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, in_features))
        self.linear = torch.nn.Linear(in_features, in_features, bias=False)
        self.linear1 = torch.nn.Linear(in_features, in_features)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(in_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, text1, textmask):
        # matmul是tensor的乘法
        logits = torch.matmul(self.linear(text), text1.transpose(1, 2))  # 对应式子（14）括号内
        masked = textmask.unsqueeze(1)
        masked = (1 - masked) * -1e20
        logits = torch.softmax(logits + masked, -1)  # 对应式子（14）的softmax==A1
        output = torch.matmul(logits, text1)  # 对应式子（16）S'_1
        #        output = self.dropout(torch.relu(self.linear1(torch.matmul(logits,text1))))+text
        output = output * textmask.unsqueeze(-1)
        if self.bias is not None:
            return output + self.bias, logits * textmask.unsqueeze(-1)
        else:
            return output, logits * textmask.unsqueeze(-1)


def init_weights(module):
    """ Initialize the weights.
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=0.01)
    elif isinstance(module, BertLayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

# 有nodeAttentionLevel
class HGATGraphAttentionConvolution(nn.Module):  # [25,25,16],512,True,0.1
    def __init__(self, in_features_list, out_features, bias=True, gamma=0.1):
        super(HGATGraphAttentionConvolution, self).__init__()
        self.ntype = len(in_features_list)  # 3
        self.in_features_list = in_features_list
        self.out_features = out_features
        self.weights = nn.ParameterList()  # 这个类实际上是将一个Parameter的List转为ParameterList
        for i in range(self.ntype):  # i=0 i=1 i=2
            cache = nn.Parameter(torch.FloatTensor(in_features_list[i], out_features))  # [25,512] [25,512] [16,512]
            nn.init.xavier_normal_(cache.data, gain=1.414)  # xavier初始化方法中服从正态分布，
            self.weights.append(cache)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
            stdv = 1. / math.sqrt(out_features)
            self.bias.data.uniform_(-stdv, stdv)
            # self.bias.cuda()
        else:
            self.register_parameter('bias', None)
            # self.register_parameter.cuda()

        self.att_list = nn.ModuleList()
        for i in range(self.ntype):
            self.att_list.append(Attention_NodeLevel(out_features, gamma))

    def forward(self, inputs_list, adj_list, global_W=None):
        h = []
        for i in range(self.ntype):
        # print('inputlist:',inputs_list.shape,';',type(inputs_list))
        # x_dimension = inputs_list.shape[0]
        # y_dimension = inputs_list.shape[1]
        # z_dimension = inputs_list.shape[-1]
        # inputs_list = inputs_list.view(-1, z_dimension)
        # print('inputlist:', inputs_list.shape, ';', type(inputs_list))
        # print('self.weight:',self.weights.shape)
        # inputs_list = dense_tensor_to_sparse(inputs_list.detach().numpy() )
            h.append(torch.spmm(inputs_list[i], self.weights[i]))
        # print('h:',h.shape,';',type(h))
        # adj_list.cuda()
        if global_W is not None:
            for i in range(self.ntype):
                h[i] = ( torch.spmm(h[i], global_W) )
        outputs = []
        for t1 in range(self.ntype):
            x_t1 = []
            for t2 in range(self.ntype):
                # adj has no non-zeros
                '''唐艺嘉写的try'''
                try:
                    if len(adj_list[t1][t2]._values()) == 0:
                        x_t1.append(torch.zeros(adj_list[t1][t2].shape[0], self.out_features, device=self.bias.device))
                        continue
                except:
                    if len(adj_list[t1][t2].to_sparse()._values()) == 0:
                        x_t1.append(torch.zeros(adj_list[t1][t2].shape[0], self.out_features, device=self.bias.device))
                        continue

                if self.bias is not None:
                    x_t1.append( self.att_list[t1](h[t1], h[t2], adj_list[t1][t2]) + self.bias)
                else:
                    x_t1.append( self.att_list[t1](h[t1], h[t2], adj_list[t1][t2]))
            outputs.append(x_t1)

        return outputs
class Attention_NodeLevel(nn.Module):  # 512, 0.1
    def __init__(self, dim_features, gamma=0.1):
        super(Attention_NodeLevel, self).__init__()

        self.dim_features = dim_features

        self.a1 = nn.Parameter(torch.zeros(size=(dim_features, 1)))  # [512,1]
        self.a2 = nn.Parameter(torch.zeros(size=(dim_features, 1)))  # [512,1]
        nn.init.xavier_normal_(self.a1.data, gain=1.414)
        nn.init.xavier_normal_(self.a2.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(0.2)  # 类别级注意和节点级注意用的激活函数
        self.gamma = gamma

    def forward(self, input1, input2, adj):
        h = input1
        g = input2
        N = h.size()[0]
        M = g.size()[0]
        # print('h shape:',h.shape)
        # print('g shape:',g.shape)
        # print('self a1 shape:',self.a1.shape)
        e1 = torch.matmul(h, self.a1).repeat(1, M)
        e2 = torch.matmul(g, self.a2).repeat(1, N).t()
        e = e1 + e2
        e = self.leakyrelu(e)  # 得到bvv'

        zero_vec = -9e15 * torch.ones_like(e)
        # print('adj类型：', torch.typename(adj))
        if 'sparse' in adj.type():
            adj_dense = adj.to_dense()
            attention = torch.where(adj_dense > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)
            attention = torch.mul(attention, adj_dense.sum(1).repeat(M, 1).t())
            attention = torch.add(attention * self.gamma, adj_dense * (1 - self.gamma))
            del (adj_dense)
        else:
            # # print('adj:',adj.shape)
            # # print('e:',e.shape)
            # # print('zero_vec:',zero_vec.shape)
            # adj = adj.view(e.shape[0],-1)
            # # print('adj:',adj.shape)
            # expand_size = e.shape[-1]//adj.shape[-1]
            # # print('expandsize:',expand_size)
            # adj = adj.repeat(1,expand_size)
            # # print('adj:',adj.shape)
            attention = torch.where(adj > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)  # 得到βvv'
            attention = torch.mul(attention, adj.sum(1).repeat(M, 1).t())
            try:
                attention = torch.add(attention * self.gamma, adj.to_dense() * (1 - self.gamma))
            except:
                attention = torch.add(attention * self.gamma, adj.to_sparse().to_dense() * (1 - self.gamma))
        del (zero_vec)
        # print('attention:',attention.shape,';',attention.dtype)
        # print('g:',g.shape,';',g.dtype)
        # attention = attention.float()
        # print('attentiton:',attention.shape,';',attention.dtype)
        h_prime = torch.matmul(attention, g)

        return h_prime
# 无nodeAttentionLevel
class HGATGraphConvolution(nn.Module): # 512， 10， True
    def __init__(self, in_features, out_features, bias=True):
        super(HGATGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features)) # [512,10]
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self): # GraphConvolution 512 -> 10
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj, global_W = None):
        # adj has no non-zeros
        try:
            if len(adj._values()) == 0:
                return torch.zeros(adj.shape[0], self.out_features, device=inputs.device)
        except:
            if len(adj.to_sparse()._values()) == 0:
                return torch.zeros(adj.shape[0], self.out_features, device=inputs.device)

        support = torch.spmm(inputs, self.weight) # torch.spmm是稀疏矩阵的乘法
        # print('support:',support.shape)
        # print('inputs:',inputs.shape)
        # print('self.weight:',self.weight.shape)
        if global_W is not None:
            support = torch.spmm(support, global_W) # 本文不执行这句，就是和简单的GraphConvolutional一样的操作
        # print('adj:',adj.shape,';',type(adj))
        # print('support:',support.shape,';',type(support))
        # adj = adj.view(-1,support.shape[0])
        # print('adj:',adj.shape)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
# 用于typeAttentionLevel
class HGATSelfAttention(nn.Module): # 512, 0, 50
    def __init__(self, in_features, idx, hidden_dim):
        super(HGATSelfAttention, self).__init__()
        self.idx = idx
        self.linear = torch.nn.Linear(in_features, hidden_dim)
        self.a = nn.Parameter(torch.FloatTensor(2 * hidden_dim, 1))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.a.size(1))
        self.a.data.uniform_(-stdv, stdv)

    def forward(self, inputs):
        # inputs size:  node_num * 3 * in_features
        x = self.linear(inputs).transpose(0, 1)
        self.n = x.size()[0]
        # # print('x:',x.shape,';',type(x))
        # x = torch.cat([x, x], dim=-1)
        # # print('x:',x.shape,';',type(x))
        # x = x.view(-1,self.a.shape[0])
        # # print('x:',x.shape)
        # U = torch.matmul(x, self.a).transpose(0, 1)
        # U = F.leaky_relu_(U) # 得到aτ==类型级注意得分
        # weights = F.softmax(U, dim=1) # 得到ατ==类型级注意权重
        # # print('weights:',weights.shape)
        # # print('inputs:',inputs.shape)
        # outputs = torch.matmul(weights, inputs).squeeze(1) * 3

        x = torch.cat([x, torch.stack([x[self.idx]] * self.n, dim=0)], dim=2)
        U = torch.matmul(x, self.a).transpose(0, 1)
        U = F.leaky_relu_(U)  # 得到aτ==类型级注意得分
        weights = F.softmax(U, dim=1)  # 得到ατ==类型级注意权重
        outputs = torch.matmul(weights.transpose(1, 2), inputs).squeeze(1) * 3
        '''这行代码解释：
        1、Outputs = weights_transpose * inputs * 3 (I'm not sure why it is multiplied by 3)
        2、redundant dimension is removed by squeeze 多余的尺寸通过squeeze去除
        3、outputs shape = (N, in_features) = (N, 512)
        4、torch.matmul uses batch multiplication.
        '''
        return outputs, weights


class HGAT(nn.Module):
    def __init__(self,opt, nfeat_list, nhid, nclass, dropout, type_attention=True, node_attention=True, gamma=0.1, sigmoid=False, orphan=True):
        '''HGAT'''
        super(HGAT, self).__init__()
        self.opt = opt
        self.sigmoid = sigmoid  # 用来做最后的softmax分类层判断的，为false就是使用log_softmax+模型外面的nll_loss()==crossentropyloss交叉熵
        self.type_attention = type_attention  # 就是type_level==layers中的SelfAttention
        self.node_attention = node_attention  # 就是node_level==layers中的Attention_NodeLevel
        self.write_emb = True
        if self.write_emb:
            self.emb = None
            self.emb2 = None
        self.nonlinear = F.relu_
        self.nclass = nclass  # 就是数据中存在的分类类别有哪些
        self.ntype = len(nfeat_list)  # 3 === 好像就是文中提到的3种节点类型：short text、topic、entity？？？
        dim_1st = 2 * opt.hidden_dim  # HGAT维度512，预训练词嵌入维度100
        dim_2nd = 2 * opt.hidden_dim
        # dim_2nd = nclass
        # if orphan:
        #     dim_2nd += self.ntype - 1  # 10
        self.gc2 = nn.ModuleList()  # 对于cnn前馈神经网络如果前馈一次写一个forward函数会有些麻烦，在此就有两种简化方式，ModuleList和Sequential。ModuleList是Module的子类, 被设计用来存储任意数量的nn. module。如果在构造函数__ init__中用到list、tuple、dict等对象时，一定要思考是否应该用ModuleList或ParameterList代替。ModuleList里面没有forward函数
        if not self.node_attention:
            self.gc1 = nn.ModuleList()
            for t in range(self.ntype):
                self.gc1.append(GraphConvolution(nfeat_list[t], dim_1st, bias=False))
                self.bias1 = nn.Parameter(torch.FloatTensor(dim_1st))
                stdv = 1. / math.sqrt(dim_1st)
                self.bias1.data.uniform_(-stdv, stdv)
        else:
            self.gc1 = HGATGraphAttentionConvolution(nfeat_list, dim_1st, gamma=gamma)  # 本文的方法图注意卷积
        self.gc2.append(HGATGraphConvolution(dim_1st, dim_2nd, bias=True))
        if self.type_attention:
            self.at1 = nn.ModuleList()
            self.at2 = nn.ModuleList()
            for t in range(self.ntype):
                self.at1.append(HGATSelfAttention(dim_1st, t, 50))
                self.at2.append(HGATSelfAttention(dim_2nd, t, 50))

        self.dropout_HGAT = dropout
    def forward(self, x_list, adj_list, nfeat_list):
        x0 = x_list
        if not self.node_attention:
            x1 = [None for _ in range(self.ntype)]
            # First Layer--gc1&&at1
            for t1 in range(self.ntype):
                x_t1 = []
                for t2 in range(self.ntype):
                    idx = t2
                    x_t1.append(self.gc1[idx](x0[t2], adj_list[t1][t2]) + self.bias1)
                if self.type_attention:
                    x_t1, weights = self.at1[t1](torch.stack(x_t1, dim=1))
                else:
                    x_t1 = reduce(torch.add, x_t1)

                x_t1 = self.nonlinear(x_t1)
                x_t1 = F.dropout(x_t1, self.dropout_HGAT, training=self.training)
                x1[t1] = x_t1
        else:
            x1 = [None for _ in range(self.ntype)]
            x1_in = self.gc1(x0, adj_list)  # 本文加入了attentionNodeLevel的图注意卷积
            for t1 in range(len(x1_in)):
                x_t1 = x1_in[t1]
                if self.type_attention:
                    x_t1, weights = self.at1[t1](torch.stack(x_t1, dim=1))
                else:
                    x_t1 = reduce(torch.add, x_t1)
                x_t1 = self.nonlinear(x_t1)
                x_t1 = F.dropout(x_t1, self.dropout_HGAT, training=self.training)
                x1[t1] = x_t1
        if self.write_emb:
            self.emb = x1[0]
        x2 = [None for _ in range(self.ntype)]
        # Second Layer-gc2&&at2
        for t1 in range(self.ntype):
            x_t1 = []
            for t2 in range(self.ntype):
                if adj_list[t1][t2] is None:
                    continue
                idx = 0
                x_t1.append(self.gc2[idx](x1[t2], adj_list[t1][t2]))
            if self.type_attention:
                x_t1, weights = self.at2[t1](torch.stack(x_t1, dim=1))
            else:
                x_t1 = reduce(torch.add, x_t1)
            x2[t1] = x_t1
            if self.write_emb and t1 == 0:
                self.emb2 = x2[t1]
            # if self.sigmoid:
            #     x2[t1] = torch.sigmoid(x_t1)
            # else:
            #     x2[t1] = F.log_softmax(x_t1, dim=1)
        return x2

class simpleGraphConvolutionalignment(nn.Module):

    def __init__(self, in_features, out_features, edge_size, bias=False):
        super(simpleGraphConvolutionalignment, self).__init__()
        self.K = 4
        self.norm = torch.nn.LayerNorm(out_features)
        self.edge_vocab = torch.nn.Embedding(edge_size, out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = nn.Dropout(0.1)
        #        self.dropout1 = nn.Dropout(0.0)
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.linear = torch.nn.Linear(out_features, out_features, bias=False)
        #        self.linear2=torch.nn.Linear(4*out_features,out_features,bias=False)
        self.align = selfalignment(out_features)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)



    def renorm(self, adj1, adj2):
        adj = adj1 * adj2
        adj = adj / (adj.sum(-1).unsqueeze(-1) + 1e-10)
        return adj

    def forward(self, text, adj1, adj2, edge1, edge2, textmask,hete):
        adj = adj1 + adj2  # 将进边和出边合并
        adj[adj >= 1] = 1
        edge = self.edge_vocab(edge1 + edge2)
        textlen = text.size(1)  # s
        attss = [adj1, adj2]

        output = hete[0]
        output = torch.mean(output, dim=0)  # 求均值
        output = torch.unsqueeze(output, dim=0)
        # print('unsqueeze:',output.shape) # [1, 200]
        output = output.expand(text.shape[0], text.shape[1], output.shape[-1])
        # print('expand:', output.shape) # ([32, 40, 200]

        for i in range(self.K):  # 3次
            outs = output
            teout = self.linear(output)
            denom1 = torch.sum(adj, dim=2, keepdim=True) + 1
            output = self.dropout(torch.relu(torch.matmul(adj, teout) / denom1))  # 对应式子（11）的Relu里面
            if self.bias is not None:
                output = output + self.bias  # 对应式子（11）的norm（）中的
            output = self.norm(output) + outs  # 对应式子（11）的norm
            outs = output
        outs1 = outs
        outs, att1 = self.align(outs, outs, textmask)
        outs = torch.cat([outs, outs1], -1)
        return outs, attss  # b,s,h

class simpleGraphConvolutionalignment_noHete(nn.Module):

    def __init__(self, in_features, out_features, edge_size, bias=False):
        super(simpleGraphConvolutionalignment_noHete, self).__init__()
        self.K = 3
        self.norm = torch.nn.LayerNorm(out_features)
        self.edge_vocab = torch.nn.Embedding(edge_size, out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = nn.Dropout(0.1)
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.linear = torch.nn.Linear(out_features, out_features, bias=False)
        self.align = selfalignment(out_features)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)



    def renorm(self, adj1, adj2):
        adj = adj1 * adj2
        adj = adj / (adj.sum(-1).unsqueeze(-1) + 1e-10)
        return adj

    def forward(self, text, adj1, adj2, edge1, edge2, textmask):
        adj = adj1 + adj2  # 将进边和出边合并
        adj[adj >= 1] = 1
        edge = self.edge_vocab(edge1 + edge2)
        textlen = text.size(1)  # s
        attss = [adj1, adj2]
        if (self.in_features != self.out_features):
            output = torch.relu(torch.matmul(text, self.weight))  # 对应式子（9）
            out = torch.relu(torch.matmul(text, self.weight))  # 对应式子（10）
            outss = out
        else:
            output = text
            out = text
            outss = text
        for i in range(self.K):  # 3次
            outs = output
            teout = self.linear(output)
            denom1 = torch.sum(adj, dim=2, keepdim=True) + 1
            output = self.dropout(torch.relu(torch.matmul(adj, teout) / denom1))
            if self.bias is not None:
                output = output + self.bias
            output = self.norm(output) + outs
            outs = output
        outs1 = outs
        outs, att1 = self.align(outs, outs, textmask)
        outs = torch.cat([outs, outs1], -1)
        return outs, attss  # b,s,h

def length2mask(length, maxlength,opt):
    size = list(length.size())
    length = length.unsqueeze(-1).repeat(*([1] * len(size) + [maxlength])).long()
    ran = torch.arange(maxlength).to(opt.device)
    ran = ran.expand_as(length)
    mask = ran < length
    return mask.float().to(opt.device)


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class biedgeGraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, edge_size, bias=True):
        super(biedgeGraphConvolution, self).__init__()
        self.K = 3
        self.norm = torch.nn.LayerNorm(out_features)
        self.edge_vocab = torch.nn.Embedding(edge_size, out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = nn.Dropout(0.3)
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.fuse1 = nn.Sequential(torch.nn.Linear(2 * out_features, out_features, bias=False), torch.nn.ReLU())
        self.fuse2 = nn.Sequential(torch.nn.Linear(2 * out_features, out_features, bias=False), torch.nn.ReLU())
        self.fc3 = nn.Sequential(torch.nn.Linear(2 * out_features, out_features, bias=False), torch.nn.ReLU())
        self.fc1 = nn.Sequential(torch.nn.Linear(3 * out_features, out_features, bias=False), torch.nn.ReLU(),
                                 torch.nn.Linear(out_features, 1, bias=True))
        self.fc2 = nn.Sequential(torch.nn.Linear(3 * out_features, out_features, bias=False), torch.nn.ReLU(),
                                 torch.nn.Linear(out_features, 1, bias=True))
        self.fc1s = nn.Sequential(torch.nn.Linear(3 * out_features, out_features, bias=False), torch.nn.ReLU(),
                                  torch.nn.Linear(out_features, 1, bias=True))
        self.fc2s = nn.Sequential(torch.nn.Linear(3 * out_features, out_features, bias=False), torch.nn.ReLU(),
                                  torch.nn.Linear(out_features, 1, bias=True))
        self.align = selfalignment(out_features)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def renorm(self, adj1, adj2, a=0.5):
        adj = adj1 + a * adj2
        adj = adj / (adj.sum(-1).unsqueeze(-1) + 1e-20)
        return adj

    def forward(self, text, adj1, adj2, edge1, edge2, textmask):
        #        print(edge1.size())
        #        edge1=self.edge_vocab(edge1)#b,s,s,h
        #        edge2=self.edge_vocab(edge2)#b,s,s,h
        textlen = text.size(1)  # s
        outss, att1 = self.align(text, text, textmask)
        output = torch.relu(torch.matmul(text, self.weight))
        #        adj1s=copy.deepcopy(adj1)
        #        adj2s=copy.deepcopy(adj2)
        #        for i in range(adj1s.size(1)):
        #            adj1s[:,i,i]=0
        #            adj2s[:,i,i]=0
        for i in range(self.K):
            text1 = output.unsqueeze(-2).repeat(1, 1, textlen, 1)
            text2 = output.unsqueeze(-3).repeat(1, textlen, 1, 1)
            teout = self.fuse1(torch.cat([text2, edge1], -1))  # b,s,s,h
            tein = self.fuse2(torch.cat([text2, edge2], -1))  # b,s,s,h
            teouts = torch.sigmoid(self.fc1(torch.cat([text1, text2, edge1], -1)))  # b,s,s,1
            teins = torch.sigmoid(self.fc2(torch.cat([text1, text2, edge2], -1)))  # b,s,s,1
            #            teoutss=torch.softmax((1-adj1s)*-1e20+self.fc1s(torch.cat([text1,text2,edge1],-1)).squeeze(-1),-1)#b,s,s,1
            #            teinss=torch.softmax((1-adj2s)*-1e20+self.fc2s(torch.cat([text1,text2,edge2],-1)).squeeze(-1),-1)#b,s,s,1
            #            for i in range(adj1s.size(1)):
            #                teoutss.data[:,i,i]=1.0
            #                teinss.data[:,i,i]=1.0
            #            hidden1 = torch.matmul(text, self.weight)
            denom1 = torch.sum(adj1, dim=2, keepdim=True) + 1
            #            adj1s=self.renorm(att1,adj1)
            #            output1 = torch.sum(adj1.unsqueeze(-1)*teout*teouts*teoutss.unsqueeze(-1),-2) / denom1
            output1 = torch.sum(adj1.unsqueeze(-1) * teout * teouts, -2) / denom1
            denom2 = torch.sum(adj2, dim=2, keepdim=True) + 1
            #            adj2s=self.renorm(att1,adj2)
            #            output2 = torch.sum(adj2.unsqueeze(-1)*tein*teins*teinss.unsqueeze(-1),-2) / denom2
            output2 = torch.sum(adj2.unsqueeze(-1) * tein * teins, -2) / denom2
            output = self.fc3(torch.cat([output1, output2], -1)) + output
            if self.bias is not None:
                output = output + self.bias
            output = self.dropout(self.norm(output))
        return output, outss  # b,s,h


class ASGCN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(ASGCN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.text_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.gc1 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.gc2 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.fc = nn.Linear(2 * opt.hidden_dim, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(0.3)

    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i, 0]):
                weight[i].append(1 - (aspect_double_idx[i, 0] - j) / context_len)
            for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1] + 1):
                weight[i].append(0)
            for j in range(aspect_double_idx[i, 1] + 1, text_len[i]):
                weight[i].append(1 - (j - aspect_double_idx[i, 1]) / context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight).unsqueeze(2).to(self.opt.device)
        return weight * x

    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i, 0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1] + 1):
                mask[i].append(1)
            for j in range(aspect_double_idx[i, 1] + 1, seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask).unsqueeze(2).float().to(self.opt.device)
        return mask * x

    def forward(self, inputs):
        text_indices, aspect_indices, left_indices, adj = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len + aspect_len - 1).unsqueeze(1)], dim=1)
        text = self.embed(text_indices)
        text = self.text_embed_dropout(text)
        text_out, (_, _) = self.text_lstm(text, text_len)
        x = F.relu(self.gc1(self.position_weight(text_out, aspect_double_idx, text_len, aspect_len), adj))
        x = F.relu(self.gc2(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        x = self.mask(x, aspect_double_idx)
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1)  # batch_size x 2*hidden_dim
        output = self.fc(x)
        return output


class mutualatts(nn.Module):
    def __init__(self, hidden_size):
        super(mutualatt, self).__init__()
        self.linear2 = torch.nn.Linear(3 * hidden_size, hidden_size)
        self.linear1 = torch.nn.Linear(hidden_size, 1)

    def forward(self, in1, in2, text, textmask):
        length = text.size(1)
        in1 = in1.unsqueeze(1).repeat(1, length, 1)
        in2 = in2.unsqueeze(1).repeat(1, length, 1)
        att = self.linear1(torch.tanh(self.linear2(torch.cat([in1, in2, text], -1)))).squeeze(-1)
        att = torch.softmax(((1 - textmask) * -1e20 + att), -1).unsqueeze(1)
        context = torch.matmul(att, text).squeeze(1)
        return context, att


class mutualatt(nn.Module):
    def __init__(self, hidden_size):
        super(mutualatt, self).__init__()
        self.linear2 = torch.nn.Linear(2 * hidden_size, hidden_size)
        self.linear1 = torch.nn.Linear(hidden_size, 1)

    def forward(self, in1, text, textmask):
        length = text.size(1)
        in1 = in1.unsqueeze(1).repeat(1, length, 1)
        att = self.linear1(torch.tanh(self.linear2(torch.cat([in1, text], -1)))).squeeze(-1)
        #        print((1-textmask)*-1e20)
        att = torch.softmax(att, -1) * textmask
        att = (att / (att.sum(-1, keepdim=True) + 1e-20)).unsqueeze(-2)
        #        print(att.size())
        #        print(text.size())
        #        att=torch.softmax(((1-textmask)*-1e20+att),-1).unsqueeze(1)
        context = torch.matmul(att, text).squeeze(1)
        return context, att

'''双通道cat之后的selfattention'''
class Self_Attention_Layer_AfterCat(nn.Module):

    # 用来实现mask-attention layer
    def __init__(self, hidden_dim, is_bi_rnn):
        super(Self_Attention_Layer_AfterCat, self).__init__()

        self.hidden_dim = hidden_dim
        self.is_bi_rnn = is_bi_rnn

        # 下面使用nn的Linear层来定义Q，K，V矩阵
        if is_bi_rnn:
            # 是双向的RNN
            self.Q_linear = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
            self.K_linear = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
            self.V_linear = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
        else:
            # 单向的RNN
            self.Q_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.K_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.V_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, inputs, length,maxlength,opt):
        '''
        lens:<class 'torch.Tensor'>
        '''
        size = inputs.size()
        # 计算生成QKV矩阵
        Q = self.Q_linear(inputs).to(opt.device)
        K = self.K_linear(inputs).permute(0, 2, 1).to(opt.device)  # 先进行一次转置
        V = self.V_linear(inputs).to(opt.device)

        # 还要计算生成mask矩阵
        # max_len = max(lens)  # 最大的句子长度，生成mask矩阵
        max_len = maxlength*2
        sentence_lengths = length.float()*2
        sentence_lengths.to(opt.device)  # 代表每个句子的长度
        mask = (torch.arange(sentence_lengths.max().item())[None, :]).to(opt.device) < sentence_lengths[:, None]
        mask = mask.unsqueeze(dim=1)  # [batch_size, 1, max_len]
        mask = mask.expand(size[0], max_len, max_len)  # [batch_size, max_len, max_len]

        # size = list(length.size())
        # length = length.unsqueeze(-1).repeat(*([1] * len(size) + [maxlength])).long()
        # ran = torch.arange(maxlength).to(opt.device)
        # ran = ran.expand_as(length)
        # mask = ran < length
        # mask = mask.float().to(opt.device)
        # print('\nmask is :', mask.size())

        # 下面生成用来填充的矩阵
        padding_num = torch.ones_like(mask)
        padding_num = -2 ** 31 * padding_num.float()

        # print('\npadding num is :', padding_num.size())

        # 下面开始计算啦
        alpha = torch.matmul(Q, K)

        # 下面开始mask
        # print('mask:',mask.shape,'alpha:',alpha.shape)
        # print('mask:',mask)
        alpha = torch.where(mask, alpha, padding_num)
        # 下面开始softmax
        alpha = F.softmax(alpha, dim=2)
        # print('\nalpha is :', alpha)

        out = torch.matmul(alpha, V)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super(MultiHeadAttention).__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        # d_model // h 仍然是要能整除，换个名字仍然意义不变
        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        # Q,K,V计算与变形：
        bsz = query.shape[0]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)

        # Q, K相乘除以scale，这是计算scaled dot product attention的第一步
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # 如果没有mask，就生成一个
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        # 然后对Q,K相乘的结果计算softmax加上dropout，这是计算scaled dot product attention的第二步：
        attention = self.do(torch.softmax(energy, dim=-1))

        # 第三步，attention结果与V相乘
        x = torch.matmul(attention, V)

        # 最后将多头排列好，就是multi-head attention的结果了
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        x = self.fc(x)
        return x
class Multihead_Attention(nn.Module):
    """
    multihead_attention
    根据<https://www.github.com/kyubyong/transformer>修改
    1.split+cat
    2.matmul(q,k)
    3.mask k
    4.softmax
    5.mask q
    6.matmul(attn,v)
    7.split+cat
    8.res q
    9.norm
    """

    def __init__(self,
                 hidden_dim,
                 C_q=None,
                 C_k=None,
                 C_v=None,
                 num_heads=1,
                 dropout_rate=0.0):
        super(Multihead_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        C_q = C_q if C_q else hidden_dim
        C_k = C_k if C_k else hidden_dim
        C_v = C_v if C_v else hidden_dim
        self.linear_Q = nn.Linear(C_q, hidden_dim)
        self.linear_K = nn.Linear(C_k, hidden_dim)
        self.linear_V = nn.Linear(C_v, hidden_dim)
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self,Q, K, V):
        """
        :param Q: A 3d tensor with shape of [N, T_q, C_q]
        :param K: A 3d tensor with shape of [N, T_k, C_k]
        :param V: A 3d tensor with shape of [N, T_v, C_v]
        :return:
        """
        num_heads = self.num_heads
        N = Q.size()[0]

        # Linear projections
        # print('linear_Q:',self.linear_Q.weight.shape,'linear_K:',self.linear_K.weight.shape,'linear_V:',self.linear_V.weight.shape)
        # print('q:',Q.shape,'K:',K.shape,'V:',V.shape)
        Q_l = nn.ReLU()(self.linear_Q(Q))
        K_l = nn.ReLU()(self.linear_K(K))
        V_l = nn.ReLU()(self.linear_V(V))

        # Split and concat
        Q_split = Q_l.split(split_size=self.hidden_dim // num_heads, dim=2)
        K_split = K_l.split(split_size=self.hidden_dim // num_heads, dim=2)
        V_split = V_l.split(split_size=self.hidden_dim // num_heads, dim=2)

        Q_ = torch.cat(Q_split, dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(K_split, dim=0)  # (h*N, T_k, C/h)
        V_ = torch.cat(V_split, dim=0)  # (h*N, T_v, C/h)

        # Multiplication
        outputs = torch.bmm(Q_, K_.transpose(2, 1))

        # Scale
        outputs = outputs / (K_.size()[-1] ** 0.5)

        # Key Masking
        key_masks = torch.sign(torch.abs(K).sum(dim=-1))  # (N, T_k)
        key_masks = key_masks.repeat(num_heads, 1)  # (h*N, T_k)
        key_masks = key_masks.unsqueeze(1).repeat(1, Q.size()[1], 1)  # (h*N, T_q, T_k)

        paddings = torch.ones_like(key_masks) * (-2 ** 32 + 1)
        outputs = torch.where(torch.eq(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = nn.Softmax(dim=2)(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = torch.sign(torch.abs(Q).sum(dim=-1))  # (N, T_q)
        query_masks = query_masks.repeat(num_heads, 1)  # (h*N, T_q)
        query_masks = query_masks.unsqueeze(-1).repeat(1, 1, K.size()[1])  # (h*N, T_q, T_k)
        outputs = outputs * query_masks  # broadcasting. (h*N, T_q, T_k)

        # Dropouts
        outputs = self.dropout(outputs)

        # Weighted sum
        outputs = torch.bmm(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = outputs.split(N, dim=0)  # (N, T_q, C)
        outputs = torch.cat(outputs, dim=2)

        # Residual connection
        outputs = outputs + Q_l

        # Normalize
        outputs = self.norm(outputs)  # (N, T_q, C)

        return outputs


class ASBIGCN(nn.Module):
    '''
    整个模型过程：
        Transformer——BiGCN[依赖图]
            BiAffine
            Norm
            Norm
    '''

    def __init__(self, embedding_matrix, opt, conceptURI, GISet, senticnetDict,adj_HGAT_data_ws_train, features_HGAT_data_ws_train,labels_HGAT_data_ws_train,
                                        adj_HGAT_data_et_train, features_HGAT_data_et_train,labels_HGAT_data_et_train,
                                        adj_HGAT_data_ws_test, features_HGAT_data_ws_test, labels_HGAT_data_ws_test,
                                        adj_HGAT_data_et_test, features_HGAT_data_et_test, labels_HGAT_data_et_test):
        '''
        embedding_metrix:数据集大小*300
        '''
        super(ASBIGCN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.text_lstm = DynamicLSTM(768, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.gc = simpleGraphConvolutionalignment(2 * opt.hidden_dim, 2 * opt.hidden_dim, opt.edge_size, bias=True)
        self.gc2 = simpleGraphConvolutionalignment(2 * opt.hidden_dim, 2 * opt.hidden_dim, opt.edge_size, bias=True)
        # self.gc_nohete = simpleGraphConvolutionalignment_noHete(2 * opt.hidden_dim, 2 * opt.hidden_dim, opt.edge_size, bias=True)
        self.fc = nn.Linear(8 * opt.hidden_dim, opt.polarities_dim)
        #        self.fc1 = nn.Linear(768*2,768)
        self.text_embed_dropout = nn.Dropout(0.1)
        self.Multihead_attention_afterCat = Multihead_Attention(hidden_dim=2 * 2 * opt.hidden_dim,num_heads=5,dropout_rate=0.05)
        self.Multihead_attention_afterSimilar = Multihead_Attention(hidden_dim=2 * 2 * opt.hidden_dim,num_heads=4,dropout_rate=0.05)
        self.Multihead_attention_afterSimilar2 = Multihead_Attention(hidden_dim=2 * 2 * opt.hidden_dim,num_heads=5,dropout_rate=0.05)
        self.conceptURI = conceptURI
        self.GISet = GISet
        self.senticnetDict = senticnetDict
        self.adj_HGAT_data_ws_train = adj_HGAT_data_ws_train
        self.features_HGAT_data_ws_train =features_HGAT_data_ws_train
        self.labels_HGAT_data_ws_train=labels_HGAT_data_ws_train
        self.adj_HGAT_data_et_train=adj_HGAT_data_et_train
        self.features_HGAT_data_et_train=features_HGAT_data_et_train
        self.labels_HGAT_data_et_train=labels_HGAT_data_et_train
        self.adj_HGAT_data_ws_test=adj_HGAT_data_ws_test
        self.features_HGAT_data_ws_test=features_HGAT_data_ws_test
        self.labels_HGAT_data_ws_test=labels_HGAT_data_ws_test
        self.adj_HGAT_data_et_test=adj_HGAT_data_et_test
        self.features_HGAT_data_et_test=features_HGAT_data_et_test
        self.labels_HGAT_data_et_test=labels_HGAT_data_et_test


    def forward(self, inputs,targets, mixed=True,trained=True):
        text_indices, span_indices, tran_indices, adj1, adj2, edge1, edge2, aspect_list, text_list = inputs
        if trained:
            input_adj_HGAT_data_ws, input_features_HGAT_data_ws, labels_HGAT_data_ws = self.adj_HGAT_data_ws_train,self.features_HGAT_data_ws_train,self.labels_HGAT_data_ws_train
            input_adj_HGAT_data_et, input_features_HGAT_data_et, labels_HGAT_data_et = self.adj_HGAT_data_et_train,self.features_HGAT_data_et_train,self.labels_HGAT_data_et_train
        else:
            input_adj_HGAT_data_ws, input_features_HGAT_data_ws, labels_HGAT_data_ws = self.adj_HGAT_data_ws_test,self.features_HGAT_data_ws_test,self.labels_HGAT_data_ws_test
            input_adj_HGAT_data_et, input_features_HGAT_data_et, labels_HGAT_data_et = self.adj_HGAT_data_et_test,self.features_HGAT_data_et_test,self.labels_HGAT_data_et_test
        # '''异构'''
        N_HGAT_data_ws = len(input_features_HGAT_data_ws)  # 3
        for i in range(N_HGAT_data_ws):
            if input_features_HGAT_data_ws[i] is not None:
                input_features_HGAT_data_ws[i] = input_features_HGAT_data_ws[i].to(self.opt.device)
        for i in range(N_HGAT_data_ws):
            for j in range(N_HGAT_data_ws):
                if input_adj_HGAT_data_ws[i][j] is not None:
                    input_adj_HGAT_data_ws[i][j] = input_adj_HGAT_data_ws[i][j].to(self.opt.device)
        labels_HGAT_data_ws = labels_HGAT_data_ws.to(self.opt.device)
        #
        N_HGAT_data_et = len(input_features_HGAT_data_et)  # 3
        for i in range(N_HGAT_data_et):
            if input_features_HGAT_data_et[i] is not None:
                input_features_HGAT_data_et[i] = input_features_HGAT_data_et[i].to(self.opt.device)
        for i in range(N_HGAT_data_et):
            for j in range(N_HGAT_data_et):
                if input_adj_HGAT_data_et[i][j] is not None:
                    input_adj_HGAT_data_et[i][j] = input_adj_HGAT_data_et[i][j].to(self.opt.device)
        labels_HGAT_data_et = labels_HGAT_data_et.to(self.opt.device)
        #
        self.ws = HGAT(self.opt,
                                                  nfeat_list=[i.shape[1] for i in input_features_HGAT_data_ws],
                                                  type_attention=True, node_attention=True, nhid=512,
                                                  nclass=labels_HGAT_data_ws.shape[1], dropout=0.7, gamma=0.1,
                                                  orphan=True).to(self.opt.device)
        self.et = HGAT(self.opt,
                                                   nfeat_list=[i.shape[1] for i in input_features_HGAT_data_et],
                                                   type_attention=True, node_attention=True, nhid=512,
                                                   nclass=labels_HGAT_data_et.shape[1], dropout=0.8, gamma=0.1,
                                                   orphan=True).to(self.opt.device)


        batchhalf = text_indices.size(0) // 2
        text_len = torch.sum(text_indices != 0, dim=-1)

        # 获取隐藏上下文表示
        if self.opt.usebert:
            outputs = self.bert(text_indices, attention_mask=length2mask(text_len, text_indices.size(1),self.opt))[0]
            output = outputs[:, 1:, :] # ([32, 45, 768]
            oss = self.text_embed_dropout(outputs[:, 0, :])
        else:

            '''
            用BILSTM提取隐藏的上下文表示
            '''
            #            print(text_len.data)
            #            print(self.embed.weight.size(0))
            #            print(text_indices.max().data)
            output, (_, _) = self.text_lstm2(self.embed(text_indices), text_len)
            #            output=self.embed(text_indices)
            output = output[:, 1:, :]
        # 将每个aspect折叠作为一个单一的单词，把每一个组成词的表示的总和作为最后它的隐藏表示
        max_len = max([len(item) for item in tran_indices])
        text_len = torch.Tensor([len(item) for item in tran_indices]).long().to(self.opt.device)
        tmps = torch.zeros(text_indices.size(0), max_len, 768).float().to(self.opt.device)
        '''
        span集的生成??——span做sum
        对应于公式（8）
        '''
        for i, spans in enumerate(tran_indices):
            for j, span in enumerate(spans):
                tmps[i, j] = torch.sum(output[i, span[0]:span[1]], 0)
        text = self.text_embed_dropout(tmps)

        text, (hout, _) = self.text_lstm(text, text_len) #text: [32, 40, 200])
        x = text
        #        text_out=torch.relu(self.linear3(text))
        hout = torch.transpose(hout, 0, 1)
        hout = hout.reshape(hout.size(0), -1)
        #        print(text_out.size())
        '''kgs知识图谱增强模块'''
        text = KG_att_text_entity2(text_list, aspect_list, text, span_indices, self.conceptURI,self.GISet,self.senticnetDict) #ConceptNet+SenticNet+GI+ Aspect-based Attention Module


        ws = self.ws(input_features_HGAT_data_ws, input_adj_HGAT_data_ws, nfeat_list = text.shape[-1]) # ws[0]:[1491, 200]
        et = self.et(input_features_HGAT_data_et, input_adj_HGAT_data_et, nfeat_list = text.shape[-1]) # et[0]:[1204, 200]
        # ws = ''
        text_ws, self.attss_ws = self.gc(text, adj1, adj2, edge1, edge2,
                                   length2mask(text_len, max_len,self.opt),ws)  # simpleGraphConvolutionalignment的forward()函数
        text_et, self.attss_et = self.gc2(text, adj1, adj2, edge1, edge2,
                                   length2mask(text_len, max_len,self.opt),et)  # simpleGraphConvolutionalignment的forward()函数
        text = torch.cat([text_ws,text_et],dim=1)
        #multihead att
        text = self.Multihead_attention_afterCat(text,text_ws,text_et)

        # # '''similar for SCI'''
        from models.weight_distribution_with_similar import weight_distribution_useChebyshev_sim
        # text_noHete = torch.cat([text_noHete,text_noHete],dim=1)
        # text_noHete = self.Multihead_attention_afterNoHeteCat(text_noHete,text_noHete,text_noHete)
        # text = torch.cat([text,text_noHete],dim=1)
        text_after_weight_distribution = weight_distribution_useChebyshev_sim(self.opt.DATASET, text, text_list,aspect_list, targets)
        text = self.Multihead_attention_afterSimilar(text_after_weight_distribution, text_after_weight_distribution,text_after_weight_distribution)

        spanlen = max([len(item) for item in span_indices])
        tmp = torch.zeros(text_indices.size(0), spanlen, 4 * self.opt.hidden_dim).float().to(self.opt.device)
        tmp1 = torch.zeros(text_indices.size(0), spanlen, 2 * self.opt.hidden_dim).float().to(self.opt.device)
        for i, spans in enumerate(span_indices):
            for j, span in enumerate(spans):
                tmp[i, j], _ = torch.max(text[i, span[0]:span[1]], -2)
                tmp1[i, j] = torch.sum(x[i, span[0]:span[1]], -2)
        #        x=tmp[:,0,:]
        #        maskas=length2mask(torch.Tensor([len(item) for item in span_indices]).long().cuda(),spanlen)#b,span
        #        x=torch.matmul(torch.softmax(torch.matmul(tmp[:,:,:],hout.unsqueeze(-1)).squeeze(-1)+(1-maskas)*-1e20,-1).unsqueeze(-2),tmp[:,:,:]).squeeze(-2)#b,span
        #        x=self.linear1(x)
        #        x1=tmp1[:,0,:]
        ##        x1=torch.matmul(torch.softmax(torch.matmul(tmp1[:,:,:],hout.unsqueeze(-1)).squeeze(-1)+(1-maskas)*-1e20,-1).unsqueeze(-2),tmp1[:,:,:]).squeeze(-2)
        #        x1=self.linear2(x1)
        ##        _, (x, _) = self.text_lstm1(tmp, torch.Tensor([len(item) for item in span_indices]).long().cuda())#b,h
        ##        x = F.relu(self.gc1(self.position_weight(text_out, aspect_double_idx, text_len, aspect_len), adj))
        ##        x = F.relu(self.gc2(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        ##        x = self.mask(x, aspect_double_idx)
        ##        x=torch.transpose(x, 0, 1)
        ##        x=x.reshape(x.size(0),-1)
        #        masked=length2mask(text_len,max_len)
        #        for i,spans in enumerate(span_indices):
        #            for j,span in enumerate(spans):
        #                masked[i,span[0]:span[1]]=0
        #        masked=(1-masked)*-1e20
        #        masked=masked.unsqueeze(-2)
        #        alpha_mat = torch.matmul(x.unsqueeze(1), text_out.transpose(1, 2))
        #        self.alpha= F.softmax(masked+alpha_mat.sum(1, keepdim=True), dim=2)
        #        x = torch.matmul(self.alpha, text_out).squeeze(1) # batch_size x 2*hidden_dim
        ##        x,self.alpha =self.mul1(x,text_out,masked)
        ##        print(x.size())
        ##        x1,self.alpha1 =self.mul2(hout,text_out,length2mask(text_len,max_len))
        ##        x = torch.matmul(self.alpha, text_out).squeeze(1) # batch_size x 2*hidden_dim
        #        alpha_mat = torch.matmul(x1.unsqueeze(1), text_out.transpose(1, 2))
        #        self.alpha1 = F.softmax(masked+alpha_mat.sum(1, keepdim=True), dim=2)
        #        x1 = torch.matmul(self.alpha1, text_out).squeeze(1) # batch_size x 2*hidden_dim
        #        output = self.fc(torch.cat([tmp[:,0,:],tmp1[:,0,:]],-1))

        #print('tmp:',tmp.shape,';tmp1:',tmp1.shape)

        # '''similar for SCI'''
        #text_noHete = torch.cat([text_noHete,text_noHete],dim=1)
        #text_noHete = self.Multihead_attention_afterNoHeteCat(text_noHete,text_noHete,text_noHete)
        # text = torch.cat([text,text_noHete],dim=1)
        from models.weight_distribution_with_similar2 import weight_distribution_useChebyshev_sim as weight_distribution_useChebyshev_sim2
        tmp_after_similar = weight_distribution_useChebyshev_sim2(self.opt.DATASET, tmp, text_list,aspect_list, targets)
        tmp_after_similar_after_att = self.Multihead_attention_afterSimilar2(tmp_after_similar, tmp_after_similar,tmp_after_similar)

        # tmp1_after_similar = weight_distribution_useChebyshev_sim2(self.opt.DATASET, tmp1, text_list, aspect_list,
        #                                                           targets)
        # tmp1_after_similar_after_att = self.Multihead_attention_afterSimilar3(tmp1_after_similar, tmp1_after_similar,
        #                                                                       tmp1_after_similar)

        # output = self.fc(torch.cat([hout, tmp_after_similar_after_att[:, 0, :], tmp1_after_similar_after_att[:, 0, :]], -1))# tmp: torch.Size([32, 1, 400]) ;tmp1: torch.Size([32, 1, 200])
        output = self.fc(torch.cat([hout, tmp_after_similar_after_att[:, 0, :], tmp1[:, 0, :]], -1))# tmp: torch.Size([32, 1, 400]) ;tmp1: torch.Size([32, 1, 200])
        # output = self.fc(torch.cat([hout, tmp[:, 0, :], tmp1[:, 0, :]], -1))# tmp: torch.Size([32, 1, 400]) ;tmp1: torch.Size([32, 1, 200])
        #        output=self.fc(oss)
        #        output = self.fc1(torch.nn.functional.relu(self.fc(torch.cat([tmp[:,0,:],tmp1[:,0,:]],-1))))
        #        output = self.fc1(torch.nn.functional.relu(self.fc(torch.cat([x],-1))))
        #        print(output.size())
        return output