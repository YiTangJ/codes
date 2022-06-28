# -*- coding: utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys
sys.path.append(os.path.dirname(sys.path[0]))
import math
import argparse
import random
import numpy
import torch
import datetime
print('troch_cuda:', torch.cuda.is_available())
import torch.nn as nn

from bucket_iterator import BucketIterator
from sklearn import metrics
from data_utils import ABSADatesetReader
from data_utils import *
from models import LSTM, ASCNN, ASGCN,ASBIGCN
from transformers import BertTokenizer, BertModel
from optimization import BertAdam
from transformers.optimization import AdamW, WarmupLinearSchedule
import pickle
import json
# from DGEDT.bucket_iterator import BucketIterator
# from DGEDT.data_utils import *
# from DGEDT.models import LSTM, ASCNN, ASGCN,ASBIGCN
# from DGEDT.transformers import BertTokenizer, BertModel
# from DGEDT.transformers.optimization import AdamW, WarmupLinearSchedule
# from DGEDT.optimization import BertAdam

#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print('加载BERT-Base-uncsed模型')
bert_model= BertModel.from_pretrained('bert-base-uncased')
print('BERT-Base-uncsed模型加载完毕')
class Instructor:
    def __init__(self, opt):
        self.opt = opt
        print('opt.dataset:', opt.dataset, '\nopt.device:', opt.device)
        '''
        print('load file:', opt.dataset + '_datas.pkl')
        '''
        absa_dataset=pickle.load(open(opt.dataset+'_datas.pkl', 'rb'))
        opt.edge_size=len(absa_dataset.edgevocab)
#        self.train_data_loader = BucketIterator(data=absa_dataset.train_data, other=absa_other.train_data, batch_size=opt.batch_size, shuffle=True)
        self.train_data_loader = BucketIterator(data=absa_dataset.train_data, batch_size=opt.batch_size, shuffle=True)
        self.test_data_loader = BucketIterator(data=absa_dataset.test_data, batch_size=opt.batch_size, shuffle=False)

        # 获取本数据集对应的URI字典
        jsObj = json.load(open('URIDict/{}/alltext_URIDict.json'.format(self.opt.dataset)))
        self.conceptURI = jsObj  # 用于整个模型查找的ConceptNet的URL存储
        # GI情感词典
        GI_sentimentLexicon = open('./datasets/GI_sentimentLexicon.txt', 'r')
        GI_sentiment_lines = GI_sentimentLexicon.readlines()
        GI_sentimentLexicon.close()
        self.GISet = set()
        for item in GI_sentiment_lines:
            GI_sentiment_word = item.strip('\n').split('\t')[0].lower()
            self.GISet.add(GI_sentiment_word)
        from datasets.senticnet6.senticnet6_polarity import senticnet
        self.senticnetDict = senticnet()
        #ws+et
        from utils_HGAT import load_data_ws,load_data_et
        HGAT_data_ws_train = load_data_ws('./datasets/{0}/{0}_ws_et/HGAT/model_data/'.format(self.opt.DATASET), self.opt.DATASET, self.opt.DATATYPE, 'train')
        HGAT_data_et_train = load_data_et('./datasets/{0}/{0}_ws_et/HGAT/model_data/'.format(self.opt.DATASET), self.opt.DATASET, self.opt.DATATYPE, 'train')
        HGAT_data_ws_test = load_data_ws('./datasets/{0}/{0}_ws_et/HGAT/model_data/'.format(self.opt.DATASET), self.opt.DATASET, self.opt.DATATYPE, 'test')
        HGAT_data_et_test = load_data_et('./datasets/{0}/{0}_ws_et/HGAT/model_data/'.format(self.opt.DATASET), self.opt.DATASET, self.opt.DATATYPE, 'test')
        self.adj_HGAT_data_ws_train, self.features_HGAT_data_ws_train, self.labels_HGAT_data_ws_train, idx_train_ori_HGAT_data_ws, _, _ ,_ = HGAT_data_ws_train
        self.adj_HGAT_data_et_train, self.features_HGAT_data_et_train, self.labels_HGAT_data_et_train, idx_train_ori_HGAT_data_et, _, _ ,_ = HGAT_data_et_train
        self.adj_HGAT_data_ws_test, self.features_HGAT_data_ws_test, self.labels_HGAT_data_ws_test, _, idx_test_ori_HGAT_data_ws, _ ,_ = HGAT_data_ws_test
        self.adj_HGAT_data_et_test, self.features_HGAT_data_et_test, self.labels_HGAT_data_et_test, _, idx_test_ori_HGAT_data_et, _ ,_ = HGAT_data_et_test

        self.model = opt.model_class(absa_dataset.embedding_matrix, opt, self.conceptURI, self.GISet,
                                     self.senticnetDict,self.adj_HGAT_data_ws_train, self.features_HGAT_data_ws_train, self.labels_HGAT_data_ws_train,\
                                        self.adj_HGAT_data_et_train, self.features_HGAT_data_et_train, self.labels_HGAT_data_et_train,\
                                        self.adj_HGAT_data_ws_test, self.features_HGAT_data_ws_test, self.labels_HGAT_data_ws_test,\
                                        self.adj_HGAT_data_et_test, self.features_HGAT_data_et_test, self.labels_HGAT_data_et_test).to(opt.device)
        # self.model = opt.model_class(absa_dataset.embedding_matrix, opt).to(opt.device)
        self._print_args()
        self.global_f1 = 0.

        if torch.cuda.is_available():
            print('cuda memory allocated:', torch.cuda.memory_allocated(device=opt.device.index))

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer,optimizer1):
        max_test_acc = 0
        max_test_f1 = 0
        global_step = 0
        continue_not_increase = 0
        for epoch in range(self.opt.num_epoch): # opt.num_epoch=100
            print('>' * 100)
            print('epoch: ', epoch+1)
            n_correct, n_total = 0, 0
            increase_flag = False
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                # sample_batched就是self.train_data_loader中每一项batch内容
                global_step += 1

                self.model.train()
                optimizer1.zero_grad()
                inputs = [sample_batched[col].to(self.opt.device) if type(sample_batched[col])!=list else sample_batched[col] for col in self.opt.inputs_cols]
                # targets = aspect集合
                targets = sample_batched['polarity'].to(self.opt.device)
                # 计算输出时，对所有的节点都进行计算
                outputs = self.model(inputs,targets)
                loss = criterion(outputs, targets) # criterion就是交叉熵
                # 反向求导
                loss.backward()
#                optimizer.step()
                # 更新所有的参数
                optimizer1.step()
                if global_step % self.opt.log_step == 0:
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total

                    test_acc, test_f1 = self._evaluate_acc_f1()
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc
                    if test_f1 > max_test_f1:
                        increase_flag = True
                        max_test_f1 = test_f1
                        if self.opt.save and test_f1 > self.global_f1:
                            self.global_f1 = test_f1
                            # torch.save(self.model.state_dict(), 'tyj/state_dict_Isomer_and_similar/'+self.opt.model_name+'_'+self.opt.dataset+'.pkl')
                            print('>>> best model saved.')
                    print('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, test_f1: {:.4f}'.format(loss.item(), train_acc, test_acc, test_f1))
            if increase_flag == False:
                continue_not_increase += 1
                if continue_not_increase >= 4:
                    print('early stop.')
                    break
            else:
                continue_not_increase = 0    
        return max_test_acc, max_test_f1

    def _evaluate_acc_f1(self):
        # switch model to evaluation mode
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                t_inputs = [t_sample_batched[col].to(opt.device) if type(t_sample_batched[col])!=list else t_sample_batched[col] for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['polarity'].to(opt.device)
                t_outputs = self.model(t_inputs,t_targets,trained=False)

                n_test_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_test_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return test_acc, f1
    def _evaluate_acc_f1withmore(self):
        # switch model to evaluation mode
        self.model.eval()
        import os
        if not os.path.exists('../tyj/results/'+self.opt.model_name+'re'+self.opt.dataset):
            os.makedirs('../tyj/results/'+self.opt.model_name+'re'+self.opt.dataset)
            os.makedirs('../tyj/results/'+self.opt.model_name+'re'+self.opt.dataset+'/right')
            os.makedirs('../tyj/results/'+self.opt.model_name+'re'+self.opt.dataset+'/false')
        n_test_correct, n_test_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        allresults=[]
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                t_inputs = [t_sample_batched[col].to(opt.device) if type(t_sample_batched[col])!=list else t_sample_batched[col] for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['polarity'].to(opt.device)
                t_outputs = self.model(t_inputs)

                n_test_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_test_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)
                for i in range(t_sample_batched['polarity'].size(0)):
                    tmpdict={}
                    tmpdict['truelabel']=t_targets[i].cpu().data.tolist()
                    tmpdict['predictlabel']=torch.argmax(t_outputs, -1)[i].cpu().data.tolist()
                    tmpdict['isright']=tmpdict['truelabel']==tmpdict['predictlabel']
                    tmpdict['text']=t_sample_batched['text'][i]
                    tmpdict['aspect']=t_sample_batched['aspect'][i]
#                    tmplen=len(t_sample_batched['text'][i])
#                    tmpdict['att']=self.model.alpha[i].cpu().data.numpy()[0][:tmplen]
#                    tmpdict['att1']=self.model.alpha1[i].cpu().data.numpy()[0][:tmplen]
#                    adjs=self.model.attss
#                    tmpdict['adj']=adjs[0][i].cpu().data.numpy()[:tmplen,:tmplen]
#                    tmpdict['adj1']=adjs[1][i].cpu().data.numpy()[:tmplen,:tmplen]
#                    tmpdict['adj2']=adjs[2][i].cpu().data.numpy()[:tmplen,:tmplen]
#                    tmpdict['adj3']=adjs[-1][i].cpu().data.numpy()[:tmplen,:tmplen]
                    if not tmpdict['isright']:
                        allresults.append(tmpdict)
        import pickle
        pickle.dump(allresults, open('../tyj/results/'+self.opt.model_name+'re'+self.opt.dataset+'/hh.result', 'wb'))
        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return test_acc, f1
    def run(self, repeats=1):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
        
        # if not os.path.exists('log/'):
        #     os.mkdir('log/')
        #
        # f_out = open('log/'+self.opt.model_name+'_'+self.opt.dataset+'_val.txt', 'w', encoding='utf-8')
        if not os.path.exists('tyj/log_Isomer_and_similar/'):
            os.mkdir('tyj/log_Isomer_and_similar/')

        f_out = open('tyj/log_Isomer_and_similar/'+self.opt.model_name+'_'+self.opt.dataset+'_val.txt', 'w', encoding='utf-8')

        max_test_acc_avg = 0
        max_test_f1_avg = 0
        for i in range(repeats): # repeats=1
            print('repeat: ', (i+1))
            f_out.write('repeat: '+str(i+1) + ' ')
            self._reset_params()
            self.model.bert=bert_model # 这一句在ASBIGCN的forward函数中self.bert使用
            self.model.to(self.opt.device)

            if self.opt.mode=="train":
                parameters=[p for name,p in self.model.named_parameters() if 'bert' in name]
                parameters=[p for name,p in self.model.named_parameters()]
                named_params=[(name,p) for name,p in self.model.named_parameters()]
                optimizer1= torch.optim.Adam(parameters, lr=0.00005, weight_decay=0.00001)
                optimizer_grouped_parameters = [
                    {'params': parameters, 'weight_decay': 0.01}
                    ]
                no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
                optimizer_grouped_parameters = [
                    {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                    {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                    ]
                if self.opt.usebert:
                    # 本文是这个
                    optimizer1 = BertAdam(optimizer_grouped_parameters, lr=0.00005, warmup=0.1, t_total=self.train_data_loader.batch_len*20)
                else:
                    optimizer1 =  torch.optim.Adam(parameters, lr=0.001, weight_decay=0.00001)
                    print('not use bert')
                max_test_acc, max_test_f1 = self._train(criterion, optimizer,optimizer1)
                print('max_test_acc: {0}     max_test_f1: {1}'.format(max_test_acc, max_test_f1))
                f_out.write('max_test_acc: {0}, max_test_f1: {1}'.format(max_test_acc, max_test_f1))
                max_test_acc_avg += max_test_acc
                max_test_f1_avg += max_test_f1
                print('#' * 100)
                print("max_test_acc_avg:", max_test_acc_avg / repeats)
                print("max_test_f1_avg:", max_test_f1_avg / repeats)
                # torch.save(self.model.state_dict(),self.opt.model_name+'_'+self.opt.dataset+'.pth')
                f_out.close()     
            else:
                self.model.load_state_dict(
                    torch.load('tyj/state_dict_asgcn_and_hete/' + self.opt.model_name + '_' + self.opt.dataset + '.pkl'))
                # self.model.load_state_dict(torch.load('state_dict/'+self.opt.model_name+'_'+self.opt.dataset+'.pkl'))
                test_acc, test_f1 = self._evaluate_acc_f1withmore()
                print("max_test_acc_avg:", test_acc / repeats)
                print("max_test_f1_avg:", test_f1 / repeats)
                f_out.close()     



if __name__ == '__main__':
    startnow = datetime.datetime.now()
    print("开始时间是: ", startnow.strftime("%Y-%m-%d %H:%M:%S"))
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='asbi', type=str)
    # parser.add_argument('--dataset', default='semeval14_lap', type=str, help='twitter, rest14, semeval14_lap, semeval15, rest16')
    parser.add_argument('--dataset', default='rest15', type=str, help='twitter, rest14, semeval14_lap, rest15, rest16')
    parser.add_argument('--DATASET', default='semeval15', type=str, help='twitter, rest14, semeval14_lap, semeval15, rest16')
    parser.add_argument('--DATATYPE', default='restaurant', type=str, help='twitter, rest14, semeval14_lap, semeval15, rest16')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--l2reg', default=0.00001, type=float)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int) # 唐艺嘉本机cuda OOM # 服务器可以
    # parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=100, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int) # positive & negative & neutral
    parser.add_argument('--save', default=True, type=bool)
    parser.add_argument('--seed', default=776, type=int)
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--usebert', default=True, type=bool)
#    parser.add_argument('--mode', default='test', type=str)
    parser.add_argument('--mode', default='train', type=str)
    opt = parser.parse_args()

    model_classes = {
        'lstm': LSTM,
        'ascnn': ASCNN,
        'asgcn': ASGCN,
        'astcn': ASGCN,
        'asbi': ASBIGCN, # 本文用的
    }
    input_colses = {
        'lstm': ['text_indices'],
        'ascnn': ['text_indices', 'aspect_indices', 'left_indices'],
        'asgcn': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],
        'astcn': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],
        'asbi': ['text_indices', 'span_indices', 'tran_indices', 'dependency_graph', 'dependency_graph1',
                 'dependency_graph2', 'dependency_graph3','aspect','text'],  # 本文用的
    }

    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_, # 本文用的
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001——————————本文用的
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name] # ASBIGCN
    opt.inputs_cols = input_colses[opt.model_name] # ['text_indices', 'span_indices', 'tran_indices','dependency_graph','dependency_graph1','dependency_graph2','dependency_graph3']
    opt.initializer = initializers[opt.initializer] # torch.nn.init.xavier_uniform_
    opt.optimizer = optimizers[opt.optimizer] # torch.optim.Adam
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)
    # 指定生成随机数的种子，从而每次生成的随机数都是相同的，通过设定随机数种子的好处是，使模型初始化的可学习参数相同，从而使每次的运行结果可以复现。
    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    ins = Instructor(opt)
    ins.run()
    endnow = datetime.datetime.now()
    print("结束时间是: ", endnow.strftime("%Y-%m-%d %H:%M:%S"))