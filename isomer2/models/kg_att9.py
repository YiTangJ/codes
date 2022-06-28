
'''
唐艺嘉的想法
加入conceptNet和SenticNet和GI
conceptNet用于找text中与方面词相关联的concept
SenticNet用于找text中属于情感极性的词！！！！

'''
import re
import json
import os
import requests
# from datasets.senticnet6.senticnet6_polarity import senticnet
KG_ATT = 0.95 # 设置的权重值
# KG_ATT = 100000 # 设置的权重值
# KG_ATT_sentic = 100000 # 是qlearning对应的值
KG_ATT_sentic = 0.95
# KG_ATT_aspect = -100000 # 是原始论文的Aspect-based Attention Module方法————————把方面词所在位置的值变成-inf=======唐艺嘉变成-100000！！！！
# KG_ATT_aspect = -10000 # 是原始论文的Aspect-based Attention Module方法————————把方面词所在位置的值变成-inf=======唐艺嘉变成-100000！！！！
KG_ATT_aspect = 0 # 是原始论文的Aspect-based Attention Module方法————————把方面词所在位置的值变成-inf=======唐艺嘉变成-100000！！！！
# 根据aspect，找前10个相关性最大的词，


'''
不直接调用api，而是使用提前调用api生成的URI文档
这个函数只对text_entity进行====也就是entity部分的text   &&&&& 对text进行
这个函数是对进入dual-level模型前的text===一开始的text， 进行attention
这个时候的text_entity = torch.Size([32, 76, 200])，意思就是32个text，每个text的长度是200===（[CLS]+words+[SCL]+padding)
同样的第0个index=[CLS]，第一个text的单词要从index=1开始，每个单词的下标要+1！！！！！！！

还多了span_indices=list，长度是32，就是32个text的对应的span的[起始位置,结束位置+1]
span_indices = span_indices_entity: [[[4, 6]], [[0, 1]], [[2, 4]], [[6, 8]], [[1, 3]], [[1, 2]], [[1, 2]], [[5, 6]], [[18, 19]], [[15, 17]], [[6, 8]], [[1, 2]], [[9, 11]], [[1, 3]], [[9, 10]], [[2, 5]], [[3, 4]], [[1, 2]], [[10, 11]], [[17, 18]], [[2, 3]], [[21, 23]], [[10, 11]], [[1, 3]], [[5, 7]], [[2, 4]], [[5, 6]], [[1, 3]], [[1, 2]], [[44, 46]], [[19, 22]], [[10, 11]]]
[4, 6] = ['i', 'highly', 'recommend', 'the', 'sophia', 'pizza', '.']这个句子的方面词起始位置是4=sophia，结束位置是6-1=pizza
也就是说这个句子的方面词是Sophia pizza

这个函数没有！！！！原始论文的Aspect-based Attention Module方法
'''


def KG_att_text_entity2(text_list, aspect_list, text_embedding_list, span_indices, URIDict,GISet,senticnetDict):
    '''
    URIDict里存在的是{'concept':[relatedConcept1,relatedConcept2,...],...}
    '''
    # 先使用ConceptNet
    for index, (textListItem, asepctListItem) in enumerate(zip(text_list, aspect_list)):
        aspect = ''
        if len(asepctListItem) != 1:
            # 方面词不止一个单词的要进行拼接！！！！！！！！！！
            aspectListItemIndex = 0
            for item in asepctListItem:
                aspect = aspect + item
                if aspectListItemIndex != len(asepctListItem) - 1:
                    aspect = aspect + ' '
                aspectListItemIndex = aspectListItemIndex + 1
        else:
            aspect = asepctListItem[0]
        text = textListItem;
        # 先查找方面词的URI是否存在，存在的就查找，不存在就过
        aspect = aspect.replace(' ', '+')  # 将方面词中的空格替换成+
        all_related_concept = []  # 清理过后的和方面词相关的所有ConceptNet中的单词
        if aspect in URIDict:  # 先在URIDict中找方面词的uri，如果有，直接用找到的relatedConceptList
            all_related_concept = URIDict.get(aspect)
        # 根据方面词的uri查找前15个！！！！！有关的ConceptNet中存在的单词

        no_index_in_concept = []
        index_in_concept = []
        if len(all_related_concept) != 0:
            noRelated_index = list(range(len(text)))  # 所有text的下标
            for related_concept in all_related_concept:
                if related_concept in text:
                    # 先查找该concept在text中的所有下标
                    related_concept_index = [i for i, x in enumerate(text) if x == related_concept]
                    for word_index in related_concept_index:
                        '''要对word_index+1才是embedding中对应单词的实际位置，因为每个句子的embedding开头都有[CLS]'''
                        '''还因为text_embedding_list_copy的shape是([32, 1, 400])'''
                        text_embedding_list[index][:, word_index + 1:word_index + 2] = text_embedding_list[index][:,
                                                                                       word_index + 1:word_index + 2] * (1+KG_ATT)  # 对text中相关的单词分配权重
                        noRelated_index.remove(word_index)  # 移除相关concept的下标
            # text中有相关词的，对text剩下的无关单词分配权重
            if len(noRelated_index) != len(text):
                # print('分配权重')
                for noRelatedIndex in noRelated_index:
                    text_embedding_list[index][:, noRelatedIndex + 1:noRelatedIndex + 2] = text_embedding_list[index][:,
                                                                                           noRelatedIndex + 1:noRelatedIndex + 2] * (
                                                                                                   1 - KG_ATT)
            no_index_in_concept = noRelated_index
            index_in_concept = [item for item in list(range(len(text))) if item not in no_index_in_concept]
        # 接着使用SenticNet 和GI情感词典
        noSentic_index = list(range(len(text)))  # 所有text的下标
        for word_index, word in enumerate(text):
            if word in senticnetDict or word in GISet:  # 是SenticNet中的情感词 或者是 GI中的
                '''要对word_index+1才是embedding中对应单词的实际位置，因为每个句子的embedding开头都有[CLS]'''
                '''还因为text_embedding_list_copy的shape是([32, 1, 400])'''
                text_embedding_list[index][:, word_index + 1:word_index + 2] = text_embedding_list[index][:,
                                                                               word_index + 1:word_index + 2] * (
                                                                                      1 + KG_ATT_sentic)  # 情感词增加权重，其他词不变
                noSentic_index.remove(word_index)
        # 給不是Sentic且不是centicnet里的
        if len(noSentic_index) != len(text):
            for noSenticIndex in noSentic_index:
                if noSenticIndex not in index_in_concept: # 不属于concept
                    text_embedding_list[index][:, noSenticIndex + 1:noSenticIndex + 2] = text_embedding_list[index][:,
                                                                                               noSenticIndex + 1:noSenticIndex + 2] * (
                                                                                                       1 - KG_ATT_sentic)


    return text_embedding_list

def KG_att_text_entity2_noConceptNet(text_list, aspect_list, text_embedding_list, span_indices, URIDict,GISet,senticnetDict):
    '''
    URIDict里存在的是{'concept':[relatedConcept1,relatedConcept2,...],...}
    '''
    for index, (textListItem, asepctListItem) in enumerate(zip(text_list, aspect_list)):
        text = textListItem;
        # 接着使用SenticNet 和GI情感词典
        noSentic_index = list(range(len(text)))  # 所有text的下标
        for word_index, word in enumerate(text):
            if word in senticnetDict or word in GISet:  # 是SenticNet中的情感词 或者是 GI中的
                '''要对word_index+1才是embedding中对应单词的实际位置，因为每个句子的embedding开头都有[CLS]'''
                '''还因为text_embedding_list_copy的shape是([32, 1, 400])'''
                # ysembedding = text_embedding_list_copy[index][0][word_index+1].float() + KG_ATT_sentic
                text_embedding_list[index][:, word_index + 1:word_index + 2] = text_embedding_list[index][:,
                                                                               word_index + 1:word_index + 2] * (
                                                                                           1 + KG_ATT_sentic)  # 情感词增加权重，其他词不变
                noSentic_index.remove(word_index)  # 移除相关senticnet单词的下标
        # text中有相关词的，对text剩下的无关单词分配权重
        if len(noSentic_index) != len(text):
            for noSenticIndex in noSentic_index:
                text_embedding_list[index][:, noSenticIndex + 1:noSenticIndex + 2] = text_embedding_list[
                                                                                           index][:,
                                                                                       noSenticIndex + 1:noSenticIndex + 2] * (
                                                                                               1 - KG_ATT_sentic)

    return text_embedding_list
def KG_att_text_entity2_noSenticNet(text_list, aspect_list, text_embedding_list, span_indices, URIDict,GISet,senticnetDict):
    '''
    URIDict里存在的是{'concept':[relatedConcept1,relatedConcept2,...],...}
    '''
    # 先使用ConceptNet
    for index, (textListItem, asepctListItem) in enumerate(zip(text_list, aspect_list)):
        aspect = ''
        if len(asepctListItem) != 1:
            # 方面词不止一个单词的要进行拼接！！！！！！！！！！
            aspectListItemIndex = 0
            for item in asepctListItem:
                aspect = aspect + item
                if aspectListItemIndex != len(asepctListItem) - 1:
                    aspect = aspect + ' '
                aspectListItemIndex = aspectListItemIndex + 1
        else:
            aspect = asepctListItem[0]
        text = textListItem;
        # 先查找方面词的URI是否存在，存在的就查找，不存在就过
        aspect = aspect.replace(' ', '+')  # 将方面词中的空格替换成+
        all_related_concept = []  # 清理过后的和方面词相关的所有ConceptNet中的单词
        if aspect in URIDict:  # 先在URIDict中找方面词的uri，如果有，直接用找到的relatedConceptList
            all_related_concept = URIDict.get(aspect)
        # 根据方面词的uri查找前15个！！！！！有关的ConceptNet中存在的单词
        if len(all_related_concept) != 0:
            noRelated_index = list(range(len(text)))  # 所有text的下标
            for related_concept in all_related_concept:
                if related_concept in text:
                    # 先查找该concept在text中的所有下标
                    # related_concept_index = re.findall(related_concept, text) # re.findall是查找字符串中的
                    related_concept_index = [i for i, x in enumerate(text) if x == related_concept]
                    for word_index in related_concept_index:
                        '''要对word_index+1才是embedding中对应单词的实际位置，因为每个句子的embedding开头都有[CLS]'''
                        '''还因为text_embedding_list_copy的shape是([32, 1, 400])'''
                        '''
                        切片变化：
                        torch.Size([2, 3, 4])
                        a=tensor([[[0., 0., 0., 0.],
                             [0., 0., 0., 0.],
                             [0., 0., 0., 0.]],

                            [[0., 0., 0., 0.],
                             [0., 0., 0., 0.],
                             [0., 0., 0., 0.]]])
                        a[:,0:3,3:4]=a[:,0:3,3:4]+2 对最后一列
                        tensor([[[0., 0., 0., 2.],
                         [0., 0., 0., 2.],
                         [0., 0., 0., 2.]],

                        [[0., 0., 0., 2.],
                         [0., 0., 0., 2.],
                         [0., 0., 0., 2.]]])
                         a[0][0:3,3:4] = a[0][0:3,3:4]+2 对第1个的最后一列
                         tensor([[[0., 0., 0., 4.],
                         [0., 0., 0., 4.],
                         [0., 0., 0., 4.]],

                        [[0., 0., 0., 2.],
                         [0., 0., 0., 2.],
                         [0., 0., 0., 2.]]])
                        '''
                        text_embedding_list[index][:, word_index + 1:word_index + 2] = text_embedding_list[index][:,
                                                                                       word_index + 1:word_index + 2] * (1+KG_ATT)  # 对text中相关的单词分配权重
                        noRelated_index.remove(word_index)  # 移除相关concept的下标
            # text中有相关词的，对text剩下的无关单词分配权重
            if len(noRelated_index) != len(text):
                for noRelatedIndex in noRelated_index:
                    text_embedding_list[index][:, noRelatedIndex + 1:noRelatedIndex + 2] = text_embedding_list[index][:,
                                                                                           noRelatedIndex + 1:noRelatedIndex + 2] * (
                                                                                                       1 - KG_ATT)
    return text_embedding_list
if __name__ == '__main__':
    print()

'''
对进入.gc()和.gc2()前的text和text_entity进行KG——att

将ConceptNet的权重为0.95,=== 对应Concept单词*权重；不对应的单词*（1-权重）
还用到了SenticNet+GI，权重为0.95===== 对应情感词*（1+权重），其他不变
没有原论文的aspect-based attention！！！！！


train_wordSentiment_entityText_dualLevel_att_reinforce_fivePointSeventeen_kg9————对应kg_att9————对应的asgcn_wordSentiment_entityText_dualLevel_att_reinforce_fivePointSeventeen_kg_two_nine
本机cpu——27.5h

开始时间是:  2021-06-22 19:55:16
troch_cuda: False
加载BERT-Base-uncsed模型
load state ok
BERT-Base-uncsed模型加载完毕
开始时间是:  2021-06-26 22:51:52
opt.dataset: semeval15 
opt.device: cpu
load file: rest15_wordSentiment_datas.pkl
n_trainable_params: 1306003, n_nontrainable_params: 18313200
> training arguments:
>>> model_name: asbiHGAT
>>> dataset: semeval15
>>> optimizer: <class 'torch.optim.adam.Adam'>
>>> initializer: <function xavier_uniform_ at 0x000001AA2BD62950>
>>> learning_rate: 0.001
>>> l2reg: 1e-05
>>> num_epoch: 100
>>> batch_size: 32
>>> log_step: 5
>>> embed_dim: 300
>>> hidden_dim: 100
>>> polarities_dim: 3
>>> save: True
>>> seed: 776
>>> device: cpu
>>> usebert: True
>>> mode: train
>>> model_class: <class 'models.asgcn_wordSentiment_entityText_dualLevel_att_reinforce_fivePointSeventeen_kg_two_nine.ASBIGCN'>
>>> inputs_cols: ['text_indices', 'span_indices', 'tran_indices', 'dependency_graph', 'dependency_graph1', 'dependency_graph2', 'dependency_graph3', 'aspect', 'text']
>>> edge_size: 514
repeat:  1
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
epoch:  1
>>> best model saved.
loss: 0.9583, acc: 0.7188, test_acc: 0.7022, test_f1: 0.2780
>>> best model saved.
loss: 0.6207, acc: 0.7344, test_acc: 0.6232, test_f1: 0.3084
loss: 0.9339, acc: 0.7188, test_acc: 0.7574, test_f1: 0.2873
>>> best model saved.
loss: 0.5458, acc: 0.7422, test_acc: 0.6801, test_f1: 0.3795
loss: 0.5692, acc: 0.7500, test_acc: 0.7739, test_f1: 0.3031
loss: 0.3990, acc: 0.7708, test_acc: 0.7188, test_f1: 0.3092
>>> best model saved.
loss: 0.2843, acc: 0.7991, test_acc: 0.8180, test_f1: 0.4538
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
epoch:  2
>>> best model saved.
loss: 0.4105, acc: 0.8750, test_acc: 0.8088, test_f1: 0.5344
>>> best model saved.
loss: 0.4115, acc: 0.8750, test_acc: 0.8860, test_f1: 0.5744
loss: 29.8338, acc: 0.8333, test_acc: 0.8842, test_f1: 0.5677
loss: 0.3440, acc: 0.8516, test_acc: 0.7555, test_f1: 0.5126
loss: 106.1736, acc: 0.7125, test_acc: 0.8051, test_f1: 0.4898
loss: 0.4974, acc: 0.7292, test_acc: 0.7316, test_f1: 0.4676
loss: 0.1737, acc: 0.7589, test_acc: 0.8750, test_f1: 0.5483
loss: 0.1367, acc: 0.7812, test_acc: 0.8695, test_f1: 0.5586
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
epoch:  3
>>> best model saved.
loss: 0.1788, acc: 0.9062, test_acc: 0.8934, test_f1: 0.5825
loss: 0.3824, acc: 0.8906, test_acc: 0.8787, test_f1: 0.5709
loss: 0.3065, acc: 0.9062, test_acc: 0.8842, test_f1: 0.5755
loss: 0.1977, acc: 0.9141, test_acc: 0.7886, test_f1: 0.5027
loss: 0.2966, acc: 0.9125, test_acc: 0.8511, test_f1: 0.5395
loss: 0.3151, acc: 0.9010, test_acc: 0.8051, test_f1: 0.5538
>>> best model saved.
loss: 452.0289, acc: 0.8036, test_acc: 0.9026, test_f1: 0.5857
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
epoch:  4
>>> best model saved.
loss: 0.3987, acc: 0.8750, test_acc: 0.9044, test_f1: 0.5929
loss: 0.2468, acc: 0.9062, test_acc: 0.8603, test_f1: 0.5514
loss: 0.4834, acc: 0.8958, test_acc: 0.8107, test_f1: 0.5487
>>> best model saved.
loss: 0.4300, acc: 0.8828, test_acc: 0.9301, test_f1: 0.6044
loss: 275.9379, acc: 0.7625, test_acc: 0.7463, test_f1: 0.5186
>>> best model saved.
loss: 0.1304, acc: 0.7969, test_acc: 0.8971, test_f1: 0.6153
loss: 65.6078, acc: 0.6964, test_acc: 0.8529, test_f1: 0.5553
loss: 0.1642, acc: 0.7266, test_acc: 0.9246, test_f1: 0.6117
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
epoch:  5
loss: 0.1357, acc: 0.9688, test_acc: 0.8640, test_f1: 0.5586
loss: 0.4028, acc: 0.9375, test_acc: 0.8971, test_f1: 0.5864
loss: 0.5260, acc: 0.9062, test_acc: 0.8621, test_f1: 0.6094
loss: 0.3167, acc: 0.9062, test_acc: 0.9136, test_f1: 0.5978
loss: 0.0973, acc: 0.9250, test_acc: 0.8768, test_f1: 0.6087
loss: 0.0970, acc: 0.9323, test_acc: 0.8732, test_f1: 0.5965
loss: 0.3873, acc: 0.9152, test_acc: 0.8107, test_f1: 0.5130
loss: 0.7406, acc: 0.9016, test_acc: 0.8879, test_f1: 0.5760
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
epoch:  6
loss: 0.1460, acc: 0.9375, test_acc: 0.9173, test_f1: 0.5938
loss: 0.2854, acc: 0.9219, test_acc: 0.8346, test_f1: 0.5374
loss: 0.1762, acc: 0.9375, test_acc: 0.8952, test_f1: 0.5883
loss: 0.0862, acc: 0.9453, test_acc: 0.8713, test_f1: 0.5936
loss: 0.1746, acc: 0.9500, test_acc: 0.8695, test_f1: 0.5597
loss: 73.6433, acc: 0.8125, test_acc: 0.7077, test_f1: 0.5215
>>> best model saved.
loss: 0.1952, acc: 0.8259, test_acc: 0.9081, test_f1: 0.6257
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
epoch:  7
loss: 0.6122, acc: 0.7812, test_acc: 0.8327, test_f1: 0.5641
loss: 0.1686, acc: 0.8594, test_acc: 0.9081, test_f1: 0.5892
loss: 0.3018, acc: 0.8542, test_acc: 0.8401, test_f1: 0.5751
loss: 0.1650, acc: 0.8828, test_acc: 0.8327, test_f1: 0.5781
>>> best model saved.
loss: 0.1444, acc: 0.8938, test_acc: 0.9210, test_f1: 0.6828
loss: 0.1480, acc: 0.9010, test_acc: 0.9210, test_f1: 0.6110
loss: 0.2823, acc: 0.8973, test_acc: 0.8879, test_f1: 0.5681
loss: 0.4254, acc: 0.8867, test_acc: 0.8309, test_f1: 0.5577
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
epoch:  8
loss: 0.2648, acc: 0.8750, test_acc: 0.8713, test_f1: 0.5830
loss: 0.1728, acc: 0.9062, test_acc: 0.8768, test_f1: 0.6099
loss: 0.3575, acc: 0.8958, test_acc: 0.9210, test_f1: 0.6011
loss: 0.3198, acc: 0.8984, test_acc: 0.9081, test_f1: 0.5889
loss: 0.1486, acc: 0.9062, test_acc: 0.8805, test_f1: 0.5629
loss: 0.1837, acc: 0.9062, test_acc: 0.8842, test_f1: 0.5724
loss: 0.2300, acc: 0.9107, test_acc: 0.8309, test_f1: 0.5639
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
epoch:  9
loss: 0.2841, acc: 0.9062, test_acc: 0.8290, test_f1: 0.5889
loss: 0.1803, acc: 0.9062, test_acc: 0.8805, test_f1: 0.6215
loss: 0.2537, acc: 0.8958, test_acc: 0.8401, test_f1: 0.5423
loss: 0.5128, acc: 0.8750, test_acc: 0.8493, test_f1: 0.5466
loss: 0.1600, acc: 0.8875, test_acc: 0.8915, test_f1: 0.5790
loss: 0.1353, acc: 0.8958, test_acc: 0.9283, test_f1: 0.6471
loss: 0.2516, acc: 0.9018, test_acc: 0.8695, test_f1: 0.5592
loss: 0.2345, acc: 0.9023, test_acc: 0.8603, test_f1: 0.6584
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
epoch:  10
loss: 0.2813, acc: 0.8750, test_acc: 0.8511, test_f1: 0.6439
loss: 0.3006, acc: 0.9062, test_acc: 0.8768, test_f1: 0.6337
loss: 0.1303, acc: 0.9271, test_acc: 0.8989, test_f1: 0.6728
>>> best model saved.
loss: 0.4551, acc: 0.8984, test_acc: 0.9062, test_f1: 0.7046
loss: 0.1477, acc: 0.9125, test_acc: 0.8548, test_f1: 0.6188
loss: 0.3045, acc: 0.9115, test_acc: 0.8566, test_f1: 0.6647
loss: 0.1838, acc: 0.9107, test_acc: 0.8824, test_f1: 0.6833
loss: 0.1615, acc: 0.9139, test_acc: 0.8438, test_f1: 0.6321
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
epoch:  11
loss: 0.0605, acc: 0.9688, test_acc: 0.8971, test_f1: 0.6501
loss: 0.4387, acc: 0.9375, test_acc: 0.8768, test_f1: 0.6636
loss: 0.0991, acc: 0.9375, test_acc: 0.8290, test_f1: 0.6448
>>> best model saved.
loss: 0.2195, acc: 0.9297, test_acc: 0.9412, test_f1: 0.8241
loss: 0.0542, acc: 0.9437, test_acc: 0.8272, test_f1: 0.6002
loss: 0.2916, acc: 0.9427, test_acc: 0.8897, test_f1: 0.6553
loss: 0.5530, acc: 0.9330, test_acc: 0.8860, test_f1: 0.6869
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
epoch:  12
>>> best model saved.
loss: 0.1031, acc: 0.9688, test_acc: 0.9357, test_f1: 0.8379
loss: 0.2447, acc: 0.9219, test_acc: 0.9191, test_f1: 0.7916
loss: 0.1653, acc: 0.9271, test_acc: 0.8162, test_f1: 0.6044
>>> best model saved.
loss: 0.4272, acc: 0.8906, test_acc: 0.9357, test_f1: 0.8441
loss: 0.0993, acc: 0.9000, test_acc: 0.9026, test_f1: 0.7052
loss: 0.1102, acc: 0.9115, test_acc: 0.8713, test_f1: 0.6704
loss: 0.0658, acc: 0.9241, test_acc: 0.9485, test_f1: 0.8228
loss: 0.1324, acc: 0.9258, test_acc: 0.7463, test_f1: 0.5607
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
epoch:  13
loss: 0.0491, acc: 1.0000, test_acc: 0.8897, test_f1: 0.8167
loss: 0.2471, acc: 0.9531, test_acc: 0.8915, test_f1: 0.8146
loss: 975.0824, acc: 0.7083, test_acc: 0.8401, test_f1: 0.6765
loss: 0.0699, acc: 0.7812, test_acc: 0.9485, test_f1: 0.8307
loss: 0.0903, acc: 0.8187, test_acc: 0.8438, test_f1: 0.6761
loss: 46.2296, acc: 0.7969, test_acc: 0.8676, test_f1: 0.7527
>>> best model saved.
loss: 405.6143, acc: 0.6920, test_acc: 0.9246, test_f1: 0.8450
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
epoch:  14
loss: 0.1679, acc: 0.9688, test_acc: 0.9393, test_f1: 0.8077
loss: 7.8556, acc: 0.8125, test_acc: 0.9026, test_f1: 0.8397
loss: 0.1568, acc: 0.8542, test_acc: 0.7996, test_f1: 0.6230
>>> best model saved.
loss: 0.1611, acc: 0.8672, test_acc: 0.9504, test_f1: 0.8601
loss: 0.0371, acc: 0.8938, test_acc: 0.9007, test_f1: 0.7443
loss: 250.0468, acc: 0.7604, test_acc: 0.8474, test_f1: 0.6492
loss: 0.0513, acc: 0.7946, test_acc: 0.9449, test_f1: 0.8437
loss: 0.0874, acc: 0.8203, test_acc: 0.9301, test_f1: 0.8358
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
epoch:  15
loss: 0.3483, acc: 0.9375, test_acc: 0.8897, test_f1: 0.7224
loss: 0.0204, acc: 0.9688, test_acc: 0.8585, test_f1: 0.6678
loss: 0.0531, acc: 0.9688, test_acc: 0.9026, test_f1: 0.8318
loss: 0.0528, acc: 0.9766, test_acc: 0.9449, test_f1: 0.8169
loss: 0.1064, acc: 0.9688, test_acc: 0.8897, test_f1: 0.8137
loss: 0.0328, acc: 0.9740, test_acc: 0.9154, test_f1: 0.8547
>>> best model saved.
loss: 0.0346, acc: 0.9732, test_acc: 0.9467, test_f1: 0.9010
loss: 0.0522, acc: 0.9754, test_acc: 0.8971, test_f1: 0.8335
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
epoch:  16
>>> best model saved.
loss: 0.0777, acc: 0.9688, test_acc: 0.9706, test_f1: 0.9342
loss: 0.1158, acc: 0.9531, test_acc: 0.8603, test_f1: 0.7015
loss: 0.0610, acc: 0.9583, test_acc: 0.9265, test_f1: 0.8217
loss: 0.1312, acc: 0.9531, test_acc: 0.9154, test_f1: 0.8400
loss: 0.1323, acc: 0.9563, test_acc: 0.8934, test_f1: 0.7986
loss: 0.0444, acc: 0.9583, test_acc: 0.8787, test_f1: 0.7312
loss: 0.0870, acc: 0.9598, test_acc: 0.9062, test_f1: 0.8314
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
epoch:  17
loss: 0.0751, acc: 0.9688, test_acc: 0.9522, test_f1: 0.9032
loss: 0.1096, acc: 0.9688, test_acc: 0.9596, test_f1: 0.9293
loss: 0.1056, acc: 0.9583, test_acc: 0.9449, test_f1: 0.8857
loss: 0.0222, acc: 0.9688, test_acc: 0.9118, test_f1: 0.7645
loss: 0.0576, acc: 0.9750, test_acc: 0.8548, test_f1: 0.7141
loss: 0.0714, acc: 0.9688, test_acc: 0.8989, test_f1: 0.7508
loss: 0.0616, acc: 0.9688, test_acc: 0.9154, test_f1: 0.7847
loss: 0.2213, acc: 0.9609, test_acc: 0.9044, test_f1: 0.7804
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
epoch:  18
loss: 0.1845, acc: 0.8750, test_acc: 0.9062, test_f1: 0.8126
loss: 0.2132, acc: 0.8594, test_acc: 0.9485, test_f1: 0.9192
loss: 0.0994, acc: 0.8958, test_acc: 0.9357, test_f1: 0.8564
loss: 0.0713, acc: 0.9141, test_acc: 0.9191, test_f1: 0.7608
loss: 0.0866, acc: 0.9187, test_acc: 0.9522, test_f1: 0.8715
loss: 0.1725, acc: 0.9167, test_acc: 0.9430, test_f1: 0.8671
loss: 102.3734, acc: 0.8929, test_acc: 0.9081, test_f1: 0.7744
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
epoch:  19
loss: 0.1712, acc: 0.9688, test_acc: 0.8621, test_f1: 0.6937
loss: 0.1865, acc: 0.9531, test_acc: 0.9614, test_f1: 0.9306
loss: 0.0335, acc: 0.9688, test_acc: 0.9228, test_f1: 0.8912
loss: 0.0288, acc: 0.9766, test_acc: 0.9007, test_f1: 0.7172
loss: 0.0551, acc: 0.9750, test_acc: 0.8934, test_f1: 0.7399
loss: 0.0582, acc: 0.9792, test_acc: 0.7923, test_f1: 0.6169
loss: 0.0363, acc: 0.9821, test_acc: 0.9688, test_f1: 0.9228
loss: 0.1165, acc: 0.9727, test_acc: 0.8897, test_f1: 0.7102
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
epoch:  20
loss: 0.1332, acc: 0.9375, test_acc: 0.8290, test_f1: 0.6794
loss: 0.2272, acc: 0.9219, test_acc: 0.8971, test_f1: 0.7875
loss: 0.1180, acc: 0.9375, test_acc: 0.8805, test_f1: 0.8290
loss: 0.0944, acc: 0.9375, test_acc: 0.9081, test_f1: 0.7800
loss: 0.2091, acc: 0.9313, test_acc: 0.8989, test_f1: 0.7693
>>> best model saved.
loss: 0.0351, acc: 0.9427, test_acc: 0.9724, test_f1: 0.9539
loss: 0.1022, acc: 0.9420, test_acc: 0.9596, test_f1: 0.9115
loss: 0.0946, acc: 0.9426, test_acc: 0.9577, test_f1: 0.9097
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
epoch:  21
loss: 0.1781, acc: 0.9062, test_acc: 0.9026, test_f1: 0.8492
loss: 0.1000, acc: 0.9375, test_acc: 0.8290, test_f1: 0.6907
loss: 0.0800, acc: 0.9479, test_acc: 0.9081, test_f1: 0.8167
loss: 0.0792, acc: 0.9531, test_acc: 0.9118, test_f1: 0.7336
loss: 0.1385, acc: 0.9500, test_acc: 0.8529, test_f1: 0.6776
loss: 0.1798, acc: 0.9531, test_acc: 0.9246, test_f1: 0.8724
loss: 0.0386, acc: 0.9554, test_acc: 0.8971, test_f1: 0.7408
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
epoch:  22
loss: 0.1080, acc: 0.9375, test_acc: 0.8879, test_f1: 0.7107
loss: 0.2843, acc: 0.9062, test_acc: 0.9430, test_f1: 0.8904
loss: 0.0967, acc: 0.9167, test_acc: 0.8750, test_f1: 0.7429
loss: 0.1698, acc: 0.9219, test_acc: 0.9007, test_f1: 0.7817
loss: 0.0612, acc: 0.9375, test_acc: 0.8860, test_f1: 0.7334
loss: 0.1796, acc: 0.9375, test_acc: 0.8658, test_f1: 0.6938
loss: 0.1096, acc: 0.9330, test_acc: 0.9081, test_f1: 0.7463
loss: 0.0826, acc: 0.9414, test_acc: 0.9522, test_f1: 0.8671
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
epoch:  23
loss: 0.1243, acc: 0.9375, test_acc: 0.9081, test_f1: 0.8396
loss: 0.0120, acc: 0.9688, test_acc: 0.8456, test_f1: 0.6955
loss: 0.0323, acc: 0.9792, test_acc: 0.9614, test_f1: 0.9214
loss: 0.1093, acc: 0.9766, test_acc: 0.9596, test_f1: 0.9390
loss: 0.3531, acc: 0.9625, test_acc: 0.8143, test_f1: 0.6561
loss: 0.1502, acc: 0.9479, test_acc: 0.9559, test_f1: 0.9282
loss: 0.1240, acc: 0.9464, test_acc: 0.7721, test_f1: 0.5959
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
epoch:  24
loss: 0.0800, acc: 0.9375, test_acc: 0.9559, test_f1: 0.8833
loss: 364.8678, acc: 0.5156, test_acc: 0.9540, test_f1: 0.9117
loss: 0.0601, acc: 0.6667, test_acc: 0.9485, test_f1: 0.8859
loss: 0.0629, acc: 0.7422, test_acc: 0.8621, test_f1: 0.7162
loss: 0.0615, acc: 0.7875, test_acc: 0.9449, test_f1: 0.8824
loss: 0.1245, acc: 0.8125, test_acc: 0.9430, test_f1: 0.8650
loss: 0.1613, acc: 0.8259, test_acc: 0.8474, test_f1: 0.6699
loss: 0.0321, acc: 0.8477, test_acc: 0.8952, test_f1: 0.7674
early stop.
max_test_acc: 0.9724264705882353     max_test_f1: 0.9539133473095737
####################################################################################################
max_test_acc_avg: 0.9724264705882353
max_test_f1_avg: 0.9539133473095737
结束时间是:  2021-06-28 02:25:41

Process finished with exit code 0





'''

