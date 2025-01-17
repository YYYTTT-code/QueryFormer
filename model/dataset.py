import torch
from torch.utils.data import Dataset
import numpy as np
import json
import pandas as pd
import sys, os
from collections import deque
from .database_util import formatFilter, formatJoin, TreeNode, filterDict2Hist
from .database_util import *

# 训练时的加载数据集 类
class PlanTreeDataset(Dataset):
    def __init__(self, json_df : pd.DataFrame, train : pd.DataFrame, encoding, hist_file, card_norm, cost_norm, to_predict, table_sample):

        self.table_sample = table_sample
        self.encoding = encoding
        self.hist_file = hist_file
        
        self.length = len(json_df)
        # train = train.loc[json_df['id']]
        
        # nodes =[由json形式表示的plan]，和我们index selection用的一样
        nodes = [json.loads(plan)['Plan'] for plan in json_df['json']]
        self.cards = [node['Actual Rows'] for node in nodes]
        self.costs = [json.loads(plan)['Execution Time'] for plan in json_df['json']]
        
        # 将label进行正则化
        self.card_labels = torch.from_numpy(card_norm.normalize_labels(self.cards))
        self.cost_labels = torch.from_numpy(cost_norm.normalize_labels(self.costs))
        
        self.to_predict = to_predict
        if to_predict == 'cost':
            self.gts = self.costs
            self.labels = self.cost_labels
        elif to_predict == 'card':
            self.gts = self.cards
            self.labels = self.card_labels
        elif to_predict == 'both': ## try not to use, just in case
            self.gts = self.costs
            self.labels = self.cost_labels
        else:
            raise Exception('Unknown to_predict type')
            
        idxs = list(json_df['id'])
        
        # 解析查询计划树
        self.treeNodes = [] ## for mem collection
        self.collated_dicts = [self.js_node2dict(i,node) for i,node in zip(idxs, nodes)]

    def js_node2dict(self, idx, node):
        # 遍历一遍计划树，解析出树结构，并且以TreeNode形式，组合起来，保留树结构
        treeNode = self.traversePlan(node, idx, self.encoding)
        
        # 将树进行遍历成列表的形式，解析tree transformer所需的相邻表、高度表等信息
        _dict = self.node2dict(treeNode)
        collated_dict = self.pre_collate(_dict)
        
        self.treeNodes.clear()
        del self.treeNodes[:]

        return collated_dict

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        
        return self.collated_dicts[idx], (self.cost_labels[idx], self.card_labels[idx])

    def old_getitem(self, idx):
        return self.dicts[idx], (self.cost_labels[idx], self.card_labels[idx])
      
    ## pre-process first half of old collator
    # 将plan的node pad到最长数目30，如果不在这补，就得在model的forward处补，可能是因为attention-bias不太好补吧
    def pre_collate(self, the_dict, max_node = 30, rel_pos_max = 20):
        # print(the_dict['features'])
        # 将该树的features，补到至少30个节点长
        x = pad_2d_unsqueeze(the_dict['features'], max_node)
        # print(x)
        N = len(the_dict['features'])
        # attn_bias维度，相比于节点数+1，应该是为了Supernode
        attn_bias = torch.zeros([N+1,N+1], dtype=torch.float)
        
        edge_index = the_dict['adjacency_list'].t()
        if len(edge_index) == 0:
            shortest_path_result = np.array([[0]])
            path = np.array([[0]])
            adj = torch.tensor([[0]]).bool()
        else:
            adj = torch.zeros([N,N], dtype=torch.bool)
            adj[edge_index[0,:], edge_index[1,:]] = True
            
            shortest_path_result = floyd_warshall_rewrite(adj.numpy())
        
        rel_pos = torch.from_numpy((shortest_path_result)).long()
        # print(rel_pos.shape) # torch.Size([5, 5])
        # print(rel_pos)
        # 猜测：是将两个节点间距离大于rel_pos_max的，就设置为负无穷，当做没链接
        attn_bias[1:, 1:][rel_pos >= rel_pos_max] = float('-inf')
        
        attn_bias = pad_attn_bias_unsqueeze(attn_bias, max_node + 1)
        rel_pos = pad_rel_pos_unsqueeze(rel_pos, max_node)

        heights = pad_1d_unsqueeze(the_dict['heights'], max_node)
        # 此时的attn_bias除了0就是-inf
        # print(attn_bias)
        
        # rel_pos给出的是任意两个节点间的最短路径，不可达设置为60+1，可达时设置为距离+1，对于不存在的点设置为0
        # print(rel_pos.shape) # torch.Size([1, 30, 30])
        # print(rel_pos)
        return {
            'x' : x,
            'attn_bias': attn_bias,
            'rel_pos': rel_pos,
            'heights': heights
        }


    def node2dict(self, treeNode):
        # 将树进行遍历成列表的形式，解析tree transformer所需的相邻表、高度表等信息
        adj_list, num_child, features = self.topo_sort(treeNode)
        heights = self.calculate_height(adj_list, len(features))

        return {
            'features' : torch.FloatTensor(features),
            'heights' : torch.LongTensor(heights),
            'adjacency_list' : torch.LongTensor(np.array(adj_list)),
          
        }
    
    def topo_sort(self, root_node):
#        nodes = []
        adj_list = [] #from parent to children
        num_child = []
        features = []

        toVisit = deque()
        toVisit.append((0,root_node))
        next_id = 1
        while toVisit:
            idx, node = toVisit.popleft()
#            nodes.append(node)
            features.append(node.feature)
            num_child.append(len(node.children))
            for child in node.children:
                toVisit.append((next_id,child))
                adj_list.append((idx,next_id))
                next_id += 1
        
        return adj_list, num_child, features
    
    # 跟我们一样，也是从json，用字符串解析出相关信息
    def traversePlan(self, plan, idx, encoding): # bfs accumulate plan
        # 首先将nodeType进行encoding
        # encoding是比我们用的one-hot更好的表示方法，不仅短，还带有意义
        nodeType = plan['Node Type']
        typeId = encoding.encode_type(nodeType)
        card = None #plan['Actual Rows']
        filters, alias = formatFilter(plan)
        join = formatJoin(plan)
        joinId = encoding.encode_join(join)
        filters_encoded = encoding.encode_filters(filters, alias)
        
        root = TreeNode(nodeType, typeId, filters, card, joinId, join, filters_encoded)
        
        self.treeNodes.append(root)

        if 'Relation Name' in plan:
            root.table = plan['Relation Name']
            root.table_id = encoding.encode_table(plan['Relation Name'])
        root.query_id = idx
        
        root.feature = node2feature(root, encoding, self.hist_file, self.table_sample)
        #    print(root)
        if 'Plans' in plan:
            for subplan in plan['Plans']:
                subplan['parent'] = plan
                node = self.traversePlan(subplan, idx, encoding)
                node.parent = root
                root.addChild(node)
        return root

    def calculate_height(self, adj_list,tree_size):
        if tree_size == 1:
            return np.array([0])

        adj_list = np.array(adj_list)
        node_ids = np.arange(tree_size, dtype=int)
        node_order = np.zeros(tree_size, dtype=int)
        uneval_nodes = np.ones(tree_size, dtype=bool)

        parent_nodes = adj_list[:,0]
        child_nodes = adj_list[:,1]

        n = 0
        while uneval_nodes.any():
            uneval_mask = uneval_nodes[child_nodes]
            unready_parents = parent_nodes[uneval_mask]

            node2eval = uneval_nodes & ~np.isin(node_ids, unready_parents)
            node_order[node2eval] = n
            uneval_nodes[node2eval] = False
            n += 1
        return node_order 


# 生成该节点的特征即向量
def node2feature(node, encoding, hist_file, table_sample):
    # 这里将filter进行的编码，由3-num_filter可知，本方法假设filter的长度不会超过3个，原来如此，也很妥协的方法

    # type, join, filter123, mask123
    # 1, 1, 3x3 (9), 3
    # TODO: add sample (or so-called table)
    num_filter = len(node.filterDict['colId'])
    pad = np.zeros((3,3-num_filter))
    filts = np.array(list(node.filterDict.values())) #cols, ops, vals
    ## 3x3 -> 9, get back with reshape 3,3
    filts = np.concatenate((filts, pad), axis=1).flatten() 
    mask = np.zeros(3)
    mask[:num_filter] = 1
    type_join = np.array([node.typeId, node.join])
    
    hists = filterDict2Hist(hist_file, node.filterDict, encoding)


    # table, bitmap, 1 + 1000 bits
    table = np.array([node.table_id])
    if node.table_id == 0:
        sample = np.zeros(1000)
    else:
        sample = table_sample[node.query_id][node.table]
    
    #return np.concatenate((type_join,filts,mask))
    return np.concatenate((type_join, filts, mask, hists, table, sample))
