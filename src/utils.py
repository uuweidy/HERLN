"""
Utility functions for link prediction
Most code is adapted from authors' implementation of RGCN link prediction:
https://github.com/MichSchli/RelationPrediction

"""
import numpy as np
import torch
import dgl
from tqdm import tqdm
import src.knowledge_graph as knwlgrh
from collections import defaultdict
import csv
import os

#######################################################################
#
# Utility function for building training and testing graphs
#
#######################################################################

def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices


#TODO filer by groud truth in the same time snapshot not all ground truth
def sort_and_rank_time_filter(batch_a, batch_r, score, target, total_triplets):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    for i in range(len(batch_a)):
        ground = indices[i]
    indices = indices[:, 1].view(-1)
    return indices


def sort_and_rank_filter(batch_a, batch_r, score, target, all_ans):
    for i in range(len(batch_a)):
        ans = target[i]
        b_multi = list(all_ans[batch_a[i].item()][batch_r[i].item()])
        ground = score[i][ans]
        score[i][b_multi] = 0
        score[i][ans] = ground
    _, indices = torch.sort(score, dim=1, descending=True)  # indices : [B, number entity]
    indices = torch.nonzero(indices == target.view(-1, 1))  # indices : [B, 2] 第一列递增， 第二列表示对应的答案实体id在每一行的位置
    indices = indices[:, 1].view(-1)
    return indices


def filter_score(test_triples, score, all_ans):
    if all_ans is None:
        return score
    test_triples = test_triples.cpu()
    for _, triple in enumerate(test_triples):
        h, r, t = triple
        ans = list(all_ans[h.item()][r.item()])
        ans.remove(t.item())
        ans = torch.LongTensor(ans)
        score[_][ans] = -10000000  #
    return score


def filter_t(triplets_to_filter, target_h, target_r, target_t, num_entities):
    target_h, target_r, target_t = int(target_h), int(target_r), int(target_t)
    filtered_t = []

    # Do not filter out the test triplet, since we want to predict on it
    if (target_h, target_r, target_t) in triplets_to_filter:
        triplets_to_filter.remove((target_h, target_r, target_t))
    # Do not consider an object if it is part of a triplet to filter
    for t in range(num_entities):
        if (target_h, target_r, t) not in triplets_to_filter:
            filtered_t.append(t)
    return torch.LongTensor(filtered_t)

def get_filtered_rank(num_entity, score, h, r, t, test_size, triplets_to_filter):
    """ Perturb object in the triplets
    """
    num_entities = num_entity
    ranks = []

    for idx in range(test_size):
        target_h = h[idx]
        target_r = r[idx]
        target_t = t[idx]
        filtered_t = filter_t(triplets_to_filter, target_h, target_r, target_t, num_entities)
        target_t_idx = int((filtered_t == target_t).nonzero())
        _, indices = torch.sort(score[idx][filtered_t], descending=True)
        rank = int((indices == target_t_idx).nonzero())
        ranks.append(rank)
    return torch.LongTensor(ranks)

def calc_filtered_mrr(num_entity, score, train_triplets, valid_triplets, test_triplets):
    with torch.no_grad():
        h = test_triplets[:, 0]
        r = test_triplets[:, 1]
        t = test_triplets[:, 2]
        test_size = test_triplets.shape[0]

        train_triplets = np.concatenate(train_triplets, axis=0)
        valid_triplets = np.concatenate(valid_triplets, axis=0)
        train_triplets = torch.Tensor([[quad[0], quad[1], quad[2]] for quad in train_triplets])
        valid_triplets = torch.Tensor([[quad[0], quad[1], quad[2]] for quad in valid_triplets])
        test_triplets = torch.Tensor([[quad[0], quad[1], quad[2]] for quad in test_triplets])
        #print(train_triplets.shape, valid_triplets.shape, test_triplets.shape)

        triplets_to_filter = torch.cat([train_triplets, valid_triplets, test_triplets]).tolist()

        triplets_to_filter = {tuple(triplet) for triplet in triplets_to_filter}

        ranks = get_filtered_rank(num_entity, score, h, r, t, test_size, triplets_to_filter)

        ranks += 1 # change to 1-indexed

        mrr = torch.mean(1.0 / ranks.float())

    return mrr.item(), ranks


def filter_score_r(test_triples, score, all_ans):
    if all_ans is None:
        return score
    test_triples = test_triples.cpu()
    for _, triple in enumerate(test_triples):
        h, r, t = triple
        ans = list(all_ans[h.item()][t.item()])
        # print(h, r, t)
        # print(ans)
        ans.remove(r.item())
        ans = torch.LongTensor(ans)
        score[_][ans] = -10000000  #
    return score


def r2e(triplets, num_rels):
    src, rel, dst = triplets.transpose()
    # get all relations
    uniq_r = np.unique(rel)
    uniq_r = np.concatenate((uniq_r, uniq_r+num_rels))
    # generate r2e
    r_to_e = defaultdict(set)   #包含了不同类型的边连接的起点
    for j, (src, rel, dst) in enumerate(triplets):
        r_to_e[rel].add(src)
        r_to_e[rel+num_rels].add(src)
    r_len = []
    e_idx = []
    idx = 0
    for r in uniq_r:
        r_len.append((idx,idx+len(r_to_e[r])))      #不同类型的边的起点的角标的起始和终止值
        e_idx.extend(list(r_to_e[r]))               #按照边的类型排列好的起点
        idx += len(r_to_e[r])
    return uniq_r, r_len, e_idx


def comp_deg_norm(g):
        in_deg = g.in_degrees(range(g.number_of_nodes())).float()
        in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
        norm = 1.0 / in_deg
        return norm


def build_sub_graph(num_nodes, num_rels, triples, idx, use_cuda, gpu):
    """
    :param node_id: node id in the large graph
    :param num_rels: number of relation
    :param src: relabeled src id
    :param rel: original rel id
    :param dst: relabeled dst id
    :param idx: snapshot index
    :param use_cuda:
    :return:
    """
    src, rel, dst = triples.transpose()
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel = np.concatenate((rel, rel + num_rels))
    time = [idx] * len(rel)

    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    g.ndata.update({'id': node_id, 'norm': norm.view(-1, 1)})
    g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
    g.edata['type'] = torch.LongTensor(rel)
    g.edata['time'] = torch.LongTensor(time)

    # uniq_r, r_len, r_to_e = r2e(triples, num_rels)
    # g.uniq_r = uniq_r
    # g.r_to_e = r_to_e
    # g.r_len = r_len
    # if use_cuda:
    #     g.to(gpu)
    #     g.r_to_e = torch.from_numpy(np.array(r_to_e))

    return g


def merge_graphs(num_nodes, graphs, use_cuda=False, gpu=0):
    #处理graphs长度为0或者为1的情况
    if len(graphs) == 0:
        return dgl.DGLGraph()
    elif len(graphs) == 1:
        return graphs[0]
    
    new_graph = dgl.DGLGraph()
    for i,g in enumerate(graphs):
        if i==0:
            src = g.edges()[0]
            dst = g.edges()[1]
            rel = g.edata['type']
            time = g.edata['time']
        else:
            src = torch.cat((src, g.edges()[0]))
            dst = torch.cat((dst, g.edges()[1]))
            rel = torch.cat((rel, g.edata['type']))
            time = torch.cat((time, g.edata['time']))
    max_time = torch.max(time)
    #print(src, rel, dst)
    new_graph.add_nodes(num_nodes)
    new_graph.add_edges(src, dst)
    norm = comp_deg_norm(new_graph)
    node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    new_graph.ndata.update({'id':node_id, 'norm':norm.view(-1, 1)})
    new_graph.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
    #print(new_graph.nodes())
    #print(new_graph.edges())
    new_graph.edata['type'] = rel
    new_graph.edata['time'] = max_time - time + 1

    # src = src.numpy()
    # dst = dst.numpy()
    # rel = rel.numpy()
    # triplets = np.array([src, dst, rel]).transpose()

    # uniq_r = np.unique(rel)
    # r_to_e = defaultdict(set)
    # for j, (src, rel, dst) in enumerate(triplets):
    #     r_to_e[rel].add(src)
    # r_len = []
    # e_idx = []
    # idx = 0
    # for r in uniq_r:
    #     r_len.append((idx,idx+len(r_to_e[r])))      #不同类型的边的起点的角标的起始和终止值
    #     e_idx.extend(list(r_to_e[r]))               #按照边的类型排列好的起点
    #     idx += len(r_to_e[r])
    # new_graph.uniq_r = uniq_r
    # new_graph.r_to_e = r_to_e
    # new_graph.r_len = r_len
    
    # if use_cuda:
    #     new_graph.to(gpu)
    #     new_graph.r_to_e = torch.from_numpy(np.array(r_to_e))
    #print('merge graph'+str(len(graphs)))
    return new_graph

def build_history_graph(num_nodes, num_rels, history_list, use_cuda, gpu):    
    max_time = len(history_list)
    for idx,triples in enumerate(history_list):
        s, r, o = triples.transpose()
        s, o = np.concatenate((s, o)), np.concatenate((o, s))
        r = np.concatenate((r, r + num_rels))
        t = [max_time - idx + 1] * len(r)
        if idx==0:
            src = s
            dst = o
            rel = r
            time = t
        else:
            src = np.concatenate((src, s))
            dst = np.concatenate((dst, o))
            rel = np.concatenate((rel, r))
            time = np.concatenate((time, t))
    #print(src, rel, dst)
    new_graph = dgl.DGLGraph()
    new_graph.add_nodes(num_nodes)
    new_graph.add_edges(src, dst)
    norm = comp_deg_norm(new_graph)
    node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    new_graph.ndata.update({'id':node_id, 'norm':norm.view(-1, 1)})
    new_graph.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
    new_graph.edata['type'] = torch.LongTensor(rel)
    new_graph.edata['time'] = torch.LongTensor(time)
    # print(new_graph.num_nodes())
    # print(new_graph.num_edges())
    # print(new_graph.edata['type'].shape)
    # print(new_graph.edata['time'].shape)
    return new_graph



def get_total_rank(test_triples, score, all_ans, eval_bz, rel_predict=0):
    num_triples = len(test_triples)
    n_batch = (num_triples + eval_bz - 1) // eval_bz
    rank = []
    filter_rank = []
    for idx in range(n_batch):
        batch_start = idx * eval_bz
        batch_end = min(num_triples, (idx + 1) * eval_bz)
        triples_batch = test_triples[batch_start:batch_end, :]
        score_batch = score[batch_start:batch_end, :]
        if rel_predict==1:
            target = test_triples[batch_start:batch_end, 1]
        elif rel_predict == 2:
            target = test_triples[batch_start:batch_end, 0]
        else:
            target = test_triples[batch_start:batch_end, 2]
        rank.append(sort_and_rank(score_batch, target))

        if rel_predict:
            filter_score_batch = filter_score_r(triples_batch, score_batch, all_ans)
        else:
            filter_score_batch = filter_score(triples_batch, score_batch, all_ans)
        filter_rank.append(sort_and_rank(filter_score_batch, target))

    rank = torch.cat(rank)
    filter_rank = torch.cat(filter_rank)
    rank += 1 # change to 1-indexed
    filter_rank += 1
    np.set_printoptions(suppress=True)
    mrr = torch.mean(1.0 / rank.float())
    filter_mrr = torch.mean(1.0 / filter_rank.float())
    return filter_mrr.item(), mrr.item(), rank, filter_rank


def stat_ranks(rank_list, method):
    hits = [1, 3, 10]
    total_rank = torch.cat(rank_list)

    mrr = torch.mean(1.0 / total_rank.float())
    print("MRR ({}): {:.6f}".format(method, mrr.item()))
    for hit in hits:
        avg_count = torch.mean((total_rank <= hit).float())
        print("Hits ({}) @ {}: {:.6f}".format(method, hit, avg_count.item()))
    return mrr


def flatten(l):
    flatten_l = []
    for c in l:
        if type(c) is list or type(c) is tuple:
            flatten_l.extend(flatten(c))
        else:
            flatten_l.append(c)
    return flatten_l

def UnionFindSet(m, edges):
    """

    :param m:
    :param edges:
    :return: union number in a graph
    """
    roots = [i for i in range(m)]
    rank = [0 for i in range(m)]
    count = m

    def find(member):
        tmp = []
        while member != roots[member]:
            tmp.append(member)
            member = roots[member]
        for root in tmp:
            roots[root] = member
        return member

    for i in range(m):
        roots[i] = i
    # print ufs.roots
    for edge in edges:
        print(edge)
        start, end = edge[0], edge[1]
        parentP = find(start)
        parentQ = find(end)
        if parentP != parentQ:
            if rank[parentP] > rank[parentQ]:
                roots[parentQ] = parentP
            elif rank[parentP] < rank[parentQ]:
                roots[parentP] = parentQ
            else:
                roots[parentQ] = parentP
                rank[parentP] -= 1
            count -= 1
    return count

def append_object(e1, e2, r, d):
    if not e1 in d:
        d[e1] = {}
    if not r in d[e1]:
        d[e1][r] = set()
    d[e1][r].add(e2)


def add_subject(e1, e2, r, d, num_rel):
    if not e2 in d:
        d[e2] = {}
    if not r+num_rel in d[e2]:
        d[e2][r+num_rel] = set()
    d[e2][r+num_rel].add(e1)


def add_object(e1, e2, r, d, num_rel):
    if not e1 in d:
        d[e1] = {}
    if not r in d[e1]:
        d[e1][r] = set()
    d[e1][r].add(e2)


def load_all_answers(total_data, num_rel):
    # store subjects for all (rel, object) queries and
    # objects for all (subject, rel) queries
    all_subjects, all_objects = {}, {}
    for line in total_data:
        s, r, o = line[: 3]
        add_subject(s, o, r, all_subjects, num_rel=num_rel)
        add_object(s, o, r, all_objects, num_rel=0)
    return all_objects, all_subjects


def load_all_answers_for_filter(total_data, num_rel, rel_p=False):
    # store subjects for all (rel, object) queries and
    # objects for all (subject, rel) queries
    def add_relation(e1, e2, r, d):
        if not e1 in d:
            d[e1] = {}
        if not e2 in d[e1]:
            d[e1][e2] = set()
        d[e1][e2].add(r)

    all_ans = {}
    for line in total_data:
        s, r, o = line[: 3]
        if rel_p:
            add_relation(s, o, r, all_ans)
            add_relation(o, s, r + num_rel, all_ans)
        else:
            add_subject(s, o, r, all_ans, num_rel=num_rel)
            add_object(s, o, r, all_ans, num_rel=0)
    return all_ans


def load_all_answers_for_time_filter(total_data, num_rels, num_nodes, rel_p=False):
    all_ans_list = []
    all_snap = split_by_time(total_data)
    for snap in all_snap:
        all_ans_t = load_all_answers_for_filter(snap, num_rels, rel_p)
        all_ans_list.append(all_ans_t)

    # output_label_list = []
    # for all_ans in all_ans_list:
    #     output = []
    #     ans = []
    #     for e1 in all_ans.keys():
    #         for r in all_ans[e1].keys():
    #             output.append([e1, r])
    #             ans.append(list(all_ans[e1][r]))
    #     output = torch.from_numpy(np.array(output))
    #     output_label_list.append((output, ans))
    # return output_label_list
    return all_ans_list

def split_by_time(data):
    snapshot_list = []
    snapshot = []
    snapshots_num = 0
    latest_t = 0
    for i in range(len(data)):
        t = data[i][3]
        train = data[i]
        # latest_t表示读取的上一个三元组发生的时刻，要求数据集中的三元组是按照时间发生顺序排序的
        if latest_t != t:  # 同一时刻发生的三元组
            # show snapshot
            latest_t = t
            if len(snapshot):
                snapshot_list.append(np.array(snapshot).copy())
                snapshots_num += 1
            snapshot = []
        snapshot.append(train[:3])
    # 加入最后一个shapshot
    if len(snapshot) > 0:
        snapshot_list.append(np.array(snapshot).copy())
        snapshots_num += 1

    union_num = [1]
    nodes = []
    rels = []
    for snapshot in snapshot_list:
        uniq_v, edges = np.unique((snapshot[:,0], snapshot[:,2]), return_inverse=True)  # relabel
        uniq_r = np.unique(snapshot[:,1])
        edges = np.reshape(edges, (2, -1))
        nodes.append(len(uniq_v))
        rels.append(len(uniq_r)*2)
    print("# Sanity Check:  ave node num : {:04f}, ave rel num : {:04f}, snapshots num: {:04d}, max edges num: {:04d}, min edges num: {:04d}, max union rate: {:.4f}, min union rate: {:.4f}"
          .format(np.average(np.array(nodes)), np.average(np.array(rels)), len(snapshot_list), max([len(_) for _ in snapshot_list]), min([len(_) for _ in snapshot_list]), max(union_num), min(union_num)))
    return snapshot_list


def slide_list(snapshots, k=1):
    """
    :param k: padding K history for sequence stat
    :param snapshots: all snapshot
    :return:
    """
    k = k  # k=1 需要取长度k的历史，在加1长度的label
    if k > len(snapshots):
        print("ERROR: history length exceed the length of snapshot: {}>{}".format(k, len(snapshots)))
    for _ in tqdm(range(len(snapshots)-k+1)):
        yield snapshots[_: _+k]



def load_data(dataset, bfs_level=3, relabel=False):
    if dataset in ['aifb', 'mutag', 'bgs', 'am']:
        return knwlgrh.load_entity(dataset, bfs_level, relabel)
    elif dataset in ['FB15k', 'wn18', 'FB15k-237']:
        return knwlgrh.load_link(dataset)
    elif dataset in ['ICEWS18', 'ICEWS14', "GDELT", "SMALL", "ICEWS14s", "ICEWS05-15","YAGO",
                     "WIKI",'DD_test']:
        return knwlgrh.load_from_local("../data", dataset)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))

def construct_snap(test_triples, num_nodes, num_rels, final_score, topK):
    sorted_score, indices = torch.sort(final_score, dim=1, descending=True)
    top_indices = indices[:, :topK]
    predict_triples = []
    for _ in range(len(test_triples)):
        for index in top_indices[_]:
            h, r = test_triples[_][0], test_triples[_][1]
            if r < num_rels:
                predict_triples.append([test_triples[_][0], r, index])
            else:
                predict_triples.append([index, r-num_rels, test_triples[_][0]])

    # 转化为numpy array
    predict_triples = np.array(predict_triples, dtype=int)
    return predict_triples

def construct_snap_r(test_triples, num_nodes, num_rels, final_score, topK):
    sorted_score, indices = torch.sort(final_score, dim=1, descending=True)
    top_indices = indices[:, :topK]
    predict_triples = []
    # for _ in range(len(test_triples)):
    #     h, r = test_triples[_][0], test_triples[_][1]
    #     if (sorted_score[_][0]-sorted_score[_][1])/sorted_score[_][0] > 0.3:
    #         if r < num_rels:
    #             predict_triples.append([h, r, indices[_][0]])

    for _ in range(len(test_triples)):
        for index in top_indices[_]:
            h, t = test_triples[_][0], test_triples[_][2]
            if index < num_rels:
                predict_triples.append([h, index, t])
                #predict_triples.append([t, index+num_rels, h])
            else:
                predict_triples.append([t, index-num_rels, h])
                #predict_triples.append([t, index-num_rels, h])

    # 转化为numpy array
    predict_triples = np.array(predict_triples, dtype=int)
    return predict_triples


def dilate_input(input_list, dilate_len):
    dilate_temp = []
    dilate_input_list = []
    for i in range(len(input_list)):
        if i % dilate_len == 0 and i:
            if len(dilate_temp):
                dilate_input_list.append(dilate_temp)
                dilate_temp = []
        if len(dilate_temp):
            dilate_temp = np.concatenate((dilate_temp, input_list[i]))
        else:
            dilate_temp = input_list[i]
    dilate_input_list.append(dilate_temp)
    dilate_input_list = [np.unique(_, axis=0) for _ in dilate_input_list]
    return dilate_input_list

def emb_norm(emb, epo=0.00001):
    x_norm = torch.sqrt(torch.sum(emb.pow(2), dim=1))+epo
    emb = emb/x_norm.view(-1,1)
    return emb

def shuffle(data, labels):
    shuffle_idx = np.arange(len(data))
    np.random.shuffle(shuffle_idx)
    relabel_output = data[shuffle_idx]
    labels = labels[shuffle_idx]
    return relabel_output, labels


def cuda(tensor):
    if tensor.device == torch.device('cpu'):
        return tensor.cuda()
    else:
        return tensor


def soft_max(z):
    t = np.exp(z)
    a = np.exp(z) / np.sum(t)
    return a

def write_args(file_name='args.log', args=None):
    with open(file_name, 'w') as f:
        f.writelines('--------------model message--------------\n')
        f.writelines('encoder: {}\n'.format(args.encoder))
        f.writelines('decoder: {}\n'.format(args.decoder))
        f.writelines('dataset: {}\n'.format(args.dataset))
        f.writelines('--------------exp settings---------------\n')
        f.writelines('seed: {}\n'.format(args.seed))
        f.writelines('epochs: {}\n'.format(args.n_epochs))
        f.writelines('lr: {}\n'.format(args.lr))
        f.writelines('evaluate epochs: {}\n'.format(args.evaluate_every))
        f.writelines('-------------hyperparameter--------------\n')
        f.writelines('hidden dim: {}\n'.format(args.n_hidden))
        f.writelines('num of layers: {}\n'.format(args.n_layers))
        f.writelines('train history length: {}\n'.format(args.train_history_len))
        f.writelines('test history length: {}\n'.format(args.test_history_len))
        f.writelines('emb update weight: {}\n'.format(args.weight))
        f.writelines('task weight: {}\n'.format(args.task_weight))
        f.writelines('attention threshold theta: {}\n'.format(args.theta))
        f.writelines('dropout: {}\n'.format(args.dropout))
        f.writelines('-------------all parameters--------------\n')
        argsDict = args.__dict__
        for eachArg, value in argsDict.items():
            f.writelines(eachArg+':'+str(value)+'\n')

def write_output(result, mrr_list, file_name='output.csv'):
    hits = [1, 3, 10]
    output_list = []
    for i in range(len(result)):
        rank_list = result[i]
        mrr = mrr_list[i]
        temp_list = [mrr.item()]
        total_rank = torch.cat(rank_list)
        for hit in hits:
            avg_count = torch.mean((total_rank <= hit).float())
            temp_list.append(avg_count.item())
        output_list.append(temp_list)
    output_list = list(np.round(np.array(output_list), 6))
    with open(file_name, 'a', newline='') as f:
        writer = csv.writer(f)
        for row in output_list:
            writer.writerow(row)
    
def analyse_class(dataset):
    path = '../data/'+dataset
    class2node = defaultdict(list)
    node2class = []
    node_with_no_edge = []
    with open(path+'/train.csv') as f:
        csv_file = csv.DictReader(f)
        for row in csv_file:
            class2node[int(row['modularity_class'])].append(int(row['Label']))
            node2class.append(int(row['modularity_class']))
            if int(row['degree']) == 0:
                node_with_no_edge.append(int(row['Label']))
    return class2node, node2class

def class_graph(class2node):
    # 这里社区内部是全连接的
    src = []
    dst = []
    for c in class2node.keys():
        nodes = class2node[c]
        for n in nodes:
            src.extend([n]*len(nodes))
            dst.extend(nodes)
    g = dgl.DGLGraph()
    g.add_edges(src, dst)
    return g

def new_class_graph(node2class, train_list):
    src = []
    dst = []
    edges = set()
    for snapshot in train_list:
        for fact in snapshot:
            s,r,o = fact[0],fact[1],fact[2]
            if node2class[s] == node2class[o]:
                edges.add((s,o))
    
    for e in edges:
        s, o = e[0], e[1]
        src.append(s)
        dst.append(o)
    g = dgl.DGLGraph()
    g.add_nodes(len(node2class))
    g.add_edges(src, dst)
    g = dgl.add_self_loop(g)
    return g