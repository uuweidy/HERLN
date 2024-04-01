'''
The code is modified based on RE-GCN: https://github.com/Lee-zix/RE-GCN

'''

from genericpath import isdir
import itertools
import os
import pickle
import random
import sys
from datetime import datetime

import dgl
import numpy as np
import torch
from tqdm import tqdm

sys.path.append("..")
from collections import defaultdict

import torch.nn.modules.rnn

from model.rrgcn import RecurrentRGCN
from src import utils
from src.config import args
from src.hyperparameter_range import hp_range
from src.knowledge_graph import _read_triplets_as_list
from src.utils import build_sub_graph, merge_graphs

# os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(datetime.today().strftime('%Y-%m-%d %H:%M:%S')+':  '+message)

    def flush(self):
        pass


def test(model, history_list, test_list, num_rels, num_nodes, class_g, use_cuda, all_ans_list, all_ans_r_list, path, model_name, mode, pop=True):
    """
    :param model: model used to test
    :param history_list:    all input history snap shot list, not include output label train list or valid list
    :param test_list:   test triple snap shot list
    :param num_rels:    number of relations
    :param num_nodes:   number of nodes
    :param use_cuda:
    :param all_ans_list:     dict used to calculate filter mrr (key and value are all int variable not tensor)
    :param all_ans_r_list:     dict used to calculate filter mrr (key and value are all int variable not tensor)
    :param model_name:
    :param mode
    :return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r
    """
    ranks_raw, ranks_filter, mrr_raw_list, mrr_filter_list = [], [], [], []
    ranks_raw_r, ranks_filter_r, mrr_raw_list_r, mrr_filter_list_r = [], [], [], []
    ranks_unaware, mrr_unaware_list = [], []

    idx = 0
    if mode == "test":
        # test mode: load parameter form file
        if use_cuda:
            checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
        else:
            checkpoint = torch.load(model_name, map_location=torch.device('cpu'))
        print("Load Model name: {}. Using best epoch : {}".format(model_name, checkpoint['epoch']))  # use best stat checkpoint
        print("\n"+"-"*10+"start testing"+"-"*10+"\n")
        if pop:
            checkpoint['state_dict'].pop('rgcn.layers.0.rel_emb')
            if args.n_layers > 1:
                checkpoint['state_dict'].pop('rgcn.layers.1.rel_emb')
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    # do not have inverse relation in test input
    if args.test_history_len == -1:
        input_list = history_list
    else:
        input_list = [snap for snap in history_list[-args.test_history_len:]]

    for time_idx, test_snap in enumerate(tqdm(test_list)):
        history_glist = [build_sub_graph(num_nodes, num_rels, input_list[i], i, use_cuda, args.gpu) for i in range(len(input_list))]
        #history_glist = [utils.build_history_graph(num_nodes, num_rels, input_list, use_cuda, args.gpu)]
        test_triples_input = torch.LongTensor(test_snap).cuda() if use_cuda else torch.LongTensor(test_snap)
        test_triples_input = test_triples_input.to(args.gpu)
        test_triples, final_score, final_r_score = model.predict(history_glist, num_rels, test_triples_input, class_g, use_cuda)

        mrr_filter_snap_r, mrr_snap_r, rank_raw_r, rank_filter_r = utils.get_total_rank(test_triples, final_r_score, all_ans_r_list[time_idx], eval_bz=1000, rel_predict=1)
        mrr_filter_snap, mrr_snap, rank_raw, rank_filter = utils.get_total_rank(test_triples, final_score, all_ans_list[time_idx], eval_bz=1000, rel_predict=0)

        #mrr_unaware_snap, rank_unaware = utils.calc_filtered_mrr(num_nodes, final_score[:len(final_score)//2], history_list, test_list, test_snap)
        mrr_unaware_snap, rank_unaware = utils.calc_filtered_mrr(num_nodes, final_score, history_list, test_list, test_snap)

        # used to global statistic
        ranks_raw.append(rank_raw)
        ranks_filter.append(rank_filter)
        ranks_unaware.append(rank_unaware)
        # used to show slide results
        mrr_raw_list.append(mrr_snap)
        mrr_filter_list.append(mrr_filter_snap)
        mrr_unaware_list.append(mrr_unaware_snap)
        # relation rank
        ranks_raw_r.append(rank_raw_r)
        ranks_filter_r.append(rank_filter_r)
        mrr_raw_list_r.append(mrr_snap_r)
        mrr_filter_list_r.append(mrr_filter_snap_r)

        # reconstruct history graph list
        if args.multi_step:
            if not args.relation_evaluation:    
                predicted_snap = utils.construct_snap(test_triples, num_nodes, num_rels, final_score, args.topk)
            else:
                predicted_snap = utils.construct_snap_r(test_triples, num_nodes, num_rels, final_r_score, args.topk)
            if len(predicted_snap):
                if args.test_history_len != -1:
                    input_list.pop(0)
                input_list.append(predicted_snap)
        else:
            if args.test_history_len != -1:
                input_list.pop(0)
            input_list.append(test_snap)
        idx += 1
    with open(path+'ranks_raw.txt', 'w') as f:
        for ranks in ranks_raw:
            for r in ranks.cpu().numpy().tolist():
                f.writelines(str(r))
                f.writelines('\n')
    with open(path+'ranks_filter.txt', 'w') as f:
        for ranks in ranks_filter:
            for r in ranks.cpu().numpy().tolist():
                f.writelines(str(r))
                f.writelines('\n')
    mrr_raw = utils.stat_ranks(ranks_raw, "raw_ent")
    mrr_filter = utils.stat_ranks(ranks_filter, "filter_ent")
    mrr_unaware = utils.stat_ranks(ranks_unaware, "unaware_ent")
    mrr_raw_r = utils.stat_ranks(ranks_raw_r, "raw_rel")
    mrr_filter_r = utils.stat_ranks(ranks_filter_r, "filter_rel")
    if mode == 'test':
        utils.write_output([ranks_raw,ranks_filter,ranks_raw_r,ranks_filter_r], 
                           [mrr_raw,mrr_filter,mrr_raw_r,mrr_filter_r], 
                           file_name=path+'results.csv')

    return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r


def run_experiment(args, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
    # load configuration for grid search the best configuration
    if n_hidden:
        args.n_hidden = n_hidden
    if n_layers:
        args.n_layers = n_layers
    if dropout:
        args.dropout = dropout
    if n_bases:
        args.n_bases = n_bases

    # load graph data
    print("loading graph data")
    data = utils.load_data(args.dataset)
    train_list = utils.split_by_time(data.train)
    valid_list = utils.split_by_time(data.valid)
    test_list = utils.split_by_time(data.test)

    num_nodes = data.num_nodes
    num_rels = data.num_rels

    all_ans_list_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, False)
    all_ans_list_r_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, True)
    all_ans_list_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, False)
    all_ans_list_r_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, True)
    #model_name = "ICEWS18-uvrgcn-convtranse-ly2-his10-weight0.5-theta1.0-task0.7-dp0.2-hidden200-epochs50-node2graphGate"
    model_name = "{}-{}-{}-ly{}-his{}-weight{}-theta{}-task{}-dp{}-hidden{}-epochs{}-node2graphGate"\
        .format(args.dataset, args.encoder, args.decoder, args.n_layers, args.train_history_len, args.weight, args.theta, args.task_weight, args.dropout, args.n_hidden, args.n_epochs)
    time = datetime.today()
    #time = '{}-{}-{}-{}:{}:{}'.format(time.year, time.month, time.day, time.hour, time.minute, time.second)
    path = '../checkpoints/{}/{}/{}/{}/'.format(time.year, time.month, time.day, time.hour)
    if not os.path.exists(path):
        os.makedirs(path)
    fold_num = sum([os.path.isdir(path+listx) for listx in os.listdir(path)])
    path = path + str(fold_num+1) + '/'
    # 新建文件夹 年/月/日/小时/{文件夹个数}/
    # 一个best model，一个last model，一个args，一个log，
    if not os.path.exists(path):
        os.makedirs(path)
    #path = '../checkpoints/2022/10/15/8/2/'
    utils.write_args(file_name=path+'args.log', args=args)
    if args.test and args.use_last_epoch:
        model_state_file = path + 'last.pt'
    else:
        model_state_file = path + 'best.pt'

    print("Sanity Check: stat name : {}".format(model_name))
    print("Sanity Check: Is cuda available ? {}".format(torch.cuda.is_available()))

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if args.use_logger:
        if args.test:
            sys.stdout = Logger(path+'test.log', sys.stdout)
        else:
            sys.stdout = Logger(path+'train.log', sys.stdout)
    print(args)
    # create stat
    model = RecurrentRGCN(args.decoder,
                          args.encoder,
                        num_nodes,
                        num_rels,
                        args.n_hidden,
                        args.opn,
                        sequence_len=args.train_history_len,
                        num_bases=args.n_bases,
                        num_basis=args.n_basis,
                        num_hidden_layers=args.n_layers,
                        dropout=args.dropout,
                        self_loop=args.self_loop,
                        skip_connect=args.skip_connect,
                        layer_norm=args.layer_norm,
                        input_dropout=args.input_dropout,
                        hidden_dropout=args.hidden_dropout,
                        feat_dropout=args.feat_dropout,
                        aggregation=args.aggregation,
                        weight=args.weight,
                        theta=args.theta, 
                        entity_prediction=args.entity_prediction,
                        relation_prediction=args.relation_prediction,
                        raw_input=args.raw_input,
                        use_cuda=use_cuda,
                        gpu = args.gpu)

    if use_cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    class2node, node2class = utils.analyse_class(args.dataset)
    #class_g = utils.class_graph(class2node)
    class_g = utils.new_class_graph(node2class, train_list)
    update_class_embedding = True
    #class_g = None

    if args.test and os.path.exists(model_state_file):
        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model, 
                                                            train_list+valid_list, 
                                                            test_list, 
                                                            num_rels, 
                                                            num_nodes, 
                                                            class_g, 
                                                            use_cuda, 
                                                            all_ans_list_test, 
                                                            all_ans_list_r_test, 
                                                            path,
                                                            model_state_file,
                                                            mode="test")
    elif args.test and not os.path.exists(model_state_file):
        print("--------------{} not exist, Change mode to train and generate stat for testing----------------\n".format(model_state_file))
    else:
        print("----------------------------------------start training----------------------------------------\n")
        best_mrr = 0
        for epoch in range(args.n_epochs):
            model.train()
            losses = []
            losses_e = []
            losses_r = []
            

            idx = [_ for _ in range(len(train_list))]
            random.shuffle(idx)

            for train_sample_num in tqdm(idx):
                if train_sample_num == 0: continue
                output = train_list[train_sample_num:train_sample_num+1]
                if args.train_history_len == -1:
                    input_list = train_list[0: train_sample_num]
                else:
                    if train_sample_num - args.train_history_len<0:
                        input_list = train_list[0: train_sample_num]
                    else:
                        input_list = train_list[train_sample_num - args.train_history_len:
                                            train_sample_num]
                if update_class_embedding:
                    update_class_embedding = False
                else:
                    class_g = None
                # generate history graph
                history_glist = [build_sub_graph(num_nodes, num_rels, input_list[i], i, use_cuda, args.gpu) for i in range(len(input_list))]
                #history_glist = [utils.build_history_graph(num_nodes, num_rels, input_list, use_cuda, args.gpu)]
                output = [torch.from_numpy(_).long().cuda() for _ in output] if use_cuda else [torch.from_numpy(_).long() for _ in output]
                loss_e, loss_r = model.get_loss(history_glist, output[0], class_g, use_cuda)
                loss = args.task_weight*loss_e + (1-args.task_weight)*loss_r
                losses.append(loss.item())
                losses_e.append(loss_e.item())
                losses_r.append(loss_r.item())

                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
                optimizer.step()
                optimizer.zero_grad()

            print("Epoch {:04d} | Ave Loss: {:.4f} | entity-relation:{:.4f}-{:.4f} Best MRR {:.4f} | Model {} "
                  .format(epoch, np.mean(losses), np.mean(losses_e), np.mean(losses_r), best_mrr, model_name))

            # validation
            if epoch and epoch % args.evaluate_every == 0:
                mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model, 
                                                                    train_list[:], 
                                                                    valid_list[:], 
                                                                    num_rels, 
                                                                    num_nodes, 
                                                                    class_g, 
                                                                    use_cuda, 
                                                                    all_ans_list_valid, 
                                                                    all_ans_list_r_valid, 
                                                                    path,
                                                                    model_state_file,  
                                                                    mode="train")
                
                if not args.relation_evaluation:  # entity prediction evalution
                    if mrr_raw < best_mrr:
                        if epoch >= args.n_epochs:
                            break
                    else:
                        best_mrr = mrr_raw
                        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
                else:
                    if mrr_raw_r < best_mrr:
                        if epoch >= args.n_epochs:
                            break
                    else:
                        best_mrr = mrr_raw_r
                        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model, 
                                                            train_list+valid_list,
                                                            test_list, 
                                                            num_rels, 
                                                            num_nodes, 
                                                            class_g, 
                                                            use_cuda, 
                                                            all_ans_list_test, 
                                                            all_ans_list_r_test, 
                                                            path,
                                                            model_state_file, 
                                                            mode="test", pop=False)
        print('No ground truth testing...')
        args.multi_step = True
        args.topk = 0
        test(model, train_list+valid_list, test_list, num_rels, num_nodes, class_g, use_cuda, all_ans_list_test, all_ans_list_r_test, path, model_state_file, mode="test", pop=False)
        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, path+'last.pt')
    return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r


if __name__ == '__main__':
    print(args)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    if args.grid_search:
        out_log = '{}.{}.gs'.format(args.dataset, args.encoder+"-"+args.decoder)
        o_f = open(out_log, 'w')
        print("** Grid Search **")
        o_f.write("** Grid Search **\n")
        hyperparameters = args.tune.split(',')

        if args.tune == '' or len(hyperparameters) < 1:
            print("No hyperparameter specified.")
            sys.exit(0)
        grid = hp_range[hyperparameters[0]]
        for hp in hyperparameters[1:]:
            grid = itertools.product(grid, hp_range[hp])
        hits_at_1s = {}
        hits_at_10s = {}
        mrrs = {}
        grid = list(grid)
        print('* {} hyperparameter combinations to try'.format(len(grid)))
        o_f.write('* {} hyperparameter combinations to try\n'.format(len(grid)))
        o_f.close()

        for i, grid_entry in enumerate(list(grid)):

            o_f = open(out_log, 'a')

            if not (type(grid_entry) is list or type(grid_entry) is list):
                grid_entry = [grid_entry]
            grid_entry = utils.flatten(grid_entry)
            print('* Hyperparameter Set {}:'.format(i))
            o_f.write('* Hyperparameter Set {}:\n'.format(i))
            signature = ''
            print(grid_entry)
            o_f.write("\t".join([str(_) for _ in grid_entry]) + "\n")
            # def run_experiment(args, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
            mrr, hits, ranks = run_experiment(args, grid_entry[0], grid_entry[1], grid_entry[2], grid_entry[3])
            print("MRR (raw): {:.6f}".format(mrr))
            o_f.write("MRR (raw): {:.6f}\n".format(mrr))
            for hit in hits:
                avg_count = torch.mean((ranks <= hit).float())
                print("Hits (raw) @ {}: {:.6f}".format(hit, avg_count.item()))
                o_f.write("Hits (raw) @ {}: {:.6f}\n".format(hit, avg_count.item()))
    # single run
    else:
        run_experiment(args)
    sys.exit()



