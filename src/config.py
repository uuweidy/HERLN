import argparse

parser = argparse.ArgumentParser(description='REGCN')

parser.add_argument("--gpu", type=int, default=-1,
					help="gpu")
parser.add_argument("--batch-size", type=int, default=1,
					help="batch-size")
parser.add_argument("-d", "--dataset", type=str, required=True,
					help="dataset to use")
parser.add_argument("--n-epochs", type=int, default=50,
					help="number of minimum training epochs on each time step")
parser.add_argument("--test", action='store_true', default=False,
					help="load stat from dir and directly test")
parser.add_argument("--multi-step", action='store_true', default=False,
					help="do multi-steps inference without ground truth")
parser.add_argument("--topk", type=int, default=10,
					help="choose top k entities as results when do multi-steps without ground truth")
parser.add_argument("--relation-evaluation", action='store_true', default=False,
					help="save model accordding to the relation evalution")
parser.add_argument("--seed", type=int, default=None,
					help="random by default")

# configuration for encoder RGCN stat
parser.add_argument("--weight", type=float, default=0.5,
					help="weight of current entity embedding in evolve")
parser.add_argument("--task-weight", type=float, default=0.7,
					help="weight of entity prediction task")
parser.add_argument("--theta", type=float, default=1,
					help="threshold of attention score in RGCN")

parser.add_argument("--encoder", type=str, default="hrgcn",
					help="method of encoder")
parser.add_argument("--aggregation", type=str, default="none",
					help="method of aggregation")
parser.add_argument("--dropout", type=float, default=0.2,
					help="dropout probability")
parser.add_argument("--skip-connect", action='store_true', default=False,
					help="whether to use skip connect in a RGCN Unit")
parser.add_argument("--n-hidden", type=int, default=200,
					help="number of hidden units")
parser.add_argument("--opn", type=str, default="corr",
					help="opn of compgcn")

parser.add_argument("--n-bases", type=int, default=-1,
					help="number of weight blocks for each relation")
parser.add_argument("--n-basis", type=int, default=100,
					help="number of basis vector for compgcn")
parser.add_argument("--n-layers", type=int, default=2,
					help="number of propagation rounds")
parser.add_argument("--self-loop", action='store_true', default=True,
					help="perform layer normalization in every layer of gcn ")
parser.add_argument("--layer-norm", action='store_true', default=False,
					help="perform layer normalization in every layer of gcn ")
parser.add_argument("--relation-prediction", action='store_true', default=False,
					help="add relation prediction loss")
parser.add_argument("--entity-prediction", action='store_true', default=False,
					help="add entity prediction loss")
parser.add_argument("--split_by_relation", action='store_true', default=False,
					help="do relation prediction")
parser.add_argument("--use-logger", action='store_true', default=False, help="output to log file")

parser.add_argument("--raw_input", action='store_true', default=False,
					help="disable the attention machsim and hyperparameter theta")
parser.add_argument("--use-last-epoch", action='store_true', default=False,
					help="use last epoch model to test")

# configuration for stat training

parser.add_argument("--lr", type=float, default=0.001,
					help="learning rate")
parser.add_argument("--grad-norm", type=float, default=1.0,
					help="norm to clip gradient to")

# configuration for evaluating
parser.add_argument("--evaluate-every", type=int, default=1,
					help="perform evaluation every n epochs")

# configuration for decoder
parser.add_argument("--decoder", type=str, default="convtranse",
					help="method of decoder")
parser.add_argument("--input-dropout", type=float, default=0.2,
					help="input dropout for decoder ")
parser.add_argument("--hidden-dropout", type=float, default=0.2,
					help="hidden dropout for decoder")
parser.add_argument("--feat-dropout", type=float, default=0.2,
					help="feat dropout for decoder")

# configuration for sequences stat
parser.add_argument("--train-history-len", type=int, default=10,
					help="history length")
parser.add_argument("--test-history-len", type=int, default=10,
					help="history length for test")

# configuration for optimal parameters
parser.add_argument("--grid-search", action='store_true', default=False,
					help="perform grid search for best configuration")
parser.add_argument("-tune", "--tune", type=str, default="n_hidden,n_layers,dropout,n_bases",
					help="stat to use")
parser.add_argument("--num-k", type=int, default=500,
					help="number of triples generated")

args = parser.parse_args()