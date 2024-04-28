import argparse

parser = argparse.ArgumentParser(description='Parser for all the training options')
parser.add_argument('--winstep', default=0.01, type=float)
parser.add_argument('--winlen', default=0.025, type=float)
parser.add_argument('--numcep', default=13, type=int)
parser.add_argument('--nfilt', default=26, type=int)
parser.add_argument('--nfft', default=512, type=int)
parser.add_argument('--ceplifter', default=22, type=int)
parser.add_argument('--dynamic', action='store_true')
parser.add_argument('--no-dynamic', dest='dynamic', action='store_false')
parser.add_argument('--test_only', action='store_true')


parser.add_argument('--train_label_file', default='./data/white_cup_user1_label.csv')
parser.add_argument('--test_label_file', default='./data/white_cup_user2_label.csv')
parser.add_argument('--tests_label_file', nargs='+',default=[])
parser.add_argument('--val_label_file', default='./data/white_cup_user1_label.csv')

parser.add_argument('--train_label_file2', default=False)

parser.add_argument('--test_size', default=0.2, type=float)
parser.add_argument('--random_state', default=42, type=int)
parser.add_argument('--fs', default=1330, type=int)
parser.add_argument('--forward', default=0.5, type=float)
parser.add_argument('--backward', default=0.7, type=float)
parser.add_argument('--repeat_num', default=3, type=int)

args = parser.parse_args()