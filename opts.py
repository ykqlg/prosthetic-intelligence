import argparse

parser = argparse.ArgumentParser(description='Parser for all the training options')
parser.add_argument('--winstep', default=0.01, type=float)
parser.add_argument('--numcep', default=13, type=int)
parser.add_argument('--nfilt', default=26, type=int)
parser.add_argument('--nfft', default=512, type=int)
parser.add_argument('--ceplifter', default=22, type=int)
parser.add_argument('--concat', action='store_true')  
parser.add_argument('--no-concat', dest='concat', action='store_false')
parser.add_argument('--dynamic', action='store_true')
parser.add_argument('--no-dynamic', dest='dynamic', action='store_false')


parser.add_argument('--label_file_path', default='./label_file.csv')
parser.add_argument('--test_size', default=0.2, type=float)
parser.add_argument('--random_state', default=42, type=int)
parser.add_argument('--fs', default=1330, type=int)
parser.add_argument('--forward', default=0.5, type=float)
parser.add_argument('--backward', default=0.7, type=float)

args = parser.parse_args()