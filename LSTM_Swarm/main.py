from data_process import nn_seq_us
from args import args_parser
from model import train ,test
import os

args = args_parser()
path = os.path.abspath(os.path.dirname(os.getcwd()))

file_name = 'train20240825.csv'
LSTM_PATH = path + '/LSTM_Swarm/model/train20240825.pkl'

dtr, val, dte, m, n = nn_seq_us(args,args.batch_size,file_name)
train(args, dtr, val, LSTM_PATH)
test(args, dte, LSTM_PATH, m, n)