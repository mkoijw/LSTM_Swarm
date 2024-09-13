from data_process import nn_seq_us
from args import args_parser
from model import train ,test
import os
import time

args = args_parser()
path = os.path.abspath(os.path.dirname(os.getcwd()))

train_file_name = 'train20240825_20240826.csv'
test_file_name = 'test20240827.csv'
LSTM_PATH = path + '/LSTM_Swarm/model/train20240825_20240826.pkl'

dtr, val, dte, m, n, dte_wl = nn_seq_us(args, args.batch_size, train_file_name, test_file_name)

start_time = time.time()
# train(args, dtr, val, LSTM_PATH)
print("训练耗时: {:.2f}秒".format(time.time() - start_time))
test(args, dte, LSTM_PATH, m, n, dte_wl)