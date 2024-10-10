from data_process import nn_seq_us, nn_seq_us_num
from args import args_parser
from model import train ,test
import os
import time
from tqdm import tqdm
from predPosB2error import predPosB2error
args = args_parser()
path = os.path.abspath(os.path.dirname(os.getcwd()))

train_file_name = ['predPosTrain20240801.csv', 'predPosTrain20240806.csv', 'predPosTrain20240811.csv',
                   'predPosTrain20240816.csv', 'predPosTrain20240821.csv', 'predPosTrain20240826.csv',
                   'predPosTrain20240831.csv']
# train_file_name = 'predPosTrain20240825_20240831.csv'
test_file_name = 'predPosTrain20240905.csv'
LSTM_PATH = path + '/LSTM_Swarm_predPos/model/predPosTrain20240825_20240831_7.pkl'

# dtr, val, dte, m, n, dte_wl = nn_seq_us(args, args.batch_size, train_file_name, test_file_name)
dtr, val, dte, m, n = nn_seq_us_num(args, args.batch_size, train_file_name, test_file_name)
start_time = time.time()
train(args, dtr, val, LSTM_PATH)
print("训练耗时: {:.2f}秒".format(time.time() - start_time))
# for (seq, target) in tqdm(dte):
#     traindata = seq[0,:,:]
#     traindata = traindata.unsqueeze(0)
#     error = predPosB2error(traindata,LSTM_PATH)
test(args, dte, LSTM_PATH, m, n)