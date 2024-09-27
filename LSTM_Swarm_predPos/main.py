from data_process import nn_seq_us
from args import args_parser
from model import train ,test
import os
import time
from tqdm import tqdm
from predPosB2error import predPosB2error
args = args_parser()
path = os.path.abspath(os.path.dirname(os.getcwd()))

train_file_name = 'predPostrain20240825.csv'
test_file_name = 'predPostrain20240825.csv'
LSTM_PATH = path + '/LSTM_Swarm_predPos/model/predPostrain20240825.pkl'
model_path = r'E:/ai_software/LSTM_Swarm/LSTM_Swarm_predPos/model/predPostrain20240825.pkl'

dtr, val, dte, m, n, dte_wl = nn_seq_us(args, args.batch_size, train_file_name, test_file_name)
start_time = time.time()
train(args, dtr, val, LSTM_PATH)
print("训练耗时: {:.2f}秒".format(time.time() - start_time))
# for (seq, target) in tqdm(dte):
#     traindata = seq[0,:,:]
#     traindata = traindata.unsqueeze(0)
#     error = predPosB2error(traindata,model_path)
test(args, dte, LSTM_PATH, m, n, dte_wl)