"""
This work is created by KunLu. Copyright reserved.
lukun199@gmail.com
19th Feb., 2021

# Inference.py
"""
import os, platform, json, time, pickle, sys, argparse
import torch
from math import log
sys.path.append('./')
from data_loader import Dataset_sentence_test, collate_func
from model import LSTMEncoder, LSTMDecoder, Embeds
from utils import Normlize_tx, Channel, smaple_n_times


_snr = 10
_iscomplex = True
channel_dim = 32

bias=127.
num_e_bits=8
num_m_bits=23

NUM_BITS = 16
def dec2bin(x, bits):
    # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, dtype=torch.int16)
    #print(mask)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()


def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, dtype=torch.int16)
    #print(mask)
    return torch.sum(mask * b, -1)

device = torch.device("cpu:0")
#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # use CPU or GPU
embeds_shared = Embeds(vocab_size=530, num_hidden=32).to(device)
encoder = LSTMEncoder(channel_dim=channel_dim, embedds=embeds_shared).to(device)
decoder = LSTMDecoder(channel_dim=channel_dim, embedds=embeds_shared, vocab_size = 530).to(device)

encoder = encoder.eval()
decoder = decoder.eval()
embeds_shared = embeds_shared.eval()


normlize_layer = Normlize_tx(_iscomplex=_iscomplex)
channel = Channel(_iscomplex=_iscomplex)

def do_test(input_data, encoder, decoder, normlize_layer, channel, len_batch):

    with torch.no_grad():
        print(input_data)
        output, _ = encoder(input_data, len_batch)
        print("encoder :")
        print(output)
        output = normlize_layer.apply(output)
        print("normlize_layer :")
        print(output)
        #output = output*0.625
        #print("float vector convert bit vector :")
        output = output *10000             # x =1234.567 
        output= torch.round(output)
        output = output.clone().type(torch.int16)
        print("float convert int : ")
        print(output)
        
        pred = dec2bin(output, NUM_BITS)
       
        pred1 = pred.squeeze(0)
        print(pred1)
      
        W=1
        x = torch.FloatTensor(32, 16).uniform_(0, 1)
        y = torch.FloatTensor(32, 16).uniform_(1, 2)
        #x[:, 0:1]= 1
        #y[:, 0:1]= 1
        idx = torch.argsort(torch.where(pred1<1,x,pred1))[:, :W] #0 to 1
        #print(idx)
        idx1 = torch.argsort(torch.where(pred1>0,y,pred1),descending=True)[:, :W] #1 to 0
        #print(idx1)
        #idx2 = torch.cat([idx,idx1],dim=1)
        pred1[torch.arange(pred1.size(0)).unsqueeze(-1), idx] = torch.rand(*idx.shape).uniform_(1,2)
        a = torch.FloatTensor([[1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4],[1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6], [1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6],[1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6],[1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6],[1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6],[1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6],[1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6],[1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6],[1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6],#1
                               [1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4],[1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6], [1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6],[1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6],[1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6],[1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6],[1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6],[1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6],[1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6],[1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6],#1
                               [1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4],[1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6], [1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6],[1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6],[1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6],[1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6],[1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6],[1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6],[1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6],[1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6],#1
                               [1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6],[1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6], #0                        
                            ])
        rand_indx = torch.randperm(len(a))
        n = a[rand_indx]
        w = torch.ones(32, 16)
        q = torch.zeros(32, 16)
        pred1 = torch.where(pred1>1,n,pred1)
        #print(pred1)
        pred1 = torch.where(pred1>1.5,q,pred1)
        #print(pred1)
        pred1[torch.arange(pred1.size(0)).unsqueeze(-1), idx1] = torch.rand(*idx1.shape).uniform_(2,3)
        #print(pred1)
        d = torch.FloatTensor([[2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6],[2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6], [2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6], [2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6], [2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6], [2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6], [2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6], [2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6], [2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6], [2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6],#0
                               [2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6],[2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6], [2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6], [2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6], [2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6], [2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6], [2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6], [2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6], [2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6], [2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6],#0
                               [2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6],[2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4], [2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6], [2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6], [2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6], [2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6], [2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6], [2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6], [2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6], [2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6],#1
                               [2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6],[2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4], #1
                              
                               ])
        rand_indx = torch.randperm(len(d))
        m = d[rand_indx]                 
        pred1 = torch.where(pred1>2,m,pred1)
        #print(pred1)
        pred1 = torch.where(pred1>2.5,w,pred1)
        #print(pred1)

        w = torch.ones(32, 16)
        q = torch.zeros(32, 16)

        pred1 = torch.where(pred1>2,q,pred1)
        pred1 = torch.where(pred1>1,w,pred1)
        pred1 = torch.round(pred1)
        print(pred1)
 
        pred1 = bin2dec(pred1, NUM_BITS)
        pred1 = pred1 / 10000
        #pred5 = (pred1 + pred2 + pred3 ) /3
        print("bit vector convert float vector :")
        pred8 = pred1.unsqueeze(0)
        #print(pred8)
        p = pred8 * 0.625
        k = pred8 * 0.625
        pred8 = torch.where(pred8>2,p,pred8)
        pred8 = torch.where(pred8<-2,k,pred8)
        print(pred8)
        output = decoder.sample_max_batch(pred8, None)
        print("decoder :")
        print(output)

    return output

SemanticRL_example = [          
                        'this message sents downlink information',
                      ]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    #parser.add_argument("--ckpt_pathCE", type=str, default='./ckpt_AWGN_CE')
    parser.add_argument("--ckpt_pathRL", type=str, default='./ckpt_AWGN_RL')  # or './ckpt_AWGN_RL'
    args = parser.parse_args()

    dict_train = pickle.load(open('./train_dict.pkl', 'rb'))
    rev_dict = {vv: kk for kk, vv in dict_train.items()}
    
    for input_str in SemanticRL_example:

        input_vector = [dict_train[x] for x in input_str.split(' ')] + [2]
        input_len = len(input_vector)
        input_vector = torch.tensor(input_vector)
    
        for ckpt_dir in [args.ckpt_pathRL]:#, args.ckpt_pathRL
            model_name = os.path.basename(ckpt_dir)
    
            encoder.load_state_dict(torch.load(ckpt_dir + '/encoder_epoch201.pth', map_location='cpu'))
            decoder.load_state_dict(torch.load(ckpt_dir + '/decoder_epoch201.pth', map_location='cpu'))
            embeds_shared.load_state_dict(torch.load(ckpt_dir + '/embeds_shared_epoch201.pth',  map_location='cpu'))
    
            for _ in range(1):
                output = do_test(input_vector.unsqueeze(0), encoder, decoder, normlize_layer, channel,
                        len_batch=torch.tensor(input_len).view(-1, ))
                output = output.cpu().numpy()[0]
                res = ' '.join(rev_dict[x] for x in output if x!=0 and x!=2)  # remove 'PAD' and 'EOS'
                print('result of {}:            {}'.format(model_name, res))
            print('--------------------------------------------------')