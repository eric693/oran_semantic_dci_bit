import os, platform, json, time, pickle, sys, argparse
import torch
from math import log
sys.path.append('./')
#sys.path.append("/usr/lib/python3.8")


#from data_loader import Dataset_sentence_test, collate_func [OK]
#from model import LSTMEncoder, LSTMDecoder, Embeds
#from utils import Normlize_tx, Channel, smaple_n_times [OK]

from torch.utils.data import Dataset
import numpy as np

from torch.distributions import Normal

from collections import namedtuple


from nltk.translate.bleu_score import sentence_bleu
from nltk.translate import bleu
from nltk.translate.bleu_score import SmoothingFunction



_snr = 10
_iscomplex = True
channel_dim = 256

def someFunction():
    print('You passed this Python program from C! Congratulations!')
    print('mwnl!')

def do_test():
    print('Semantic!')

##################
class Dataset_sentence_test(Dataset):
    def __init__(self, _path):
        if not _path: _path = '/home/eric/test/SemanticRL/Europarl'
        self._path = os.path.join(_path, 'english_vocab.pkl')
        self.dict = {}
        tmp = pickle.load(open(self._path, 'rb'))
        for kk,vv in tmp['voc'].items(): self.dict[kk] = vv+3
        # add sos, eos, and pad.
        self.dict['PAD'], self.dict['SOS'], self.dict['EOS'] = 0, 1, 2
        self.len_range = tmp['len_range']
        self.rev_dict = {vv: kk for kk, vv in self.dict.items()}
        self.data_num = [[1] + list(map(lambda t:self.dict[t], x.split(' '))) + [2]
                         + (self.len_range[1]-len(x.split(' ')))*[0]
                         for idx, x in enumerate(tmp['sent_str']) if idx%5==0]
        print('[*]------------vocabulary size is:----', self.get_dict_len())
        print('[*]------------sentences size is:----', len(self.data_num))

    def __getitem__(self, index):
        return torch.tensor(self.data_num[index])

    def __len__(self):
        return len(self.data_num)

    def get_dict_len(self):
        return len(self.dict)


def collate_func(in_data):
    batch_tensor, batch_len = list(zip(*(sorted(in_data, key=lambda s:-s[1]))))
    return torch.stack(batch_tensor, dim=0), batch_len

  

##################

##################
class Normlize_tx:
    def __init__(self, _iscomplex):
        self._iscomplex = _iscomplex
    def apply(self, _input):
        _dim = _input.shape[1]//2 if self._iscomplex else _input.shape[1]
        _norm = _dim**0.5 / torch.sqrt(torch.sum(_input**2, dim=1))
        return _input*_norm.view(-1,1)

class Channel:
    # returns the message when passed through a channel.
    # AGWN, Fading
    # Note that we need to make sure that the colle map will not change in this
    # step, thus we should not use *= and +=.
    def __init__(self, _iscomplex):
        self._iscomplex = _iscomplex

    def ideal_channel(self, _input):
        return _input

    def awgn(self, _input, _snr):
        _std = (10**(-_snr/10.)/2)**0.5 if self._iscomplex else (10**(-_snr/10.))**0.5  # for complex signals.
        _input = _input + torch.randn_like(_input) * _std
        #print(_std)
        #print(_input)
        return _input
    
    def smaple_n_times(n, x):
        if n>1:
            x = x.unsqueeze(1) # Bx1x...
            x = x.expand(-1, n, *([-1]*len(x.shape[2:])))
            x = x.reshape(x.shape[0]*n, *x.shape[2:])
        return x
    
if __name__ =='__main__':

    is_complex = False
    n = Normlize_tx(is_complex)
    x = torch.tensor([[1., 2., 3., 4., 5., 6., 7., 8., 9., 10.], [18., 2., 3., 4., 5., 6., 7., 8., 9., 10.]])
    y = n.apply(x)
    print(y)
    #for i in range(x.shape[1]//2):
    #    print(y[:,i], y[:,5+i])

    c = Channel(is_complex)
    # x = torch.ones(2,4)
    z = c.awgn(y,10)
    print(z)
    
normlize_layer = Normlize_tx(_iscomplex=_iscomplex)
channel = Channel(_iscomplex=_iscomplex)
##################

##################
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Decoder_Meta():
    
    # Note, `self` methods/attributes should be defined in child classes.
    def forward_rl(self, input_features, sample_max=None, multiple_sample=5, x_mask=None):

        if x_mask is not None:  # i.e., backbone is Transformer
            max_seq_len = input_features.shape[-1] // self.channel_dim
            input_features = smaple_n_times(multiple_sample, input_features.view(input_features.shape[0], max_seq_len, -1))
            input_features = self.from_channel_emb(input_features)
            x_mask = smaple_n_times(multiple_sample, x_mask)
        else: # LSTM
            def smaple_n_times(n, x):
                    if n>1:
                        x = x.unsqueeze(1) # Bx1x...
                        x = x.expand(-1, n, *([-1]*len(x.shape[2:])))
                        x = x.reshape(x.shape[0]*n, *x.shape[2:])
                    return x
            max_seq_len = 21  # we set the max sentence length to 20, plus an EOS token. You can adjust this value.
            input_features = smaple_n_times(multiple_sample, input_features)

        batch_size = input_features.size(0)
        state = self.init_hidden(input_features)

        seq = input_features.new_zeros((batch_size, max_seq_len), dtype=torch.long)
        seq_logprobs = input_features.new_zeros((batch_size, max_seq_len))
        seq_masks = input_features.new_zeros((batch_size, max_seq_len))
        it = input_features.new_zeros(batch_size, dtype=torch.long).fill_(self.sos_id)
        unfinished = it == self.sos_id
        for t in range(max_seq_len):
            logprobs, state = self._forward_step(it, state, input_features, x_mask)  # bs*vocab_size
            if sample_max:
                sample_logprobs, it = torch.max(logprobs.detach(), 1)
            else:
                it = torch.distributions.Categorical(logits=logprobs.detach()).sample()
                sample_logprobs = logprobs.gather(1, it.unsqueeze(1))  # gather the logprobs at sampled positions
            it = it.view(-1).long()
            sample_logprobs = sample_logprobs.view(-1)

            seq_masks[:, t] = unfinished
            it = it * unfinished.type_as(it)  # bs
            seq[:, t] = it
            seq_logprobs[:, t] = sample_logprobs

            unfinished = unfinished * (it != self.eos_id)  # update if finished according to EOS
            if unfinished.sum() == 0:
                break

        return seq, seq_logprobs, seq_masks


class Embeds(nn.Module):
    def __init__(self, vocab_size, num_hidden):
        super(Embeds, self).__init__()
        self.emb = nn.Embedding(vocab_size, num_hidden, padding_idx=0)  # learnable params, nn.Embedding是用來將一個數字變成一個指定維度的向量的
        #vocab.GloVe(name='6B', dim=50, cache='../Glove') This is optional.

    def __call__(self, inputs):
        return self.emb(inputs)

class LSTMEncoder(nn.Module):
    def __init__(self, channel_dim, embedds):
        super(LSTMEncoder, self).__init__()

        self.num_hidden = 128
        self.pad_id = 0
        self.word_embed_encoder = embedds
        self.lstm_encoder = nn.LSTM(input_size=self.num_hidden, hidden_size=self.num_hidden,
                                    num_layers=2, bidirectional=True, batch_first=True)
        self.to_chanenl_embedding = nn.Sequential(nn.Linear(2*self.num_hidden, 2*self.num_hidden), nn.ReLU(),
                                                    nn.Linear(2*self.num_hidden, channel_dim))

class LSTMDecoder(nn.Module, Decoder_Meta):
    def __init__(self, channel_dim, embedds, vocab_size):
        super(LSTMDecoder, self).__init__()
        self.num_hidden = 128
        self.vocab_size = vocab_size
        self.channel_dim = channel_dim
        self.pad_id, self.sos_id, self.eos_id, self.unk_id = 0, 1, 2, 3
        self.word_embed_decoder = embedds
        self.from_channel_emb = nn.Linear(channel_dim, 2*self.num_hidden)
        self.lstmcell_decoder = nn.LSTMCell(input_size=self.num_hidden, hidden_size=self.num_hidden)
        self.linear_and_dropout_classifier_decoder = nn.Sequential(
            nn.Linear(self.num_hidden, self.num_hidden),
            nn.Dropout(0.5), nn.Linear(self.num_hidden, self.vocab_size))

device = torch.device("cpu:0")
#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # use CPU or GPU
embeds_shared = Embeds(vocab_size=24064, num_hidden=128).to(device)
encoder = LSTMEncoder(channel_dim=channel_dim, embedds=embeds_shared).to(device)
decoder = LSTMDecoder(channel_dim=channel_dim, embedds=embeds_shared, vocab_size = 24064).to(device)
#print(encoder)
#print(decoder)
encoder = encoder.eval()
decoder = decoder.eval()
#print(embeds_shared)

embeds_shared = embeds_shared.eval()
##################
def do_test2(input_data, encoder, decoder, normlize_layer, channel, len_batch):
    
    print('successful!')

    with torch.no_grad():
        
        encoder_output, _ = encoder(input_data, len_batch)

    with open('do_test2_input_data.txt', 'w') as w:
        w.write(str(input_data))
        
    with open('/home/mwnl/o-ran_project/o-du-l2/src/5gnrmac/semanticRL_example2_input_data.txt', 'w') as w:
        w.write(str(input_data))

    with open('do_test2_encoder_output.txt', 'w') as w:
        w.write(str(encoder_output))
        
    with open('/home/mwnl/o-ran_project/o-du-l2/src/5gnrmac/semanticRL_example2_encoder_output.txt', 'w') as w:
        w.write(str(encoder_output))

        normlize_layer_output = normlize_layer.apply(encoder_output)

    with open('do_test2_normlize_layer_output.txt', 'w') as w:
        w.write(str(normlize_layer_output))
        
    with open('/home/mwnl/o-ran_project/o-du-l2/src/5gnrmac/semanticRL_example2_normlize_layer_output.txt', 'w') as w:
        w.write(str(normlize_layer_output))

        channel_output = channel.awgn(normlize_layer_output, _snr=_snr)

    with open('do_test2_channel_output.txt', 'w') as w:
        w.write(str(channel_output))
        
    with open('/home/mwnl/o-ran_project/o-du-l2/src/5gnrmac/semanticRL_example2_channel_output.txt', 'w') as w:
        w.write(str(channel_output))

        decoder_output = decoder.sample_max_batch(channel_output, None)

    with open('do_test2_decoder_output.txt', 'w') as w:
        w.write(str(decoder_output))
        
    with open('/home/mwnl/o-ran_project/o-du-l2/src/5gnrmac/semanticRL_example2_decoder_output.txt', 'w') as w:
        w.write(str(decoder_output))

    return decoder_output



def do_test3(input_data, encoder, decoder, normlize_layer, channel, len_batch):
    print('successful!')
    



 

