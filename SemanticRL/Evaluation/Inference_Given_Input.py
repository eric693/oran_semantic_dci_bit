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
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate import bleu
from nltk.translate.bleu_score import SmoothingFunction
from ctypes import *
import ctypes
#from binary_converter import float2bit, bit2float
#import tensorflow as tf
_snr = 10
_iscomplex = True
channel_dim = 32 #


device = torch.device("cpu:0")
#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # use CPU or GPU
embeds_shared = Embeds(vocab_size=529, num_hidden=32).to(device)
encoder = LSTMEncoder(channel_dim=channel_dim, embedds=embeds_shared).to(device)
decoder = LSTMDecoder(channel_dim=channel_dim, embedds=embeds_shared, vocab_size = 529).to(device)
#print(encoder)
#print(decoder)
encoder = encoder.eval()
decoder = decoder.eval()
#print(embeds_shared)

embeds_shared = embeds_shared.eval()


normlize_layer = Normlize_tx(_iscomplex=_iscomplex)
channel = Channel(_iscomplex=_iscomplex)
#print(normlize_layer)
#print(channel)

def do_test1(input_data, encoder, decoder, normlize_layer, channel, len_batch):

    with torch.no_grad():
        
        encoder_output, _ = encoder(input_data, len_batch)

    with open('do_test1_input_data.txt', 'w') as w:
        w.write(str(input_data))
        
    with open('/home/eric/semantic/o-du-l2/src/5gnrmac/semanticRL_example1_input_data.txt', 'w') as w:
        w.write(str(input_data))
    
    with open('do_test1_encoder_output.txt', 'w') as w:
        w.write(str(encoder_output))
        
    with open('/home/eric/semantic/o-du-l2/src/5gnrmac/semanticRL_example1_encoder_output.txt', 'w') as w:
        w.write(str(encoder_output))

        normlize_layer_output = normlize_layer.apply(encoder_output)

    with open('do_test1_normlize_layer_output.txt', 'w') as w:
        w.write(str(normlize_layer_output))
        
    with open('/home/eric/semantic/o-du-l2/src/5gnrmac/semanticRL_example1_normlize_layer_output.txt', 'w') as w:
        w.write(str(normlize_layer_output))

        channel_output = channel.awgn(normlize_layer_output, _snr=_snr)

    with open('do_test1_channel_output.txt', 'w') as w:
        w.write(str(channel_output))
        
    with open('/home/eric/semantic/o-du-l2/src/5gnrmac/semanticRL_example1_channel_output.txt', 'w') as w:
        w.write(str(channel_output))

        decoder_output = decoder.sample_max_batch(channel_output, None)

    with open('do_test1_decoder_output.txt', 'w') as w:
        w.write(str(decoder_output))
        
    with open('/home/eric/semantic/o-du-l2/src/5gnrmac/semanticRL_example1_decoder_output.txt', 'w') as w:
        w.write(str(decoder_output))
        
    return decoder_output

        
def do_test2(input_data, encoder, decoder, normlize_layer, channel, len_batch):

    with torch.no_grad():
        
        encoder_output, _ = encoder(input_data, len_batch)

    with open('do_test2_input_data.txt', 'w') as w:
        w.write(str(input_data))
        
    with open('/home/eric/semantic/o-du-l2/src/5gnrmac/semanticRL_example2_input_data.txt', 'w') as w:
        w.write(str(input_data))

    with open('do_test2_encoder_output.txt', 'w') as w:
        w.write(str(encoder_output))
        
    with open('/home/eric/semantic/o-du-l2/src/5gnrmac/semanticRL_example2_encoder_output.txt', 'w') as w:
        w.write(str(encoder_output))

        normlize_layer_output = normlize_layer.apply(encoder_output)

    with open('do_test2_normlize_layer_output.txt', 'w') as w:
        w.write(str(normlize_layer_output))
        
    with open('/home/eric/semantic/o-du-l2/src/5gnrmac/semanticRL_example2_normlize_layer_output.txt', 'w') as w:
        w.write(str(normlize_layer_output))

        channel_output = channel.awgn(normlize_layer_output, _snr=_snr)

    with open('do_test2_channel_output.txt', 'w') as w:
        w.write(str(channel_output))
        
    with open('/home/eric/semantic/o-du-l2/src/5gnrmac/semanticRL_example2_channel_output.txt', 'w') as w:
        w.write(str(channel_output))

        decoder_output = decoder.sample_max_batch(channel_output, None)

    with open('do_test2_decoder_output.txt', 'w') as w:
        w.write(str(decoder_output))
        
    with open('/home/eric/semantic/o-du-l2/src/5gnrmac/semanticRL_example2_decoder_output.txt', 'w') as w:
        w.write(str(decoder_output))

    return decoder_output



# Specification define
SemanticRL_example1 = [
                       'the messages will transmit from gnb to ue',
                      ]
SemanticRL_example2 = [   
                       'this message sents downlink information',
                      ]

if __name__ == "__main__":
    


    parser = argparse.ArgumentParser()
    #parser.add_argument("--ckpt_pathCE", type=str, default='./ckpt_AWGN_CE_Stage2')
    parser.add_argument("--ckpt_pathRL", type=str, default='./ckpt_AWGN_RL_SCSIU_')  # or './ckpt_AWGN_RL'
    args = parser.parse_args()

    dict_train = pickle.load(open('./train_dict.pkl', 'rb'))
    rev_dict = {vv: kk for kk, vv in dict_train.items()}

    for input_str in SemanticRL_example1:

        input_vector = [dict_train[x] for x in input_str.split(' ')] + [2]
        input_len = len(input_vector)
        input_vector = torch.tensor(input_vector)
    
        for ckpt_dir in [args.ckpt_pathRL]:#, args.ckpt_pathCE if args.ckpt_resume>0:
            model_name = os.path.basename(ckpt_dir)
    
            encoder.load_state_dict(torch.load(ckpt_dir + '/encoder_epoch201.pth', map_location='cpu'))
            decoder.load_state_dict(torch.load(ckpt_dir + '/decoder_epoch201.pth', map_location='cpu'))
            embeds_shared.load_state_dict(torch.load(ckpt_dir + '/embeds_shared_epoch201.pth',  map_location='cpu'))
    
            
            SemanticRL_example1_output = do_test1(input_vector.unsqueeze(0), encoder, decoder, normlize_layer, channel,
                      len_batch=torch.tensor(input_len).view(-1, ))
         

            SemanticRL_example1_output = SemanticRL_example1_output.cpu().numpy()[0]
            res = ' '.join(rev_dict[x] for x in SemanticRL_example1_output if x!=0 and x!=2)  # remove 'PAD' and 'EOS'
            print('-----------------------------------------------')
            print('-------------------Case1-----------------------')
            print('-----------------------------------------------')
            #print('(candidate)result of {}:         {}'.format(model_name, res))

            sent_a_reference = 'the messages will transmit from gnb to ue'.split()
            print('case1 reference sentence = {} '.format(sent_a_reference))
            sent_b_candidate = ' {} '.format(res).split()
            print('case1 candidate sentence = {} '.format(sent_b_candidate))
            
            with open('/home/eric/semantic/o-du-l2/src/5gnrmac/semanticRL_example1_candidate_sentence.txt', 'w') as w:
                w.write(str(res))
                
            with open('semanticRL_example1_candidate_sentence.txt', 'w') as w:
                w.write(str(res))
                 
            
            print('-----------------------------------------------')
            print('-------------------Case2-----------------------')
            print('-----------------------------------------------')
        
                
    for input_str in SemanticRL_example2:

        input_vector = [dict_train[x] for x in input_str.split(' ')] + [2]
        input_len = len(input_vector)
        input_vector = torch.tensor(input_vector)
    
        for ckpt_dir in [args.ckpt_pathRL]:#, args.ckpt_pathCE if args.ckpt_resume>0:
            model_name = os.path.basename(ckpt_dir)
    
            encoder.load_state_dict(torch.load(ckpt_dir + '/encoder_epoch201.pth', map_location='cpu'))
            decoder.load_state_dict(torch.load(ckpt_dir + '/decoder_epoch201.pth', map_location='cpu'))
            embeds_shared.load_state_dict(torch.load(ckpt_dir + '/embeds_shared_epoch201.pth',  map_location='cpu'))
    
            
            SemanticRL_example2_output = do_test2(input_vector.unsqueeze(0), encoder, decoder, normlize_layer, channel,
                      len_batch=torch.tensor(input_len).view(-1, ))

            
            SemanticRL_example2_output = SemanticRL_example2_output.cpu().numpy()[0]
            res = ' '.join(rev_dict[x] for x in SemanticRL_example2_output if x!=0 and x!=2)  # remove 'PAD' and 'EOS'
            #print('(candidate)result of {}:         {}'.format(model_name, res))

            
            sent_c_reference = 'this message sents downlink information'.split()
            print('case2 reference sentence = {} '.format(sent_c_reference))
            sent_d_candidate = ' {} '.format(res).split()
            print('case2 candidate sentence = {} '.format(sent_d_candidate))

            with open('/home/eric/semantic/o-du-l2/src/5gnrmac/semanticRL_example2_candidate_sentence.txt', 'w') as w:
                w.write(str(res))
                
            with open('semanticRL_example2_candidate_sentence.txt', 'w') as w:
                w.write(str(res))
                
            print('-----------------------------------------------')
            print('-----------------BLEU score--------------------')
            print('-----------------------------------------------')
            

            smoothie = SmoothingFunction().method1
            bleu1= bleu([sent_a_reference], sent_b_candidate, smoothing_function=smoothie)
            smoothie = SmoothingFunction().method2
            bleu2= bleu([sent_c_reference], sent_b_candidate, smoothing_function=smoothie)
            smoothie = SmoothingFunction().method3
            bleu3= bleu([sent_c_reference], sent_d_candidate, smoothing_function=smoothie)
            smoothie = SmoothingFunction().method4
            bleu4= bleu([sent_a_reference], sent_d_candidate, smoothing_function=smoothie)
            
            
            print('bleu score 1 (case1 reference sentence, case1 candidate sentence)= {} '.format(bleu1))
            print('bleu score 2 (case2 reference sentence, case1 candidate sentence)= {} '.format(bleu2))
            print('bleu score 3 (case2 reference sentence, case2 candidate sentence)= {} '.format(bleu3))
            print('bleu score 4 (case1 reference sentence, case2 candidate sentence)= {} '.format(bleu4))
            
            
            
            print('-----------------------------------------------')
            print('-------------------Compare---------------------')
            print('-----------------------------------------------')
            
            
            if ( bleu1 > bleu2 ):
                print('bleu score 1 > bleu score 2 = {} '.format(bleu1))
                print('case1 reference sentence = {} '.format(sent_a_reference))
                print('case1 candidate sentence = {} '.format(sent_b_candidate))
                print('Confirmation case1 sent message \n')
                
            else:
                print('case2 bleu score = {} '.format(bleu2))
                print('case2 reference sentence = {} '.format(sent_c_reference))
                print('case2 candidate sentence = {} '.format(sent_d_candidate))
                print('Confirmation case2 sent message \n')
                
        if ( bleu3 > bleu4 ):
            print('bleu score 3 > bleu score 4 = {} '.format(bleu3))
            print('case2 reference sentence = {} '.format(sent_c_reference))
            print('case2 candidate sentence = {} '.format(sent_d_candidate))
            print('Confirmation case2 sent message')
                
        else:
            print('case1 bleu score = {} '.format(bleu4))
            print('case1 reference sentence = {} '.format(sent_a_reference))
            print('case1 candidate sentence = {} '.format(sent_b_candidate))
            print('Confirmation case1 sent message')
            