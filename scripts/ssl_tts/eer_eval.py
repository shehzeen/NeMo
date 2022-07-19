from nemo.collections.tts.models import ssl_tts
import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
import os
import json
from sklearn.metrics import roc_curve, auc
import numpy as np


def sv_emb_from_audio(audio_path,ssl_model):
    wav = wav_featurizer.process(audio_path)
    audio_signal = wav[None].cuda()
    audio_signal_length = torch.tensor( [ wav.shape[0] ]).cuda()
    ssl_model_cuda = ssl_model.cuda()
    emb = ssl_model_cuda(audio_signal,audio_signal_length)
    
    return emb[1]

def get_similarity(audio_path_1, audio_path_2,ssl_model):
#     audio_path_1 = "/home/shehzeenh/datasets/speaker_verification_full/vox1/segments/id10986/KH-yJAsKo1Q/00019_0_4.wav"
#     audio_path_2 = "/home/shehzeenh/datasets/speaker_verification_full/vox1/segments/id10986/KH-yJAsKo1Q/00037_0_4.wav"
    sv_emb1 = sv_emb_from_audio(audio_path_1,ssl_model)
    sv_emb2 = sv_emb_from_audio(audio_path_2,ssl_model)
    similarity = F.cosine_similarity(sv_emb1,sv_emb2) 
    similarity = similarity.item()
    # print(similarity)
    return similarity

def get_checkpoint(folder_path):
    
    ckpt_path = None
    for filename in os.listdir(folder_path):
        if filename.endswith('last.ckpt'):
            ckpt_path = os.path.join(folder_path, filename)
              
    return ckpt_path

path = "/home/shehzeenh/nemo_local/NeMo/examples/tts/conf/ssl_tts.yaml"
cfg = omegaconf.OmegaConf.load(path)
cfg.model.train_ds.manifest_filepath = "dummy"
cfg.model.validation_ds.manifest_filepath = "dummy"
ssl_model = ssl_tts.SSLDisentangler(cfg=cfg.model)
cfg.pop('init_from_pretrained_model')

# ckpt_path = '/home/shehzeenh/nemo_local/NeMo/examples/tts/nemo_experiments/Conformer-SSL/2022-07-13_10-56-22/checkpoints/Conformer-SSL--val_loss=1.1792-epoch=11-last.ckpt'
ckpt_path = get_checkpoint('/home/shehzeenh/nemo_local/NeMo/examples/tts/nemo_experiments/Conformer-SSL/2022-07-13_10-56-22/checkpoints/')
print ("CKPT PATH", ckpt_path)
cfg.init_from_ptl_ckpt = ckpt_path
ssl_model.maybe_init_from_pretrained_checkpoint(cfg=cfg)
wav_featurizer = WaveformFeaturizer(sample_rate=16000, int_values=False, augmentor=None)

y_score = []
y_true = []

# with open('/home/shehzeenh/datasets/speaker_verification_full/vox1_test/validation_test_pairs.txt') as f:
with open('/home/shehzeenh/datasets/speaker_verification_full/vox_o_trial.txt') as f:
    lines = f.readlines() # list containing lines of file
    
    for line in lines:
        line = line.strip() # remove leading/trailing white spaces
#         print(line)
        label, wav1, wav2 = line.split(' ')
        # print(label, wav1, wav2)
        
        wav1_path = '/home/shehzeenh/datasets/speaker_verification_full/test_wav/'+ str(wav1)
        wav2_path = '/home/shehzeenh/datasets/speaker_verification_full/test_wav/'+ str(wav2)
        
        ssl_model.eval()
        with torch.no_grad():
            sim_score = get_similarity(wav1_path,wav2_path,ssl_model)
            
        y_score.append(sim_score)
        y_true.append(int(label))

        

fpr, tpr, thresholds = roc_curve(y_true, y_score)
_auc = auc(fpr, tpr)
fnr = 1 - tpr
eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
eer_verify = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

assert abs(eer - eer_verify) < 1.0
print ("eer", eer)
print ("auc", _auc)