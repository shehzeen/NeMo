import torch
import torch.nn.functional as F
import numpy as np
import scipy.signal
from pathlib import Path
from typing import Union, Tuple
from transformers import Wav2Vec2FeatureExtractor
import torchaudio

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.interpolate import interp1d

from dataclasses import dataclass
from typing import Optional, Tuple

from transformers import (
    AutoConfig,
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)
from transformers.file_utils import ModelOutput




# from config import parameters
parameters = {'training_parameters': {'train_csv': 'datasets_csv/total_df_riva_multilang.csv', 'val_csv': 'datasets_csv/nvidia_test.csv', 'train_dataset_root': 'datasets/total_dataset/', 'test_dataset_root': 'datasets/nvidia-test/', 'noise_dataset_root': 'datasets/noise/', 'output_path': 'output/', 'n_epochs': 20.0, 'learning_rate': 1e-05, 'weight_decay': 0.0001, 'lr_scheduler_type': 'linear', 'warmup_ratio': 0.0, 'per_device_train_batch_size': 8, 'per_device_eval_batch_size': 8, 'gradient_accumulation_steps': 2, 'n_epochs_train_eval': 0, 'evaluation_strategy': 'steps', 'save_steps': 50000000, 'eval_steps': 20, 'logging_steps': 200, 'save_total_limit': 2, 'random_seed': 0}, 'training_datasets': {'CREMA-D': True, 'EmoV-DB': False, 'JL': True, 'MELD': False, 'RAVDESS': True, 'MSP-Podcast': False, 'Lindy_and_Rodney': True, 'Riva_Multilang': False}, 'resampling': {'enabled': False, 'balance_classes': False, 'balance_datasets': False}, 'augmentation_proba': {'part': 0.3, 'crop': 0.0, 'shift': 0.2, 'speed': 0.0, 'pitch': 0.0, 'gauss_noise': 0.2, 'uniform_noise': 0.2, 'background_sound': 0.2, 'concat_mixup': 0.0}, 'model_parameters': {'base_model': 'facebook/wav2vec2-base', 'final_dropout': 0.2, 'pooling_mode': 'mean', 'classes': ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']}}


# load parameters from YAML configuration file
model_parameters = parameters['model_parameters']

# base model
model_name_or_path = model_parameters['base_model']

# labels and cfg
label_list = sorted(model_parameters['classes'])
num_labels = len(label_list)
config = AutoConfig.from_pretrained(
    model_name_or_path,
    num_labels=num_labels,
    label2id={label: i for i, label in enumerate(label_list)},
    id2label={i: label for i, label in enumerate(label_list)},
    finetuning_task="wav2vec2_clf",
)


@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class ClassificationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(model_parameters['final_dropout'])
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features):
        x = features
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ClassifierModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config=None):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = ClassificationHead()
        self.mode = 'classifier'
        self.pooling = nn.AvgPool1d(5, stride=1, padding=2)
        self.dpo_pooling = nn.AvgPool1d(6, stride=6)

    def freeze_feature_extractor(self):
        self.wav2vec2.freeze_feature_extractor()

    def dpo_set_reference_model(self, reference_model):
        self.reference_model = reference_model
        for param in self.reference_model.parameters():
            param.requires_grad = False
        self.reference_model.mode = 'seq2seq_dirichlet'

    def get_emotion_embedding(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
            timestamps=None,
            dpo_emotion=None,
            l2_norm=True,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs.last_hidden_state
        if self.mode in ['classifier', 'classifier_no_neutral_sigmoid', 'classifier_dirichlet']:
            hidden_states = torch.mean(hidden_states, dim=1)
        elif self.mode in ['seq2seq', 'seq2seq_no_neutral', 'seq2seq_dirichlet']:
            hidden_states = self.pooling(hidden_states.transpose(1, 2)).transpose(1, 2)

        if l2_norm:
            hidden_states = torch.nn.functional.normalize(hidden_states, p=2, dim=1)
        return hidden_states

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
            timestamps=None,
            dpo_emotion=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs.last_hidden_state
        if self.mode in ['classifier', 'classifier_no_neutral_sigmoid', 'classifier_dirichlet']:
            hidden_states = torch.mean(hidden_states, dim=1)
        elif self.mode in ['seq2seq', 'seq2seq_no_neutral', 'seq2seq_dirichlet']:
            hidden_states = self.pooling(hidden_states.transpose(1, 2)).transpose(1, 2)

        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            if self.mode == 'classifier':
                loss = F.cross_entropy(logits, labels)
            elif self.mode == 'classifier_no_neutral_sigmoid':
                mask = torch.tensor([1, 1, 1, 1, 0, 1], device=labels.device).bool()
                loss = F.binary_cross_entropy_with_logits(logits[..., mask], labels[..., mask])
            elif self.mode == 'classifier_dirichlet':
                params = logits ** 2 + 1.0e-6
                dirichlet = torch.distributions.Dirichlet(params)
                coef = 0.1
                labels = (1 - coef) * labels + coef * (1 / num_labels)
                loss = -dirichlet.log_prob(labels).mean()
            elif self.mode == 'seq2seq':
                loss = F.cross_entropy(logits[0], labels[0])
            elif self.mode == 'seq2seq_no_neutral':
                mask = torch.tensor([1, 1, 1, 1, 0, 1], device=labels.device).bool()
                loss = F.binary_cross_entropy_with_logits(logits[0, ..., mask], labels[0, ..., mask])
            elif self.mode == 'seq2seq_dirichlet':
                params = logits ** 2 + 1.0e-6
                dirichlet = torch.distributions.Dirichlet(params)
                coef = 0.01
                labels = (1 - coef) * labels + coef * (1 / num_labels)
                loss = -dirichlet.log_prob(labels).mean()
            elif self.mode == 'dpo':
                if labels == 1:
                    y_w, y_l = dpo_emotion
                elif labels == 2:
                    y_l, y_w = dpo_emotion
                else:
                    raise Exception('bad label', labels)

                dir_params_train = logits ** 2 + 1.0e-6
                with torch.no_grad():
                    dir_params_ref = self.reference_model(input_values).logits ** 2 + 1.0e-6

                dir_train = torch.distributions.Dirichlet(dir_params_train)
                dir_ref = torch.distributions.Dirichlet(dir_params_ref)

                #y_w = self.dpo_pooling(y_w.transpose(0, 1)).transpose(0, 1)
                #y_l = self.dpo_pooling(y_l.transpose(0, 1)).transpose(0, 1)

                beta = 0.5
                term_1 = dir_train.log_prob(y_w) - dir_ref.log_prob(y_w)
                term_2 = dir_train.log_prob(y_l) - dir_ref.log_prob(y_l)
                loss = -F.logsigmoid(beta * (term_1 - term_2)).mean()

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


def _resample_to_16k(audio, orig_sample_rate):
    number_of_samples = round(len(audio) * float(16000) / orig_sample_rate)
    resampled_audio = scipy.signal.resample(audio, number_of_samples)
    return resampled_audio


def _validate_and_resample(speech_array: np.ndarray, orig_sample_rate: int):
    if not isinstance(speech_array, np.ndarray):
        raise TypeError('speech_array must be np.ndarray')
    if speech_array.ndim > 1:
        if speech_array.shape[0] == 1:
            speech_array = speech_array[0]
        elif speech_array.shape[0] == 2:
            speech_array = speech_array.mean(axis=0)
        else:
            raise Exception('Wrong shape of input.')
    speech_array = _resample_to_16k(speech_array, orig_sample_rate)
    return speech_array


def predict(
    speech_array: np.ndarray,
    orig_sample_rate: int
) -> np.ndarray:
    r"""
        Predict emotional distribution for a single input audio.

        Args:
            speech_array (:obj:`np.ndarray`):
                1d input audio sequence
            orig_sample_rate (:obj:`int`):
                Sampling rate of the provided input sequence
        Returns:
            scores (:obj:`np.ndarray`):
                1d array containing probability distribution.
                Index to emotion:
                    0 - anger
                    1 - disgust
                    2 - fear
                    3 - happy
                    4 - neutral
                    5 - sad
    """
    speech_array = _validate_and_resample(speech_array, orig_sample_rate)
    features = processor(
        speech_array,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    input_values = features.input_values.cuda()
    with torch.no_grad():
        outputs = model(input_values)
        logits = outputs.logits
    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    return scores

def get_emotion_embedding(
    speech_array: np.ndarray,
    orig_sample_rate: int,
    model,
    processor,
) -> np.ndarray:
    r"""
        Predict emotional embedding for a single input audio.

        Args:
            speech_array (:obj:`np.ndarray`):
                1d input audio sequence
            orig_sample_rate (:obj:`int`):
                Sampling rate of the provided input sequence
        Returns:
            embedding (:obj:`np.ndarray`):
    """
    speech_array = _validate_and_resample(speech_array, orig_sample_rate)
    features = processor(
        speech_array,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    input_values = features.input_values.cuda()

    with torch.no_grad():
        embedding = model.get_emotion_embedding(input_values)
        
    return embedding


def get_emotion_embedding_from_file(
    path,
    model,
    processor,
) -> np.ndarray:
    audio, sample_rate = torchaudio.load(path)

    audio = audio.numpy().mean(axis=0)
    # prediction = predict(audio, sample_rate)
    embedding = get_emotion_embedding(audio, sample_rate, model, processor)
    return embedding

def init_emotion_encoder(weights_path, use_cuda=True):
    # loading the model and preprocessor
    model = ClassifierModel.from_pretrained(
        Path(weights_path),
        config=config
    )
    model.eval()
    if use_cuda:
        model.cuda()
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_name_or_path
    )
    return model, processor


if __name__ == '__main__':
    import os
    # input model weights
    weights_path = '/home/ecasanova/Projects/Checkpoints/emotion_classifier/w2v2_l071_acc08.bin' # download checkpoint from here: https://gitlab-master.nvidia.com/ilyaf/audio2emotion

    emotion_encoder, emotion_encoder_processor = init_emotion_encoder(weights_path)
    

    # check emotion integrite

    # emotion_files_path = "/home/ecasanova/Projects/Datasets/Lindy/CMU_HAPPY/" # 0.99
    # emotion_files_path = "/home/ecasanova/Projects/Datasets/Lindy/CMU_DISGUSTED/" # 0.99
    # emotion_files_path = "/home/ecasanova/Projects/Datasets/Lindy/CMU_CALM/"# 0.6958
    # emotion_files_path = "/home/ecasanova/Projects/Datasets/Lindy/CMU_SAD/" # 0.99
    emotion_files_path = "/home/ecasanova/Projects/Datasets/Lindy/CMU_ANGRY/" # 0.99
    # emotion_files_path = "/home/ecasanova/Projects/Datasets/Lindy/WIZWIKI/" # 0.6837 -- NO DEFINED EMOTION


    files = os.listdir(emotion_files_path) 
    # debug
    files = files[:10]
    index = 0
    prev_emb = None
    sims = []
    for file in files:
        file_path = os.path.join(emotion_files_path, file)
        embedding = get_emotion_embedding_from_file(file_path, emotion_encoder, emotion_encoder_processor)
        if prev_emb is not None:
            sim = torch.nn.functional.cosine_similarity(embedding, prev_emb)
            sims.append(sim.item())
        prev_emb = embedding.clone()

    print("Average similarity:", sum(sims)/len(sims))


    embedding_angry = get_emotion_embedding_from_file("/home/ecasanova/Projects/Datasets/Lindy/CMU_ANGRY/LINDY_CMU_ANGRY_000117.wav", emotion_encoder, emotion_encoder_processor)
    embedding_happy = get_emotion_embedding_from_file("/home/ecasanova/Projects/Datasets/Lindy/CMU_HAPPY/LINDY_CMU_HAPPY_000117.wav", emotion_encoder, emotion_encoder_processor)
    print("Angry vs Happy:", torch.nn.functional.cosine_similarity(embedding_angry, embedding_happy).item())
