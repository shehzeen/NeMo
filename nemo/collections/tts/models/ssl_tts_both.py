from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.loggers import LoggerCollection, TensorBoardLogger
from nemo.core.classes import ModelPT
from nemo.collections.asr.models import ssl_models
from nemo.collections.asr.losses.angularloss import AngularSoftmaxLoss
import nemo.collections.tts.torch.data as TTSData
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from hydra.utils import instantiate
from dataclasses import dataclass
from nemo.collections.tts.torch.tts_tokenizers import BaseTokenizer, EnglishCharsTokenizer, EnglishPhonemesTokenizer


def decode(tokenizer, token_list):
    return tokenizer.sep.join(tokenizer._id2token[t] for t in token_list)


class SSLDisentangler(ModelPT):

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)
        self.preprocessor = SSLDisentangler.from_config_dict(self._cfg.preprocessor)
        self.encoder = SSLDisentangler.from_config_dict(self._cfg.encoder)
        self._tb_logger = None

        self.downstream_nets = nn.ModuleDict()
        for task in self._cfg.downstream_heads.task_names:
            
            if task == 'speaker_verification':
                in_dim = self._cfg.encoder.d_model
                out_dim = self._cfg.downstream_heads.speaker_embed_size
                num_speakers = self._cfg.downstream_heads.num_speakers
                self.downstream_nets[task] = nn.Linear(in_dim,out_dim)
                self.sv_linear = nn.Linear(out_dim,num_speakers)
                self.sv_loss = AngularSoftmaxLoss(scale=30, margin=0.4)
                # self.sv_loss = nn.CrossEntropyLoss()

            elif task == 'content':
                in_dim = self._cfg.encoder.d_model
                out_dim = self._cfg.downstream_heads.content_embed_size
                num_chars = len(self._text_tokenizer.tokens) #list of english tokens
                self.downstream_nets[task] = nn.Linear(in_dim,out_dim)
                self.content_linear = nn.Linear(out_dim,num_chars)
                self.ctc_loss = nn.CTCLoss(blank=self._text_tokenizer.blank)
           

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        results = []

        model = PretrainedModelInfo(
            pretrained_model_name="ssl_en_conformer_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:ssl_en_conformer_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/ssl_en_conformer_large/versions/1.10.1/files/ssl_en_conformer_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="ssl_en_conformer_xlarge",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:ssl_en_conformer_xlarge",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/ssl_en_conformer_xlarge/versions/1.10.0/files/ssl_en_conformer_xlarge.nemo",
        )
        results.append(model)

        return results

    @property
    def tb_logger(self):
        if self._tb_logger is None:
            if self.logger is None and self.logger.experiment is None:
                return None
            tb_logger = self.logger.experiment
            if isinstance(self.logger, LoggerCollection):
                for logger in self.logger:
                    if isinstance(logger, TensorBoardLogger):
                        tb_logger = logger.experiment
                        break
            self._tb_logger = tb_logger
        return self._tb_logger

    def __setup_dataloader_from_config(self, data_config):
        # _text_tokenizer = instantiate(self._cfg.text_tokenizer)
        _text_tokenizer = self._text_tokenizer = EnglishCharsTokenizer(add_blank_at="last")
        for task in self._cfg.downstream_heads.task_names:
            if task == 'speaker_verification':
                sv_dataset = TTSData.TTSDataset(
                    manifest_filepath=data_config['manifest_speaker_verification_fp'], 
                    sample_rate=16000, 
                    text_tokenizer=_text_tokenizer,
                    max_duration=16.7,
                    sup_data_types=['speaker_id'])
                sv_loader = torch.utils.data.DataLoader(
                    sv_dataset,
                    batch_size=data_config['batch_size'],
                    collate_fn=sv_dataset.general_collate_fn,
                    shuffle=data_config['shuffle'],
                    num_workers=data_config.get('num_workers', 0),
                    pin_memory=data_config.get('pin_memory', False))

            if task == 'content':  
                content_dataset = TTSData.TTSDataset(
                    manifest_filepath=data_config['manifest_content_fp'],
                    sample_rate=16000,
                    text_tokenizer=_text_tokenizer,
                    max_duration=16.7)
                content_loader = torch.utils.data.DataLoader(
                    content_dataset, 
                    batch_size=data_config['batch_size'],
                    collate_fn=content_dataset.general_collate_fn,
                    shuffle=data_config['shuffle'],
                    num_workers=data_config.get('num_workers', 0),
                    pin_memory=data_config.get('pin_memory', False))
        
        loaders = {"sv": sv_loader, "content": content_loader}
        return loaders


    def setup_training_data(self, cfg):
        self._train_dl = self.__setup_dataloader_from_config(self._cfg.train_ds)

    def setup_validation_data(self, cfg):
        self._validation_dl = CombinedLoader(self.__setup_dataloader_from_config(self._cfg.validation_ds))

    def forward(self, input_signal=None, input_signal_length=None):
        
        processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length,
            )

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length) #b,c,t

        for task in self._cfg.downstream_heads.task_names:
            if task == "speaker_verification":
                speaker_embedding = self.downstream_nets['speaker_verification'](encoded[:,:,0])
                l2_norm = torch.norm(speaker_embedding, p=2,dim=-1, keepdim=True)
                speaker_embedding_normalized = speaker_embedding/l2_norm
                speaker_logits = self.sv_linear(speaker_embedding_normalized)

            elif task == "content":
                encoded_btc = encoded.permute(0, 2, 1)
                content_embedding = self.downstream_nets['content'](encoded_btc)
                content_logits = self.content_linear(content_embedding)
                content_log_probs = content_logits.log_softmax(dim=2)
                content_log_probs = content_log_probs.permute(1, 0, 2) #t,b,c for ctc
                
                
        return speaker_logits, speaker_embedding_normalized, content_embedding, content_log_probs, encoded_len
        

    def training_step(self, batch, batch_idx):
        
        loss = 0.0
        for key in batch.keys():  
            if key == 'sv':
                signal = batch[key]['audio']
                signal_len = batch[key]['audio_lens']
                speaker_id = batch[key]['speaker_id']
                sv_logits, sv_emb, _, _, _ = self.forward(
                    input_signal=signal, input_signal_length=signal_len
                )
                
                pred_speaker = torch.argmax(sv_logits, dim=1)
                
                sv_loss = self.sv_loss(logits=sv_logits, labels=speaker_id)
                loss += sv_loss
                
                correct = pred_speaker.eq(speaker_id.data.view_as(pred_speaker)).sum().item()
                acc = (correct/len(speaker_id))*100
                
                self.log("t_sv_loss", sv_loss)
                self.log("t_sv_accuracy", acc)
            
            elif key == "content":
                signal = batch[key]['audio']
                signal_len = batch[key]['audio_lens']
                target = batch[key]['text'] # (B, T)
                target_len = batch[key]['text_lens']
                
                _, _, content_embedding, content_log_probs, encoded_len = self.forward(
                    input_signal=signal, input_signal_length=signal_len
                )

                ctc_loss = self.ctc_loss(content_log_probs, target, encoded_len, target_len)
                loss += ctc_loss

                self.log("t_content_loss", ctc_loss)


        self.log("t_loss", loss)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        
        loss_total = 0
        for key in batch.keys():
            if key == 'sv':
                signal = batch[key]['audio']
                signal_len = batch[key]['audio_lens']
                speaker_id = batch[key]['speaker_id']
                sv_logits, sv_emb, _, _, _ = self.forward(
                input_signal=signal, input_signal_length=signal_len
                )
                
                pred_speaker = torch.argmax(sv_logits, dim=1)
                sv_loss = self.sv_loss(logits=sv_logits, labels=speaker_id)
                loss_total += sv_loss

                correct = pred_speaker.eq(speaker_id.data.view_as(pred_speaker)).sum().item()
                acc = (correct/len(speaker_id))*100
                acc_val = torch.as_tensor(acc)

            if key == 'content':
                signal = batch[key]['audio']
                signal_len = batch[key]['audio_lens']
                target = batch[key]['text'] # (B, T)
                target_len = batch[key]['text_lens']
                
                _, _, content_embedding, content_log_probs, encoded_len = self.forward(
                    input_signal=signal, input_signal_length=signal_len
                )

                ctc_loss = self.ctc_loss(content_log_probs, target, encoded_len, target_len)
                loss_total +=  ctc_loss
                pred_char_batch = torch.argmax(content_log_probs, dim=2)
                pred_char_batch = pred_char_batch.permute(1,0)
                pred_char = decode(self._text_tokenizer, pred_char_batch[0].tolist() )
                target_char = decode(self._text_tokenizer, target[0].tolist() )
                
        
        return {
            'val_loss': loss_total,
            'sv_loss' : sv_loss,
            'ctc_loss' : ctc_loss,
            'accuracy_sv': acc_val
        }

    def validation_epoch_end(self, outputs):
        collect = lambda key: torch.stack([x[key] for x in outputs]).mean()
        val_loss = collect("val_loss")
        val_sv_loss = collect("sv_loss")
        val_ctc_loss = collect("ctc_loss")
        accuracy_sv = collect("accuracy_sv")
        self.log("val_loss", val_loss)
        self.log("sv_loss", val_sv_loss)
        self.log("ctc_loss", val_ctc_loss)
        self.log("accuracy_sv", accuracy_sv)

        


