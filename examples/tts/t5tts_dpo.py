# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytorch_lightning as pl
from omegaconf import OmegaConf, open_dict

from nemo.collections.tts.models import T5TTS_Model, T5TTS_ModelDPO
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

@hydra_runner(config_path="conf/t5tts", config_name="t5tts")
def main(cfg):
    logging.info('\nConfig Params:\n%s', OmegaConf.to_yaml(cfg, resolve=True))
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    model_cfg = cfg.model
    with open_dict(model_cfg):
        model_cfg.reference_model_ckpt_path = cfg.init_from_ptl_ckpt
    model = T5TTS_ModelDPO(cfg=model_cfg, trainer=trainer)
    model.maybe_init_from_pretrained_checkpoint(cfg=cfg)
    trainer.fit(model)

if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
