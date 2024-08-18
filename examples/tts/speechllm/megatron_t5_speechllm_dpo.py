# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import torch
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf, open_dict

from nemo.collections.tts.models.speechllm.megatron_t5_speechllm_model import MegatronT5SpeechLMModel_DPO
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="megatron_t5_speechllm_medium.yaml")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    # MegatronTrainerBuilder compat checks
    if "gradient_as_bucket_view" not in cfg.model:
        with open_dict(cfg):
            cfg.model.gradient_as_bucket_view = False

    trainer = MegatronTrainerBuilder(cfg).create_trainer()
    # trainer_ref = MegatronTrainerBuilder(cfg).create_trainer()
    exp_manager(trainer, cfg.exp_manager)
    # exp_manager(trainer_ref, cfg.exp_manager)

    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(cfg):
        cfg.model.precision = cfg.trainer.precision

    # load existing or init new soft prompt T5 model
    if cfg.model.get("restore_path", None) is not None:
        logging.info(f"cfg.model.restore_path {cfg.model.restore_path}")
        model = MegatronT5SpeechLMModel_DPO.restore_from(
            cfg.model.restore_path, cfg.model, trainer=trainer, save_restore_connector=NLPSaveRestoreConnector()
        )
    else:
        logging.info(f"cfg.model.restore_path is None")
        model = MegatronT5SpeechLMModel_DPO(cfg.model, trainer=trainer)
        model.maybe_init_from_pretrained_checkpoint(cfg=cfg)

    model_ref = MegatronT5SpeechLMModel_DPO.load_from_checkpoint(
        checkpoint_path="/Data/JunEOSCheckpoints/desta_less_sophia_213850.ckpt", trainer=trainer, cfg=cfg.model
    )
    model_ref.eval()
    # To setup the reference model correctly
    trainer.test(model_ref)

    model.additional_models['model_ref'] = model_ref
    trainer.fit(model)

if __name__ == '__main__':
    main()
