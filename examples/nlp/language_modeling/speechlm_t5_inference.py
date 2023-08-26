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
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import TorchElasticEnvironment

from nemo.collections.nlp.models.language_modeling.megatron_t5_speechlm_model import MegatronT5SpeechLMModel

# from nemo.collections.nlp.models.language_modeling.megatron_t5_speechlm_pretrain_model import (
#     MegatronT5SpeechLMModel,
# )
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    NLPDDPStrategy,
    NLPSaveRestoreConnector,
    PipelineMixedPrecisionPlugin,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager



@hydra_runner(config_path="conf", config_name="speechlm_inference.yaml")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    megatron_amp_o2 = cfg.model.get('megatron_amp_O2', False)
    with_distributed_adam = cfg.model.optim.get('name') == 'distributed_fused_adam'

    plugins = []
    strategy = NLPDDPStrategy(
        no_ddp_communication_hook=True,  # we don't use DDP for async grad allreduce
        gradient_as_bucket_view=False,
        find_unused_parameters=False,
    )
    if cfg.trainer.precision in [16, 'bf16']:
        scaler = None
        if cfg.trainer.precision == 16:
            scaler = GradScaler(
                init_scale=cfg.model.get('native_amp_init_scale', 2 ** 8),
                growth_interval=cfg.model.get('native_amp_growth_interval', 1000),
                hysteresis=cfg.model.get('hysteresis', 2),
            )
        if megatron_amp_o2 and not with_distributed_adam:
            plugins.append(MegatronHalfPrecisionPlugin(precision=cfg.trainer.precision, device='cuda', scaler=scaler))
        else:
            plugins.append(PipelineMixedPrecisionPlugin(precision=cfg.trainer.precision, device='cuda', scaler=scaler))

    if cfg.get('cluster_type', None) == 'BCP':
        plugins.append(TorchElasticEnvironment())

    trainer = Trainer(plugins=plugins, strategy=strategy, **cfg.trainer)
    exp_manager(trainer, cfg.exp_manager)

    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(cfg):
        cfg.model.precision = cfg.trainer.precision

    # checkpoint_path = "/home/shehzeenh/Code/NeMo/nemo_experiments/p_tuning_squad_t5/checkpoints/Step34770.ckpt"
    # checkpoint_path = "/Data/Experiments/full_data_tts/p_tuning_squad_t5/checkpoints/Step46794.ckpt"
    # checkpoint_path = "/Data/Experiments/full_data_tts/p_tuning_squad_t5/checkpoints/Step100k.ckpt"
    # checkpoint_path = "/Data/Experiments/NewDatasetClassOverfitting1000/p_tuning_squad_t5/checkpoints/Step164164.ckpt"
    # checkpoint_path = "/Data/Experiments/Selene_Runs/Run512_24000_AdamW-220M-unfreeze-tts_lr5e-5_nosched_posemb512_FP32/p_tuning_squad_t5/2023-08-14_19-49-38/checkpoints/Step32697.ckpt"
    # checkpoint_path = "/Data/Experiments/FromPretrain30k.ckpt"
    # checkpoint_path = "/Data/Experiments/breakCode_30ktrainingExamples_512/p_tuning_squad_t5/checkpoints/Step38043.ckpt"
    # checkpoint_path = "/Data/Experiments/breakCode_30ktrainingExamples_512/Step16k.ckpt"
    # checkpoint_path = "/Data/Experiments/Selene_Runs/Run512_24000_AdamW-220M-unfreeze-tts_lr5e-5_nosched_posemb512_FP32/p_tuning_squad_t5/2023-08-14_19-49-38/checkpoints/Step47229.ckpt"
    # checkpoint_path = "/Data/Experiments/Selene_Runs/NewDL_24000_AdamW-220M-unfreeze-tts_lr5e-5_nosched_posemb1536_FP32/p_tuning_squad_t5/2023-08-14_19-49-38_latest/checkpoints/Step89200.ckpt"
    # checkpoint_path = "/Data/Experiments/ComparisonBreakUnbreak/NoBreak/p_tuning_squad_t5/checkpoints/Step29k.ckpt"
    # checkpoint_path = "/Data/Experiments/ComparisonBreakUnbreak/WithBreak/p_tuning_squad_t5/checkpoints/Step82348.ckpt"
    checkpoint_path = "/Data/Experiments/ComparisonBreakUnbreak/NoBreak/p_tuning_squad_t5/checkpoints/Step82k.ckpt"
    model = MegatronT5SpeechLMModel.load_from_checkpoint(
        checkpoint_path=checkpoint_path, trainer=trainer, cfg=cfg.model
    )
    model.eval()
    model = model.cuda()
    trainer.test(model)


if __name__ == '__main__':
    main()
