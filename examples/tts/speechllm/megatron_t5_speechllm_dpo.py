# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from nemo.collections.asr.parts.utils import manifest_utils
import json
import os
import copy

def inference_for_records(model, trainer, cfg, records):
    model.predict_step_outputs = []
    model._test_ds.examples = []
    model._test_ds.examples = model._test_ds.load_data(records)
    sampler = torch.utils.data.distributed.DistributedSampler(
        model._test_ds, num_replicas=1, rank=0, shuffle=False, seed=1
    )
    model._test_dl = torch.utils.data.DataLoader(
        model._test_ds,
        collate_fn=model._test_ds.collate_fn,
        sampler=sampler,
        batch_size=cfg.model.micro_batch_size,
        drop_last=False,
        num_workers=1,
        pin_memory=False,
        persistent_workers=True
    )
    model.cfg.data.test_ds = None
    trainer.test(model, model._test_dl)
    inference_list_json = os.path.join(model.override_log_dir, "inference_lists.json")
    inference_metrics_json = os.path.join(model.override_log_dir, "output_metrics.json")

    with open(inference_list_json) as f:
        inference_list = json.load(f)

    with open(inference_metrics_json) as f:
        inference_metrics = json.load(f)

    return inference_list, inference_metrics

def generate_samples_and_compute_reward(model, trainer, cfg, current_state={}):
    model.override_log_dir = os.path.join(model.logger.log_dir, "rlhf_generations")
    model.predict_step_outputs = []
    # if os.path.exists(model.override_log_dir):
    #     shutil.rmtree(model.override_log_dir)

    if not os.path.exists(model.override_log_dir):
        os.makedirs(model.override_log_dir)

    train_records = []
    for train_ds in cfg.model.rlhf_train_ds:
        train_records.extend(manifest_utils.read_manifest(train_ds))

    train_records_repeated = []
    for record in train_records:
        for _ in range(cfg.model.rlhf_num_samples_per_example):
            train_records_repeated.append(copy.deepcopy(record))
    si = current_state.get('completed_samples', 0) % len(train_records_repeated)
    ei = si + cfg.model.rlhf_num_generations_per_iteration
    train_records_repeated = train_records_repeated[si:ei]

    inference_list, inference_metrics = inference_for_records(model, trainer, cfg, train_records_repeated)

    assert len(model._test_ds.examples) == len(inference_list['predicted_token_files'])

    new_records = []
    for ridx, record in enumerate(model._test_ds.examples):
        record['answer'] = inference_list['predicted_token_files'][ridx]
        record['answer_duration'] = inference_list['predicted_durations'][ridx]
        record['cer_gts'] = None
        record['wer_gts'] = None
        record['pred_context_similarity'] = None
        record['transcript_pred'] = None
        record['transcript_gt'] = None
        new_records.append(record)

    generated_outputs_manifest = os.path.join(model.override_log_dir, "generated_outputs_manifest.json")
    manifest_utils.write_manifest(generated_outputs_manifest, new_records)

    return {
        'completed_samples' : current_state.get('completed_samples', 0) + cfg.model.rlhf_num_generations_per_iteration,
        'generated_outputs_manifest' : generated_outputs_manifest,
    }

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

    mode = cfg.get("mode", "train")
    # load existing or init new soft prompt T5 model
    pretrained_checkpoint = cfg.get("init_from_ptl_ckpt", None)
    if cfg.model.get("restore_path", None) is not None:
        logging.info(f"cfg.model.restore_path {cfg.model.restore_path}")
        model = MegatronT5SpeechLMModel_DPO.restore_from(
            cfg.model.restore_path, cfg.model, trainer=trainer, save_restore_connector=NLPSaveRestoreConnector()
        )
    else:
        logging.info(f"cfg.model.restore_path is None")
        model = MegatronT5SpeechLMModel_DPO(cfg.model, trainer=trainer)
        model.maybe_init_from_pretrained_checkpoint(cfg=cfg)

    if mode == "train":
        model_ref = MegatronT5SpeechLMModel_DPO.load_from_checkpoint(
            checkpoint_path=pretrained_checkpoint, trainer=trainer, cfg=cfg.model
        )
        model_ref.eval()
        model_ref.dummy_test = True
        # To setup the reference model correctly
        trainer.test(model_ref)
        model.additional_models['model_ref'] = model_ref
        trainer.fit(model)
    elif mode == "generate":
        # dummy test to set things up
        model.dummy_test = True
        trainer.test(model)
        model.dummy_test = False
        model.save_only_output_tokens = True
        generate_samples_and_compute_reward(model, trainer, cfg)

if __name__ == '__main__':
    main()