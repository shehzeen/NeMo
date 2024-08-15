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

from nemo.collections.tts.models.speechllm.megatron_t5_speechllm_model import MegatronT5SpeechLMModel
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
import json
import os
import shutil
import copy

def read_records(manifest_path):
    with open(manifest_path, 'r') as f:
        lines = f.readlines()
        records = []
        for line in lines:
            records.append(json.loads(line.strip()))
    return records

def write_records(fp, records):
    with open(fp, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

def create_dummy_manifest(exp_dir):
    dummy_codes = torch.ones(8, 300).cpu().type(torch.int16)
    dummy_codes_fp = os.path.join(exp_dir, "dummy_codes.pt")
    torch.save(dummy_codes, dummy_codes_fp)
    dummy_record = {
        "question" : "Phoneme TTS Sample Text",
        "answer" : dummy_codes_fp,
        "context" : dummy_codes_fp,
        "context_type" : "REFSPEAKERCODEC",
        "question_type" : "TEXT",
        "answer_type" : "AUDIOCODEC",
        "context_duration" : 5.0,
        "answer_duration" : 5.0,
        "taskname" : "squad"
    }

    dummy_val_file = os.path.join(exp_dir, "dummy_val.json")
    write_records(dummy_val_file, [dummy_record])
    return dummy_val_file


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
        batch_size=cfg.model.rlhf.generation_batch_size,
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
    for train_ds in cfg.model.rlhf.train_ds:
        train_records.extend(read_records(train_ds))

    train_records_repeated = []
    for record in train_records:
        for _ in range(cfg.model.rlhf.num_samples_per_example):
            train_records_repeated.append(copy.deepcopy(record))
    si = current_state.get('completed_samples', 0) % len(train_records_repeated)
    ei = si + cfg.model.rlhf.num_generations_per_iteration
    train_records_repeated = train_records_repeated[si:ei]
    
    inference_list, inference_metrics = inference_for_records(model, trainer, cfg, train_records_repeated)
    
    assert len(model._test_ds.examples) == len(inference_list['predicted_token_files'])

    new_records = []
    for ridx, record in enumerate(model._test_ds.examples):
        record['answer'] = inference_list['predicted_token_files'][ridx]
        record['answer_duration'] = inference_list['predicted_durations'][ridx]
        record['cer_gts'] = inference_list['cer_gts'][ridx]
        record['wer_gts'] = inference_list['wer_gts'][ridx]
        record['pred_context_similarity'] = inference_list['pred_context_similarity'][ridx]
        new_records.append(record)
    
    ridx = 0
    rlhf_records = []
    while ridx < len(new_records):
        sample_group = copy.deepcopy(new_records[ridx:ridx+cfg.model.rlhf.num_samples_per_example])
        
        cer_sim_indices = []
        for sidx, record in enumerate(sample_group):
            cer_sim_indices.append((record['cer_gts'], -record['pred_context_similarity'], sidx))
        
        # sort by cer and similarity
        cer_sim_indices = sorted(cer_sim_indices)
        if cer_sim_indices[0][0] < cfg.model.rlhf.cer_threshold:
            best_record = copy.deepcopy(sample_group[cer_sim_indices[0][2]])
            worst_record = copy.deepcopy(sample_group[cer_sim_indices[-1][2]])
            best_record['reward'] = 1.0
            worst_record['reward'] = -0.1
            rlhf_records.append(best_record)
            rlhf_records.append(worst_record)
        
        ridx += cfg.model.rlhf.num_samples_per_example
    
    rlhf_manifest = os.path.join(model.override_log_dir, "rlhf_manifest.json")
    write_records(rlhf_manifest, rlhf_records)
    model.override_log_dir = None
    print(rlhf_manifest)
    # import ipdb; ipdb.set_trace()
    
    return {
        'completed_samples' : current_state.get('completed_samples', 0) + cfg.model.rlhf.num_generations_per_iteration,
        'rlhf_manifest' : rlhf_manifest,
        'inference_metrics' : inference_metrics,
    }


def train_on_rewards(model, trainer, cfg, rlhf_manifest):
    train_records = read_records(rlhf_manifest)
    model._train_ds.examples = []
    model._train_ds.examples = model._train_ds.load_data(train_records)
    sampler = torch.utils.data.distributed.DistributedSampler(
        model._train_ds, num_replicas=1, rank=0, shuffle=False, seed=1
    )
    model._train_dl = torch.utils.data.DataLoader(
        model._train_ds,
        collate_fn=model._train_ds.collate_fn,
        sampler=sampler,
        batch_size=cfg.model.rlhf.train_batch_size,
        drop_last=False,
        num_workers=1,
        pin_memory=False,
        persistent_workers=True
    )
    model.cfg.data.train_ds = None
    trainer.fit(model, model._train_dl)

@hydra_runner(config_path="conf", config_name="megatron_t5_speechllm_medium.yaml")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    # MegatronTrainerBuilder compat checks
    if "gradient_as_bucket_view" not in cfg.model:
        with open_dict(cfg):
            cfg.model.gradient_as_bucket_view = False
    
    trainer = MegatronTrainerBuilder(cfg).create_trainer()
    exp_manager(trainer, cfg.exp_manager)

    dummy_test_file = create_dummy_manifest(cfg.exp_manager.exp_dir)
    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(cfg):
        cfg.model.precision = cfg.trainer.precision
        cfg.model.data.test_ds = [dummy_test_file]
        cfg.model.data.train_ds = [dummy_test_file]

    # load existing or init new soft prompt T5 model
    if cfg.model.get("restore_path", None) is not None:
        logging.info(f"cfg.model.restore_path {cfg.model.restore_path}")
        model = MegatronT5SpeechLMModel.restore_from(
            cfg.model.restore_path, cfg.model, trainer=trainer, save_restore_connector=NLPSaveRestoreConnector()
        )
    else:
        logging.info(f"cfg.model.restore_path is None")
        model = MegatronT5SpeechLMModel(cfg.model, trainer=trainer)
        model.maybe_init_from_pretrained_checkpoint(cfg=cfg)

    
    checkpoint_dir = os.path.join(model.logger.log_dir, "checkpoints")
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)

    # dummy test to set things up
    trainer.test(model)
    model.setup_training_data()
    generation_results = {}
    for num_iter in range(cfg.trainer.max_steps):
        generation_results = generate_samples_and_compute_reward(model, trainer, cfg, generation_results)
        
        # Save inference metrics
        with open(os.path.join(model.logger.log_dir, f"inference_metrics_{num_iter}.json"), "w") as f:
            json.dump(generation_results['inference_metrics'], f)

        train_on_rewards(model, trainer, cfg, generation_results['rlhf_manifest'])
        # Remove checkpoint files
        
        checkpoint_dir = os.path.join(model.logger.log_dir, "checkpoints")
        # Backup checkpoint
        if os.path.exists(checkpoint_dir):
            if num_iter % 5 == 0:
                shutil.copytree(checkpoint_dir, os.path.join(model.logger.log_dir, f"checkpoints_{num_iter}"))
            shutil.rmtree(checkpoint_dir)
        
        if num_iter > 3 and num_iter % 4 == 0:
            # Inference on Test Set
            model.override_log_dir = os.path.join(model.logger.log_dir, f"test_inference_{num_iter}")
            if not os.path.exists(model.override_log_dir):
                os.makedirs(model.override_log_dir)
            test_records = read_records(cfg.model.rlhf.test_ds)
            _, test_metrics = inference_for_records(model, trainer, cfg, test_records)
            model.override_log_dir = None

        # Reset trainer
        trainer = MegatronTrainerBuilder(cfg).create_trainer()
        exp_manager(trainer, cfg.exp_manager)
        model.trainer = trainer


if __name__ == '__main__':
    main()
