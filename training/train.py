import copy
import os
import time
from pathlib import Path
from typing import Union, Optional

import click
import mmcv
import torch
from mmcv import Config
from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger
from dataclasses import dataclass, asdict


@dataclass
class SubsetStructure:
    annotation_path: str
    image_dir: str


@dataclass
class COCODatasetStructure:
    train: SubsetStructure
    val: SubsetStructure
    test: SubsetStructure


def override_dataset_structure(cfg: Config, new_struct: COCODatasetStructure) -> Config:
    _struct = asdict(new_struct)
    for subset in _struct.keys():
        cfg.data[subset]['ann_file'] = _struct[subset]['annotation_path']
        cfg.data[subset]['img_prefix'] = _struct[subset]['image_dir']
    return cfg


def run_training(
    cfg_path: Path,
    work_dir: Optional[Union[str, Path]] = None,
    load_from: Optional[Union[str, Path]] = None,
    resume_from: Optional[Union[str, Path]] = None,
    dataset_structure: Optional[COCODatasetStructure] = None,
    seed: Optional[int] = None,
    validate: bool = True,
    epochs: int = 40,
):
    """
    Execute training pipeline.
    :param epochs: total epochs for train
    :param cfg_path: Path to model`s config.
    :param work_dir: Path to the directory, where all training assets are stored.
    :param load_from: Checkpoint to load model from.
    :param resume_from: Checkpoint to continue learning from.
    :param dataset_structure: Configuration of dataset paths, that overrides the
    default config.
    :param seed: Random seed for deterministic results.
    :param validate: Execute validation pipeline.
    :return:
    """
    cfg = Config.fromfile(cfg_path)

    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    if work_dir is not None:
        cfg.work_dir = work_dir
    if load_from is not None:
        cfg.load_from = load_from
    if resume_from is not None:
        cfg.resume_from = resume_from
    if dataset_structure is not None:
        cfg = override_dataset_structure(cfg, dataset_structure)
    if epochs is not None:
        cfg.total_epochs = epochs

    # Apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
    cfg.optimizer["lr"] = cfg.optimizer["lr"] * len(cfg.gpu_ids) / 8

    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(cfg.work_dir, "{}.log".format(timestamp))
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # Init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()

    # Log env info
    env_info_dict = collect_env()
    env_info = "\n".join([("{}: {}".format(k, v)) for k, v in env_info_dict.items()])
    dash_line = "-" * 60 + "\n"
    logger.info("Environment info:\n" + dash_line + env_info + "\n" + dash_line)
    meta["env_info"] = env_info

    # log the config itself
    logger.info(f"Config:\n{cfg.text}")

    # set random seeds
    if seed is not None:
        logger.info(f"Set random seed to {seed}")
        set_random_seed(seed, deterministic=True)
    cfg.seed = seed
    meta["seed"] = seed

    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # Save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__, config=cfg.text, CLASSES=datasets[0].CLASSES
        )
    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=validate,
        timestamp=timestamp,
        meta=meta,
    )


def get_paths_by_set(dataset_path: Path, set_name: str) -> SubsetStructure:
    set_path = dataset_path / set_name
    return SubsetStructure(
        str((set_path / f"{set_name}.json").absolute()),
        str((set_path / "images").absolute())
    )


def get_dataset_struct(dataset_path: Path) -> COCODatasetStructure:
    return COCODatasetStructure(*[get_paths_by_set(dataset_path, subset) for subset in ("train", "val", "test")])


@click.command()
@click.argument("dataset_path", type=str)
@click.argument("working_dir", type=str)
@click.option("--load-from", type=str)
@click.option("--resume-from", type=str)
@click.option("--config", type=str)
@click.option("--num-epoch", type=int)
def train(dataset_path,
          working_dir,
          load_from,
          resume_from,
          config,
          num_epoch
          ):
    coco_struct = get_dataset_struct(Path(dataset_path))
    working_dir_path = Path(working_dir)
    working_dir_path.mkdir(exist_ok=True, parents=True)
    config_path = Path(config) if config else Path(__file__).parent.parent.joinpath(
        "configs/config_3_cls_w18.py"
    )

    run_training(
        config_path,
        work_dir=str(working_dir_path.absolute()),
        dataset_structure=coco_struct,
        load_from=load_from,
        resume_from=resume_from,
        seed=None,
        validate=True,
        epochs=num_epoch
    )


if __name__ == '__main__':
    train()
