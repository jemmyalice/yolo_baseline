# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import dataloader, distributed, Sampler, BatchSampler

from ultralytics.data.dataset import GroundingDataset, YOLODataset, YOLOMultiModalDataset, YOLODatasetf, YOLOMultiModalDatasetf
from ultralytics.data.loaders import (
    LOADERS,
    LoadImagesAndVideos,
    LoadPilAndNumpy,
    LoadScreenshots,
    LoadStreams,
    LoadTensor,
    SourceTypes,
    autocast_list,
)
from ultralytics.data.utils import IMG_FORMATS, PIN_MEMORY, VID_FORMATS
from ultralytics.utils import RANK, colorstr
from ultralytics.utils.checks import check_file


class InfiniteDataLoader(dataloader.DataLoader):
    """
    Dataloader that reuses workers.

    Uses same syntax as vanilla DataLoader.
    """

    def __init__(self, *args, **kwargs):
        """Dataloader that infinitely recycles workers, inherits from DataLoader."""
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()
    # 返回的是batch_sampler的sampler的长度，这通常是数据集的大小。
    def __len__(self):
        """Returns the length of the batch sampler's sampler."""
        return len(self.batch_sampler.sampler)
    # 返回的是batch_sampler的sampler的长度，这通常是数据集的大小。
    def __iter__(self):
        """Creates a sampler that repeats indefinitely."""
        # 不同dataloader分别进入这个iter，len(self)为batch个数 也就是数据集/batch
        for i in range(len(self)):
            # print(f'-----------------------------------------------------------------{i}')
            yield next(self.iterator)

    # 重置迭代器。这在你想要在训练过程中修改数据集的设置时非常有用。
    def reset(self):
        """
        Reset iterator.

        This is useful when we want to modify settings of dataset while training.
        """
        self.iterator = self._get_iterator()


# class InfiniteMutilDataLoader(dataloader.DataLoader):
#     """
#     DataLoader that infinitely recycles workers.
#     Accepts external `batch_sampler` directly for better flexibility.
#     """
#     def __init__(self, *args, batch_sampler=None, **kwargs):
#         if batch_sampler is not None:
#             kwargs['batch_sampler'] = batch_sampler  # 替换默认逻辑
#             kwargs.pop('shuffle', None)  # batch_sampler 不支持 shuffle 参数
#         super().__init__(*args, **kwargs)
#         self.iterator = super().__iter__()
#
#     def __len__(self):
#         """Returns the length of the batch sampler's sampler."""
#         return len(self.batch_sampler)
#
#     def __iter__(self):
#         """Creates an infinite iterator."""
#         for _ in range(len(self)):
#             yield next(self.iterator)
#
#     def reset(self):
#         """Reset the iterator for flexibility."""
#         self.iterator = self._get_iterator()

class _RepeatSampler:
    """
    Sampler that repeats forever.

    Args:
        sampler (Dataset.sampler): The sampler to repeat.
    """

    def __init__(self, sampler):
        """Initializes an object that repeats a given sampler indefinitely."""
        self.sampler = sampler

    def __iter__(self):
        """Iterates over the 'sampler' and yields its contents."""
        while True:
            yield from iter(self.sampler)

    # def __len__(self):
    #     # 返回底层 sampler 的长度（如果存在）
    #     if hasattr(self.sampler, "__len__"):
    #         return len(self.sampler)
    #     else:
    #         raise TypeError(f"{type(self.sampler)} object has no len()")

class PairedSampler(Sampler):
    def __init__(self, dataset1, dataset2, shuffle=True, seed=None):
        """
        Paired Sampler ensures the indices of two datasets are aligned.
        """
        assert len(dataset1) == len(dataset2), "Datasets must have the same length!"
        self.dataset_length = len(dataset1)
        self.shuffle = shuffle
        self.indices = list(range(self.dataset_length))
        self.seed = seed  # 如果传入 seed，就使用固定种子；否则动态生成
        self.epoch_shuffle = True  # 用于缓存当前epoch的索引

    def reset(self):
        """在每个epoch结束时重置索引缓存"""
        self.epoch_shuffle = True

    def _generate_seed(self):
        """生成随机种子，优先使用用户指定的种子；否则动态生成"""
        return self.seed if self.seed is not None else int(time.time() * 1000) % 2 ** 32

    def __iter__(self):
        if self.shuffle and self.epoch_shuffle:  # 如果当前没有索引，就生成新的
            # Shuffle indices for both datasets at the same time
            self.epoch_shuffle = False
            seed = self._generate_seed()
            random.seed(seed)  # 设置随机种子
            random.shuffle(self.indices)
        # print("PairedSampler indices:", self.indices)  # 打印返回的索引
        return iter(self.indices)

    def __len__(self):
        return self.dataset_length

def seed_worker(worker_id):  # noqa
    """Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def build_yolo_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32, multi_modal=False):
    """Build YOLO Dataset."""
    dataset = YOLOMultiModalDataset if multi_modal else YOLODataset
    # if isinstance(img_path, list):
    #     dataset = YOLOMultiModalDatasetf if multi_modal else YOLODatasetf
    return dataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )

def build_grounding(cfg, img_path, json_file, batch, mode="train", rect=False, stride=32):
    """Build YOLO Dataset."""
    return GroundingDataset(
        img_path=img_path,
        json_file=json_file,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )

def build_dataloader(dataset, batch, workers, shuffle=True, rank=-1):
    """返回InfiniteDataLoader或DataLoader用于训练或验证集"""
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min(os.cpu_count() // max(nd, 1), workers)  # number of workers
    # 设置数据采样器
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK) # 表示当前进程在分布式训练中的编号
    return InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=PIN_MEMORY,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
    )

def build_mutil_dataloader(dataset1, dataset2, batch, workers, shuffle=True, rank=-1):
    """Returns synchronized MultiInfiniteDataLoader."""
    """返回InfiniteDataLoader或DataLoader用于训练或验证集"""
    assert len(dataset1)==len(dataset2), "Datasets must have the same size for pairing!"
    batch = min(batch, len(dataset1))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min(os.cpu_count() // max(nd, 1), workers)  # number of workers
    # 设置数据采样器
    paired_sampler = PairedSampler(dataset1, dataset2, shuffle=shuffle)

    # 初始化两个 Sampler，使用相同的种子
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK) # 表示当前进程在分布式训练中的编号
    loader1 = InfiniteDataLoader(
        dataset=dataset1,
        batch_size=batch,
        shuffle=shuffle and paired_sampler is None,
        num_workers=nw,
        sampler=paired_sampler,
        pin_memory=PIN_MEMORY,
        collate_fn=getattr(dataset1, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
    )
    loader2 = InfiniteDataLoader(
        dataset=dataset2,
        batch_size=batch,
        shuffle=shuffle and paired_sampler is None,
        num_workers=nw,
        sampler=paired_sampler,
        pin_memory=PIN_MEMORY,
        collate_fn=getattr(dataset2, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
    )
    # # 包装 PairedSampler，让其无限重复
    # repeated_sampler = _RepeatSampler(paired_sampler)
    #
    # # 创建共享的 BatchSampler
    # batch_sampler = BatchSampler(
    #     sampler=repeated_sampler,  # 使用 _RepeatSampler 包装后的 PairedSampler
    #     batch_size=batch,
    #     drop_last=False
    # )
    # # Create DataLoaders
    # loader1 = InfiniteMutilDataLoader(
    #     dataset=dataset1,
    #     batch_sampler=batch_sampler,
    #     shuffle=shuffle and paired_sampler is None,
    #     num_workers=nw,
    #     pin_memory=PIN_MEMORY,
    #     collate_fn=getattr(dataset1, "collate_fn", None),
    #     worker_init_fn=seed_worker,
    #     generator=generator,
    # )
    #
    # loader2 = InfiniteMutilDataLoader(
    #     dataset=dataset2,
    #     batch_sampler=batch_sampler,
    #     shuffle=shuffle and paired_sampler is None,
    #     num_workers=nw,
    #     pin_memory=PIN_MEMORY,
    #     collate_fn=getattr(dataset2, "collate_fn", None),
    #     worker_init_fn=seed_worker,
    #     generator=generator,
    # )
    return loader1, loader2

def check_source(source):
    """Check source type and return corresponding flag values."""
    webcam, screenshot, from_img, in_memory, tensor = False, False, False, False, False
    if isinstance(source, (str, int, Path)):  # int for local usb camera
        source = str(source)
        is_file = Path(source).suffix[1:] in (IMG_FORMATS | VID_FORMATS)
        is_url = source.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://"))
        webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
        screenshot = source.lower() == "screen"
        if is_url and is_file:
            source = check_file(source)  # download
    elif isinstance(source, LOADERS):
        in_memory = True
    elif isinstance(source, (list, tuple)):
        source = autocast_list(source)  # convert all list elements to PIL or np arrays
        from_img = True
    elif isinstance(source, (Image.Image, np.ndarray)):
        from_img = True
    elif isinstance(source, torch.Tensor):
        tensor = True
    else:
        raise TypeError("Unsupported image type. For supported types see https://docs.ultralytics.com/modes/predict")

    return source, webcam, screenshot, from_img, in_memory, tensor


def load_inference_source(source=None, batch=1, vid_stride=1, buffer=False):
    """
    Loads an inference source for object detection and applies necessary transformations.

    Args:
        source (str, Path, Tensor, PIL.Image, np.ndarray): The input source for inference.
        batch (int, optional): Batch size for dataloaders. Default is 1.
        vid_stride (int, optional): The frame interval for video sources. Default is 1.
        buffer (bool, optional): Determined whether stream frames will be buffered. Default is False.

    Returns:
        dataset (Dataset): A dataset object for the specified input source.
    """
    source, stream, screenshot, from_img, in_memory, tensor = check_source(source)
    source_type = source.source_type if in_memory else SourceTypes(stream, screenshot, from_img, tensor)

    # Dataloader
    if tensor:
        dataset = LoadTensor(source)
    elif in_memory:
        dataset = source
    elif stream:
        dataset = LoadStreams(source, vid_stride=vid_stride, buffer=buffer)
    elif screenshot:
        dataset = LoadScreenshots(source)
    elif from_img:
        dataset = LoadPilAndNumpy(source)
    else:
        dataset = LoadImagesAndVideos(source, batch=batch, vid_stride=vid_stride)

    # Attach source types to the dataset
    setattr(dataset, "source_type", source_type)

    return dataset
