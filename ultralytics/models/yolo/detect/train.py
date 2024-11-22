# Ultralytics YOLO 🚀, AGPL-3.0 license

import math
import random
from copy import copy

import numpy as np
import torch.nn as nn

from ultralytics.data import build_dataloader, build_yolo_dataset, build_mutil_dataloader
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first


class DetectionTrainer(BaseTrainer):
    """
    一个扩展BaseTrainer类的类，用于基于检测模型进行训练。

    Example:
        ```python
        from ultralytics.models.yolo.detect import DetectionTrainer

        # 生成字典再传传入
        args = dict(model="yolo11n.pt", data="coco8.yaml", epochs=3)
        trainer = DetectionTrainer(overrides=args)
        trainer.train()
        ```
    """

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        生成yolo DataSets.

        Args:
            img_path (str): 包含图像的文件夹的路径。
            mode (str): `train` mode or `val` mode, 用户可以为每种模式定制不同的增强功能。
            batch (int, optional): 批量大小，这是针对“rect”的。默认为“None”。
        """
        # de_parallel(self.model).stride.max()取出模型中最大的步长 然后gs=stride max和32中的max
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)

        # self.data是解析后的数据集全部的信息和路径。 img_path是训练/测试的数据集 返回一个pytroch重构的dataset对象
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """构建并且返回一个 dataloader."""
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2

        # 返回一个pytroch重构的dataloader对象
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)

    def get_mutil_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """构建并且返回两个 dataloader."""
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            if isinstance(dataset_path, list):
                dataset1 = self.build_dataset(dataset_path[0], mode, batch_size) # 0是上面的为vis
                dataset2 = self.build_dataset(dataset_path[1], mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset1, "rect", False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        if getattr(dataset2, "rect", False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2

        # 返回一个pytroch重构的dataloader对象
        return build_mutil_dataloader(dataset1, dataset2, batch_size, workers, shuffle, rank)

    def preprocess_batch(self, batch):
        """通过缩放和转换为浮点数来预处理一批图像。"""
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        if self.args.multi_scale:
            imgs = batch["img"]
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def preprocess_multi_batch(self, batch):
        """
        对两个数据源（ir 和 rgb）同时进行统一的预处理，确保尺寸一致。

        Args:
            batch (dict): 包含 'ir' 和 'rgb' 数据的字典。

        Returns:
            dict: 预处理后的 batch，'ir' 和 'rgb' 数据具有一致的尺寸。
        """
        # 处理 rgb 图像
        batch["rgb"]["img"] = batch["rgb"]["img"].to(self.device, non_blocking=True).float() / 255
        # 处理 ir 图像
        batch["ir"]["img"] = batch["ir"]["img"].to(self.device, non_blocking=True).float() / 255

        if self.args.multi_scale:
            # 获取 ir 和 rgb 的最大尺寸
            imgs_rgb = batch["rgb"]["img"]
            imgs_ir = batch["ir"]["img"]
            max_dim = max(
                max(batch["rgb"]["img"].shape[2:]),  # rgb 最大尺寸
                max(batch["ir"]["img"].shape[2:])  # ir 最大尺寸
            )
            # 计算目标尺寸（统一为 stride 的倍数）
            sz = (
                    random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                    // self.stride
                    * self.stride
            )  # size
            # 计算缩放因子
            sf = sz / max_dim  # scale factor
            if sf!=1:
                # 计算新的尺寸
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs_ir.shape[2:]
                ]  # new shape (stretched to gs-multiple)

                # 对 ir 和 rgb 同步插值缩放
                imgs_rgb = nn.functional.interpolate(imgs_rgb, size=ns, mode="bilinear", align_corners=False)
                imgs_ir = nn.functional.interpolate(imgs_ir, size=ns, mode="bilinear", align_corners=False)
            batch["rgb"]["img"] = imgs_rgb
            batch["ir"]["img"] = imgs_ir
        return batch["rgb"], batch["ir"]

    def set_model_attributes(self):
        """Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)."""
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  #scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.names = self.data["names"]  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        # 最后一行的注释可能是一个待办事项，表示作者计划在未来添加一个根据数据集标签计算类别权重并将其附加到模型上的功能。
        # 这可能是为了处理类别不平衡问题，即数据集中某些类别的样本数量比其他类别多得多。
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    # 从这里进入的DetectionModel
    def get_model(self, cfg=None, weights=None, verbose=True):
        """返回 a YOLO detection model."""
        # 从这里进入的DetectionModel
        model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)

        # 如果有预训练权重进行加载
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """返回 a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss" # 三个损失的名称
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def get_mutil_validator(self):
        """返回 a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss" # 三个损失的名称
        return yolo.detect.DetectionValidator(
            [self.test_loader, self.test_ir_loader], save_dir=self.save_dir,
            args=copy(self.args), _callbacks=self.callbacks, infusion = True
        )

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        返回一个带有标记的训练损失项张量的损失字典。

        Not needed for classification but necessary for segmentation & detection
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    # 返回每个epoch的加载信息
    def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    # 下面三个都是画图
    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        plot_images(
            images=batch["img"],
            batch_idx=batch["batch_idx"],
            cls=batch["cls"].squeeze(-1),
            bboxes=batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg", #文件名
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        """Plots metrics from a CSV file."""
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)
