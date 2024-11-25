# Ultralytics YOLO 🚀, AGPL-3.0 license
from .AddModules import *
import contextlib
import pickle
import re
import types
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.nn.modules import (
    AIFI,
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    ELAN1,
    OBB,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    AConv,
    ADown,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    Classify,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Pose,
    RepC3,
    RepConv,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    RTDETRDecoder,
    SCDown,
    Segment,
    WorldDetect,
    v10Detect,
)
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, colorstr, emojis, yaml_load
from ultralytics.utils.checks import check_requirements, check_suffix, check_yaml
from ultralytics.utils.loss import (
    E2EDetectLoss,
    v8ClassificationLoss,
    v8DetectionLoss,
    v8OBBLoss,
    v8PoseLoss,
    v8SegmentationLoss,
)
from ultralytics.utils.ops import make_divisible
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.torch_utils import (
    fuse_conv_and_bn,
    fuse_deconv_and_bn,
    initialize_weights,
    intersect_dicts,
    model_info,
    scale_img,
    time_sync,
)

try:
    import thop
except ImportError:
    thop = None


class BaseModel(nn.Module):
    """The BaseModel class serves as a base class for all the models in the Ultralytics YOLO family."""

    def forward(self, x, *args, **kwargs):
        """
            执行模型的前向传递，用于训练或推理。
            如果x是一个字典，则计算并返回训练的损失。否则，返回用于推理的预测。

        Args:
            x（torch.Tensor|dict）：用于推理的输入张量，或用于训练的带图像张量和标签的dict。
            *args（Any）：可变长度参数列表。
            **kwargs（Any）：任意关键字参数。

        Returns:
            (torch.Tensor): 如果x是字典（训练）或网络预测（推理），则进入loss。
        """
        # 传入的x是一个batch字典，其中batch["ir"] batch["rgb"]又为一个字典，其中有取文件的信息
        # 训练后验证进去虽然传入的是一个batch，但是preds不为None，不影响
        if isinstance(x, dict):  # 适用于训练和训练时验证的情况。
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs) # 这进行了网络构建，预测也会进这个

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        """
        通过网络执行前向传递。
        Perform a forward pass through the network.

        Args:
            x（torch.Tensor）：模型的输入张量。
            profile（bool）：如果为True，则打印每一层的计算时间，默认为False。
            visualize（bool）：如果为True，则保存模型的特征图，默认为False。
            auction（bool）：在预测过程中增强图像，默认为False。
            embed（list，可选）：要返回的特征向量/嵌入的列表。

        Returns:
            (torch.Tensor): The last output of the model.
        """
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize, embed)

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """
        通过网络执行前向传递。
        生成网络结构图的
        Perform a forward pass through the network.

        Args:
            x（torch.Tensor）：模型的输入张量。
            profile（bool）：如果为True，则打印每一层的计算时间，默认为False。
            visualize（bool）：如果为True，则保存模型的特征图，默认为False。
            embed（list，可选）：要返回的特征向量/嵌入的列表。

        Returns:
            (torch.Tensor): The last output of the model.
        """
        # train是时候传进来的第一维为rgb，第二维为ir
        y, dt, embeddings = [], [], []  # outputs
        # x存储每一个块的输入， y存储每一个块的输出
        for m in self.model:
            if m.f != -1:  # if not from previous layer 这就是需要concat多重输入的部分
                # 如果 m.f（前一层层索引） 是整数，从 y 中直接取索引为 m.f 的元素，赋值给 x
                # 为list则 对 m.f 的每个元素 j： 如果 j == -1，保留当前的 x。 如果 j != -1，从 y 中取索引为 j 的元素。
                x = y[m.f] if isinstance(m.f, int) else [x if j==-1 else y[j] for j in m.f]

                #增加两种情况，一种是训练从batch中读入x为list，另一种是构建模型框架时第一层需要两个输入的情况
                # if -2 not in m.f:
                #     x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                # # ultralytics/nn/tasks.py:317训练时候要走这里，这里对于两个输入进行特殊处理
                # # 因为一开始进来的时候传的是list所以可以这样
                # elif isinstance(x, list):
                #     x = x
                # else:# 在yolo模型定义时第一次建造网络时传入的是一个tensor，使用infusion
                #     x = [x, x]
            if profile:
                self._profile_one_layer(m, x, dt)

            # 把要输入给网络的内容 输入进去！
            # concat应该是把两个输入x作为一个整体list直接输入然后forward中连接
            # 然而我需要的是两个输入到m中

            # x = m(x)

            if m.i != 0:
                x = m(x)  # run
            elif isinstance(x, list):# 传入逻辑为[batch["rgb"], batch["ir"]]
                x = m(x[0], x[1]) # 实际训练/预测的时候 输入m.i = 0但是x为list
            else:
                x = m(x, x) # 构建网络的输入 m.i = 0但是x为单tensor

            y.append(x if m.i in self.save else None)  # save output 保存每一层的输出
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference."""
        LOGGER.warning(
            f"WARNING ⚠️ {self.__class__.__name__} does not support 'augment=True' prediction. "
            f"Reverting to single-scale prediction."
        )
        return self._predict_once(x)

    def _profile_one_layer(self, m, x, dt):
        """
        根据给定的输入，分析模型单个层的计算时间和 FLOP。将结果附加到
        提供的列表中。
        参数：
            m (nn.Module)：要分析的层。
            x (torch.Tensor)：层的输入数据。
            dt (list)：用于存储层计算时间的列表。
        返回：
            无
        """
        c = m==self.model[-1] and isinstance(x, list)  # is final layer list, copy input as inplace fix
        flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1e9 * 2 if thop else 0  # GFLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m==self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self, verbose=True):
        """
        Fuse the `Conv2d()` and `BatchNorm2d()` layers of the model into a single layer, in order to improve the
        computation efficiency.

        Returns:
            (nn.Module): The fused model is returned.
        """
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, "bn"):
                    if isinstance(m, Conv2):
                        m.fuse_convs()
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, ConvTranspose) and hasattr(m, "bn"):
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, RepConv):
                    m.fuse_convs()
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, RepVGGDW):
                    m.fuse()
                    m.forward = m.forward_fuse
            self.info(verbose=verbose)

        return self

    def is_fused(self, thresh=10):
        """
        Check if the model has less than a certain threshold of BatchNorm layers.

        Args:
            thresh (int, optional): The threshold number of BatchNorm layers. Default is 10.

        Returns:
            (bool): True if the number of BatchNorm layers in the model is less than the threshold, False otherwise.
        """
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        return sum(isinstance(v, bn) for v in self.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model

    def info(self, detailed=False, verbose=True, imgsz=640):
        """
        Prints model information.

        Args:
            detailed (bool): if True, prints out detailed information about the model. Defaults to False
            verbose (bool): if True, prints out the model information. Defaults to False
            imgsz (int): the size of the image that the model will be trained on. Defaults to 640
        """
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, fn):
        """
        Applies a function to all the tensors in the model that are not parameters or registered buffers.

        Args:
            fn (function): the function to apply to the model

        Returns:
            (BaseModel): An updated BaseModel object.
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def load(self, weights, verbose=True):
        """
        Load the weights into the model.
        给模型加载权重

        Args:
            weights (dict | torch.nn.Module): The pre-trained weights to be loaded.
            verbose (bool, optional): Whether to log the transfer progress. Defaults to True.
        """
        model = weights["model"] if isinstance(weights, dict) else weights  # torchvision models are not dicts
        csd = model.float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, self.state_dict())  # intersect
        self.load_state_dict(csd, strict=False)  # load
        if verbose:
            LOGGER.info(f"Transferred {len(csd)}/{len(self.model.state_dict())} items from pretrained weights")

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): batch 是一个字典，通常包含输入图像和相应的标签（如类别标签、边界框坐标等）。
                这个字典通常是在训练过程中从 DataLoader 中提取的批数据。
            preds (torch.Tensor | List[torch.Tensor]): 模型的预测结果，通常在模型的 forward 方法中生成。
                如果未提供，函数会通过调用模型的 forward 方法自动生成 preds。
        """
        # 如果当前对象没有定义 criterion（损失函数），则调用 init_criterion 方法初始化它。
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        # 如果是一个epoch开始训练，就会用forward生成图片的预测结果
        if "ir" in batch and "rgb" in batch:
            preds = self.forward([batch["rgb"]["img"], batch["ir"]["img"]]) if preds is None else preds
        else:
            preds = self.forward(batch["img"]) if preds is None else preds

        # 通过预测结果和batch原标签的区别计算损失函数
        # 进入ultralytics.utils.loss.v8DetectionLoss.__call__
        return self.criterion(preds, batch)

    def init_criterion(self):
        """Initialize the loss criterion for the BaseModel."""
        raise NotImplementedError("compute_loss() needs to be implemented by task heads")


class DetectionModel(BaseModel):
    """YOLOv8 detection model."""
    '''
        1. 预加载会把cfg权重覆盖了
    '''
    def __init__(self, cfg="yolov11n.yaml", ch=3, nc=None, verbose=True):  # model, input channels, number of classes
        """Initialize the YOLOv8 detection model with the given cfg配置文件 and parameters参数."""
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  #读取出 cfg 字典
        # yolov9的特殊处理
        if self.yaml["backbone"][0][2] == "Silence":
            LOGGER.warning(
                "WARNING ⚠️ YOLOv9 `Silence` module is deprecated in favor of nn.Identity. "
                "Please delete local *.pt file and re-download the latest model checkpoint."
            )
            self.yaml["backbone"][0][2] = "nn.Identity"

        # print(cfg, self.yaml.get("ch", ch))
        # Define model 这个ch是输入图片的ch数
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels

        # nc不匹配 会把yaml中的改为输入的nc
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value

        # model, savelist已经拼接完了（给你看）
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)
        # 输出的类别索引映射为可读的名称或标签。
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        # 这个标志通常用来指示是否在某些操作（如激活函数）中进行原地操作，以节省内存。
        self.inplace = self.yaml.get("inplace", True)
        # end2end属性通常用于指示模型是否是端到端的，即模型的输入到输出是否是完全自动化和连接的，没有需要手动干预的步骤。
        # 如果 self.model[-1] 有 end2end 属性，getattr 会返回该属性的值。
        # 如果 self.model[-1] 没有 end2end 属性，getattr 会返回默认值 False。
        self.end2end = getattr(self.model[-1], "end2end", False)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect
            s = 256  # 定义了一个基准值 s，通常用于计算步幅。
            m.inplace = self.inplace # 用于控制在某些操作中是否使用原地计算以节省内存。

            def _forward(x):
                """Performs a forward pass through the model, handling different Detect subclass types accordingly."""
                if self.end2end:
                    return self.forward(x)["one2many"]
                return self.forward(x)[0] if isinstance(m, (Segment, Pose, OBB)) else self.forward(x)

            # 通过对一个全零的张量（形状为 (1, ch, s, s)）进行前向传播，计算输出特征图的尺寸，并据此计算步幅。
            # 步幅是输入尺寸与输出特征图尺寸的比值。结果存储在 m.stride 和 self.stride 中。
            # x 返回每一层的输入  计算输入尺寸 s 和输出高度之间的比例。
            # 它们的比值通常表示输入和输出尺寸之间的缩放因子，或者可以理解为模型的步幅。
            # 这段代码的核心是通过伪输入（零张量）来计算模型在前向传播过程中每一层的空间缩放因子或步幅。
            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])  # forward
            # 上面这一句就是forward对网络进行初始化
            self.stride = m.stride

            # 专门用于初始化 Detect 类或其子类中的偏置（bias）。它特别关注检测任务中的偏置初始化，而不是整个模型的权重。
            m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        # Init weights, biases
        # 对整个模型的权重进行初始化。它遍历模型中的每个模块，并根据模块的类型进行特定的初始化操作。
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference and train outputs."""
        if getattr(self, "end2end", False) or self.__class__.__name__ != "DetectionModel":
            LOGGER.warning("WARNING ⚠️ Model does not support 'augment=True', reverting to single-scale prediction.")
            return self._predict_once(x)
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]  # forward
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, -1), None  # augmented inference, train

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """De-scale predictions following augmented inference (inverse operation)."""
        p[:, :4] /= scale  # de-scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # de-flip ud
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """Clip YOLO augmented inference tails."""
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return E2EDetectLoss(self) if getattr(self, "end2end", False) else v8DetectionLoss(self)


class OBBModel(DetectionModel):
    """YOLOv8 Oriented Bounding Box (OBB) model."""

    def __init__(self, cfg="yolov8n-obb.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOv8 OBB model with given config and parameters."""
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the model."""
        return v8OBBLoss(self)


class SegmentationModel(DetectionModel):
    """YOLOv8 segmentation model."""

    def __init__(self, cfg="yolov8n-seg.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOv8 segmentation model with given config and parameters."""
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the SegmentationModel."""
        return v8SegmentationLoss(self)


class PoseModel(DetectionModel):
    """YOLOv8 pose model."""

    def __init__(self, cfg="yolov8n-pose.yaml", ch=3, nc=None, data_kpt_shape=(None, None), verbose=True):
        """Initialize YOLOv8 Pose model."""
        if not isinstance(cfg, dict):
            cfg = yaml_model_load(cfg)  # load model YAML
        if any(data_kpt_shape) and list(data_kpt_shape) != list(cfg["kpt_shape"]):
            LOGGER.info(f"Overriding model.yaml kpt_shape={cfg['kpt_shape']} with kpt_shape={data_kpt_shape}")
            cfg["kpt_shape"] = data_kpt_shape
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the PoseModel."""
        return v8PoseLoss(self)


class ClassificationModel(BaseModel):
    """YOLOv8 classification model."""

    def __init__(self, cfg="yolov8n-cls.yaml", ch=3, nc=None, verbose=True):
        """Init ClassificationModel with YAML, channels, number of classes, verbose flag."""
        super().__init__()
        self._from_yaml(cfg, ch, nc, verbose)

    def _from_yaml(self, cfg, ch, nc, verbose):
        """Set YOLOv8 model configurations and define the model architecture."""
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        elif not nc and not self.yaml.get("nc", None):
            raise ValueError("nc not specified. Must specify nc in model.yaml or function arguments.")
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.stride = torch.Tensor([1])  # no stride constraints
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.info()

    @staticmethod
    def reshape_outputs(model, nc):
        """Update a TorchVision classification model to class count 'n' if required."""
        name, m = list((model.model if hasattr(model, "model") else model).named_children())[-1]  # last module
        if isinstance(m, Classify):  # YOLO Classify() head
            if m.linear.out_features != nc:
                m.linear = nn.Linear(m.linear.in_features, nc)
        elif isinstance(m, nn.Linear):  # ResNet, EfficientNet
            if m.out_features != nc:
                setattr(model, name, nn.Linear(m.in_features, nc))
        elif isinstance(m, nn.Sequential):
            types = [type(x) for x in m]
            if nn.Linear in types:
                i = len(types) - 1 - types[::-1].index(nn.Linear)  # last nn.Linear index
                if m[i].out_features != nc:
                    m[i] = nn.Linear(m[i].in_features, nc)
            elif nn.Conv2d in types:
                i = len(types) - 1 - types[::-1].index(nn.Conv2d)  # last nn.Conv2d index
                if m[i].out_channels != nc:
                    m[i] = nn.Conv2d(m[i].in_channels, nc, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None)

    def init_criterion(self):
        """Initialize the loss criterion for the ClassificationModel."""
        return v8ClassificationLoss()


class RTDETRDetectionModel(DetectionModel):
    """
    RTDETR (Real-time DEtection and Tracking using Transformers) Detection Model class.

    This class is responsible for constructing the RTDETR architecture, defining loss functions, and facilitating both
    the training and inference processes. RTDETR is an object detection and tracking model that extends from the
    DetectionModel base class.

    Attributes:
        cfg (str): The configuration file path or preset string. Default is 'rtdetr-l.yaml'.
        ch (int): Number of input channels. Default is 3 (RGB).
        nc (int, optional): Number of classes for object detection. Default is None.
        verbose (bool): Specifies if summary statistics are shown during initialization. Default is True.

    Methods:
        init_criterion: Initializes the criterion used for loss calculation.
        loss: Computes and returns the loss during training.
        predict: Performs a forward pass through the network and returns the output.
    """

    def __init__(self, cfg="rtdetr-l.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize the RTDETRDetectionModel.

        Args:
            cfg (str): Configuration file name or path.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes. Defaults to None.
            verbose (bool, optional): Print additional information during initialization. Defaults to True.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the RTDETRDetectionModel."""
        from ultralytics.models.utils.loss import RTDETRDetectionLoss

        return RTDETRDetectionLoss(nc=self.nc, use_vfl=True)

    def loss(self, batch, preds=None):
        """
        Compute the loss for the given batch of data.

        Args:
            batch (dict): Dictionary containing image and label data.
            preds (torch.Tensor, optional): Precomputed model predictions. Defaults to None.

        Returns:
            (tuple): A tuple containing the total loss and main three losses in a tensor.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        img = batch["img"]
        # NOTE: preprocess gt_bbox and gt_labels to list.
        bs = len(img)
        batch_idx = batch["batch_idx"]
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
        targets = {
            "cls": batch["cls"].to(img.device, dtype=torch.long).view(-1),
            "bboxes": batch["bboxes"].to(device=img.device),
            "batch_idx": batch_idx.to(img.device, dtype=torch.long).view(-1),
            "gt_groups": gt_groups,
        }

        preds = self.predict(img, batch=targets) if preds is None else preds
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]
        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)

        dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])  # (7, bs, 300, 4)
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])

        loss = self.criterion(
            (dec_bboxes, dec_scores), targets, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta
        )
        # NOTE: There are like 12 losses in RTDETR, backward with all losses but only show the main three losses.
        return sum(loss.values()), torch.as_tensor(
            [loss[k].detach() for k in ["loss_giou", "loss_class", "loss_bbox"]], device=img.device
        )

    def predict(self, x, profile=False, visualize=False, batch=None, augment=False, embed=None):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool, optional): If True, profile the computation time for each layer. Defaults to False.
            visualize (bool, optional): If True, save feature maps for visualization. Defaults to False.
            batch (dict, optional): Ground truth data for evaluation. Defaults to None.
            augment (bool, optional): If True, perform data augmentation during inference. Defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        y, dt, embeddings = [], [], []  # outputs
        for m in self.model[:-1]:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        head = self.model[-1]
        x = head([y[j] for j in head.f], batch)  # head inference
        return x


class WorldModel(DetectionModel):
    """YOLOv8 World Model."""

    def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOv8 world model with given config and parameters."""
        self.txt_feats = torch.randn(1, nc or 80, 512)  # features placeholder
        self.clip_model = None  # CLIP model placeholder
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def set_classes(self, text, batch=80, cache_clip_model=True):
        """Set classes in advance so that model could do offline-inference without clip model."""
        try:
            import clip
        except ImportError:
            check_requirements("git+https://github.com/ultralytics/CLIP.git")
            import clip

        if (
            not getattr(self, "clip_model", None) and cache_clip_model
        ):  # for backwards compatibility of models lacking clip_model attribute
            self.clip_model = clip.load("ViT-B/32")[0]
        model = self.clip_model if cache_clip_model else clip.load("ViT-B/32")[0]
        device = next(model.parameters()).device
        text_token = clip.tokenize(text).to(device)
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        self.txt_feats = txt_feats.reshape(-1, len(text), txt_feats.shape[-1])
        self.model[-1].nc = len(text)

    def predict(self, x, profile=False, visualize=False, txt_feats=None, augment=False, embed=None):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool, optional): If True, profile the computation time for each layer. Defaults to False.
            visualize (bool, optional): If True, save feature maps for visualization. Defaults to False.
            txt_feats (torch.Tensor): The text features, use it if it's given. Defaults to None.
            augment (bool, optional): If True, perform data augmentation during inference. Defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        txt_feats = (self.txt_feats if txt_feats is None else txt_feats).to(device=x.device, dtype=x.dtype)
        if len(txt_feats) != len(x):
            txt_feats = txt_feats.repeat(len(x), 1, 1)
        ori_txt_feats = txt_feats.clone()
        y, dt, embeddings = [], [], []  # outputs
        for m in self.model:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if isinstance(m, C2fAttn):
                x = m(x, txt_feats)
            elif isinstance(m, WorldDetect):
                x = m(x, ori_txt_feats)
            elif isinstance(m, ImagePoolingAttn):
                txt_feats = m(x, txt_feats)
            else:
                x = m(x)  # run

            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | List[torch.Tensor]): Predictions.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"], txt_feats=batch["txt_feats"])
        return self.criterion(preds, batch)


class Ensemble(nn.ModuleList):
    """Ensemble of models."""

    def __init__(self):
        """Initialize an ensemble of models."""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Function generates the YOLO network's final layer."""
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 2)  # nms ensemble, y shape(B, HW, C)
        return y, None  # inference, train output


# Functions ------------------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def temporary_modules(modules=None, attributes=None):
    """
    Context manager for temporarily adding or modifying modules in Python's module cache (`sys.modules`).

    This function can be used to change the module paths during runtime. It's useful when refactoring code,
    where you've moved a module from one location to another, but you still want to support the old import
    paths for backwards compatibility.

    Args:
        modules (dict, optional): A dictionary mapping old module paths to new module paths.
        attributes (dict, optional): A dictionary mapping old module attributes to new module attributes.

    Example:
        ```python
        with temporary_modules({"old.module": "new.module"}, {"old.module.attribute": "new.module.attribute"}):
            import old.module  # this will now import new.module
            from old.module import attribute  # this will now import new.module.attribute
        ```

    Note:
        The changes are only in effect inside the context manager and are undone once the context manager exits.
        Be aware that directly manipulating `sys.modules` can lead to unpredictable results, especially in larger
        applications or libraries. Use this function with caution.
    """
    if modules is None:
        modules = {}
    if attributes is None:
        attributes = {}
    import sys
    from importlib import import_module

    try:
        # Set attributes in sys.modules under their old name
        for old, new in attributes.items():
            old_module, old_attr = old.rsplit(".", 1)
            new_module, new_attr = new.rsplit(".", 1)
            setattr(import_module(old_module), old_attr, getattr(import_module(new_module), new_attr))

        # Set modules in sys.modules under their old name
        for old, new in modules.items():
            sys.modules[old] = import_module(new)

        yield
    finally:
        # Remove the temporary module paths
        for old in modules:
            if old in sys.modules:
                del sys.modules[old]


class SafeClass:
    """A placeholder class to replace unknown classes during unpickling."""

    def __init__(self, *args, **kwargs):
        """Initialize SafeClass instance, ignoring all arguments."""
        pass

    def __call__(self, *args, **kwargs):
        """Run SafeClass instance, ignoring all arguments."""
        pass


class SafeUnpickler(pickle.Unpickler):
    """Custom Unpickler that replaces unknown classes with SafeClass."""

    def find_class(self, module, name):
        """Attempt to find a class, returning SafeClass if not among safe modules."""
        safe_modules = (
            "torch",
            "collections",
            "collections.abc",
            "builtins",
            "math",
            "numpy",
            # Add other modules considered safe
        )
        if module in safe_modules:
            return super().find_class(module, name)
        else:
            return SafeClass


def torch_safe_load(weight, safe_only=False):
    """
    Attempts to load a PyTorch model with the torch.load() function. If a ModuleNotFoundError is raised, it catches the
    error, logs a warning message, and attempts to install the missing module via the check_requirements() function.
    After installation, the function again attempts to load the model using torch.load().

    Args:
        weight (str): The file path of the PyTorch model.
        safe_only (bool): If True, replace unknown classes with SafeClass during loading.

    Example:
    ```python
    from ultralytics.nn.tasks import torch_safe_load

    ckpt, file = torch_safe_load("path/to/best.pt", safe_only=True)
    ```

    Returns:
        ckpt (dict): The loaded model checkpoint.
        file (str): The loaded filename
    """
    from ultralytics.utils.downloads import attempt_download_asset

    check_suffix(file=weight, suffix=".pt")
    file = attempt_download_asset(weight)  # search online if missing locally
    try:
        with temporary_modules(
            modules={
                "ultralytics.yolo.utils": "ultralytics.utils",
                "ultralytics.yolo.v8": "ultralytics.models.yolo",
                "ultralytics.yolo.data": "ultralytics.data",
            },
            attributes={
                "ultralytics.nn.modules.block.Silence": "torch.nn.Identity",  # YOLOv9e
                "ultralytics.nn.tasks.YOLOv10DetectionModel": "ultralytics.nn.tasks.DetectionModel",  # YOLOv10
                "ultralytics.utils.loss.v10DetectLoss": "ultralytics.utils.loss.E2EDetectLoss",  # YOLOv10
            },
        ):
            if safe_only:
                # Load via custom pickle module
                safe_pickle = types.ModuleType("safe_pickle")
                safe_pickle.Unpickler = SafeUnpickler
                safe_pickle.load = lambda file_obj: SafeUnpickler(file_obj).load()
                with open(file, "rb") as f:
                    ckpt = torch.load(f, pickle_module=safe_pickle)
            else:
                ckpt = torch.load(file, map_location="cpu")

    except ModuleNotFoundError as e:  # e.name is missing module name
        if e.name == "models":
            raise TypeError(
                emojis(
                    f"ERROR ❌️ {weight} appears to be an Ultralytics YOLOv5 model originally trained "
                    f"with https://github.com/ultralytics/yolov5.\nThis model is NOT forwards compatible with "
                    f"YOLOv8 at https://github.com/ultralytics/ultralytics."
                    f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
                    f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolov8n.pt'"
                )
            ) from e
        LOGGER.warning(
            f"WARNING ⚠️ {weight} appears to require '{e.name}', which is not in Ultralytics requirements."
            f"\nAutoInstall will run now for '{e.name}' but this feature will be removed in the future."
            f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
            f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolov8n.pt'"
        )
        check_requirements(e.name)  # install missing module
        ckpt = torch.load(file, map_location="cpu")

    if not isinstance(ckpt, dict):
        # File is likely a YOLO instance saved with i.e. torch.save(model, "saved_model.pt")
        LOGGER.warning(
            f"WARNING ⚠️ The file '{weight}' appears to be improperly saved or formatted. "
            f"For optimal results, use model.save('filename.pt') to correctly save YOLO models."
        )
        ckpt = {"model": ckpt.model}

    return ckpt, file


def attempt_load_weights(weights, device=None, inplace=True, fuse=False):
    """Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a."""
    ensemble = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt, w = torch_safe_load(w)  # load ckpt
        args = {**DEFAULT_CFG_DICT, **ckpt["train_args"]} if "train_args" in ckpt else None  # combined args
        model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

        # Model compatibility updates
        model.args = args  # attach args to model
        model.pt_path = w  # attach *.pt file path to model
        model.task = guess_model_task(model)
        if not hasattr(model, "stride"):
            model.stride = torch.tensor([32.0])

        # Append
        ensemble.append(model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval())  # model in eval mode

    # Module updates
    for m in ensemble.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(ensemble) == 1:
        return ensemble[-1]

    # Return ensemble
    LOGGER.info(f"Ensemble created with {weights}\n")
    for k in "names", "nc", "yaml":
        setattr(ensemble, k, getattr(ensemble[0], k))
    ensemble.stride = ensemble[int(torch.argmax(torch.tensor([m.stride.max() for m in ensemble])))].stride
    assert all(ensemble[0].nc == m.nc for m in ensemble), f"Models differ in class counts {[m.nc for m in ensemble]}"
    return ensemble


def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False):
    """Loads a single model weights."""
    ckpt, weight = torch_safe_load(weight)  # load ckpt
    args = {**DEFAULT_CFG_DICT, **(ckpt.get("train_args", {}))}  # combine model and default args, preferring model args
    model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

    # Model compatibility updates
    model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # attach args to model
    model.pt_path = weight  # attach *.pt file path to model
    model.task = guess_model_task(model)
    if not hasattr(model, "stride"):
        model.stride = torch.tensor([32.0])

    model = model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval()  # model in eval mode

    # Module updates
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model and ckpt
    return model, ckpt


def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
    import ast

    # Args
    legacy = True  # 为了v3/5/8/9等等反向传播适配性
    max_channels = float("inf")
    """
        nc: 5 # number of classes
        scales: # model compound scaling constants, i.e. 'model=yolo11n.yaml' will call yolo11.yaml with scale 'n'
          # [depth, width, max_channels]
          n: [0.50, 0.25, 1024] # summary: 319 layers, 2624080 parameters, 2624064 gradients, 6.6 GFLOPs
          s: [0.50, 0.50, 1024] # summary: 319 layers, 9458752 parameters, 9458736 gradients, 21.7 GFLOPs
          m: [0.50, 1.00, 512] # summary: 409 layers, 20114688 parameters, 20114672 gradients, 68.5 GFLOPs
          l: [1.00, 1.00, 512] # summary: 631 layers, 25372160 parameters, 25372144 gradients, 87.6 GFLOPs
          x: [1.00, 1.50, 512] # summary: 631 layers, 56966176 parameters, 56966160 gradients, 196.0 GFLOPs
        """
    # 获取yaml文件的nc和scales，激活函数没定义默认为silu
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    # 继续读取scales文件
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0] # 默认取出yaml文件scales的的第一行
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        #  n: [0.50, 0.25, 1024]网络深度宽度，最大通道数
        depth, width, max_channels = scales[scale]
    # 定义一个激活函数，默认为silu
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch] # channel定义为列表
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, channel输出
    '''
        对 args 参数进行调整，确保每一层模块在解析时得到正确的输入参数。
        这是因为在构建 YOLO 模型时，不同的层可能会需要不同的输入参数。
        所以根据实际层的输入参数进行的一个调整
    '''
    # 把yaml中的backbone和head网络层拼接到一起进行访问
    # [from, repeats, module, args]
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        # 根据module字符串获取模块,如果yaml中的网络层的字符串中有nn.什么就把他后面部分取出来和torch.nn拼接成为网络层名
        m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]
        # 如果取出来的args中有字符串如  - [-1, 1, efficientnet_v2, [efficientnet_v2_s, 0.25,0.5]] # 0-4 P1/2，就进行处理成为网络层名
        for j, a in enumerate(args):
            if isinstance(a, str):
                try:
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
                except ValueError:
                    pass
        # n*depth是表示这个模块的重复次数，至少一个，这个求的 n 是每个模块重复次数，round取整
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in {
            Classify,
            Conv,
            ConvTranspose,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            C2fPSA,
            C2PSA,
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            RepNCSPELAN4,
            ELAN1,
            ADown,
            AConv,
            SPPELAN,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            RepC3,
            PSA,
            SCDown,
            C2fCIB,
            # new
            AKConv, C3k2_AKConv, ACDF, eca_layer
        }:
            # c1是输入通道深度（数），c2是输出通道深度
            # ch[f]中f一般-1，取出的是上一层网络最后的那个输出通道数，args[0]是模型输出的通道数
            c1, c2 = ch[f], args[0]
            # 确保输出通道数 c2 在不超过最大通道数的前提下 网络中的通道数调整为8的整数倍
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            #如果模块是C2fAttn，那么它会根据模型缩放因子调整embed channels和num heads的大小。这个模块主要用于注意力机制
            if m is C2fAttn:
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)  # embed channels
                args[2] = int(
                    max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2]
                )  # num heads

            #这个是普遍的输入到网络中的参数，后面有特殊进行一个修改
            #将c1和c2（输入输出通道数）放入args列表中，并保持其他参数不变。
            if m in{ACDF}:
                c2 = 3
            args = [c1, c2, *args[1:]]
            if m in {eca_layer}:
                ch[f] = ch[f] * 2
                c1 = ch[f] # ACDF后要把out改为真的out，输入的out是真的out的一半
                c2 = c1
                args = [c1]

            # 如果模块是这里面的部分    它会在参数列表中插入重复的次数n
            if m in {
                BottleneckCSP,
                C1,
                C2,
                C2f,
                C3k2,
                C2fAttn,
                C3,
                C3TR,
                C3Ghost,
                C3x,
                RepC3,
                C2fPSA,
                C2fCIB,
                C2PSA,
                C3k2_AKConv
            }:
                args.insert(2, n)  # 将变量n插入到args列表的索引位置2
                n = 1
            # 如果用了C3k2就不用 legacy
            if m is C3k2:  # for M/L/X sizes
                legacy = False
                if scale in "mlx":
                    args[3] = True
        # 如果模块是AIFI，那么它会调整参数列表，把最近一层输出加入。
        elif m is AIFI:
            args = [ch[f], *args]
        # 如果模块是HGStem，HGBlock，那么它会调整参数列表
        # 并且如果模块是HGBlock，还会在参数列表中插入重复的次数n。
        elif m in {HGStem, HGBlock}:
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # number of repeats
                n = 1
        # 如果模块是ResNetLayer，那么它会调整输出通道数。
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        # 如果模块是nn.BatchNorm2d，那么它会调整参数列表。
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        # 如果模块是Concat，那么它会计算输出通道数。
        elif m is Concat:
            if -2 in f:
                # print('yes')
                # c2 = ch[-1] + 1
                c2 = ch[-1] * 2
                c1 = ch[-1]
            else:
                c2 = sum(ch[x] for x in f) # f中会有要连接的层和-1,ch取出对应通道数进行一个通道融合
        # 如果模块是以下目标 即最后一层任务，那么它会调整参数列表
        elif m in {Detect, WorldDetect, Segment, Pose, OBB, ImagePoolingAttn, v10Detect}:
            # 列表通常包含了需要融合的层的通道信息。
            # 确保每个模块可以访问需要融合的多个层的通道数（如特征图的维度），以便在之后的特征融合中可以正确处理这些通道信息。
            args.append([ch[x] for x in f]) # 把16 19 22层的输出通道数加入args
            if m is Segment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            if m in {Detect, Segment, Pose, OBB}:
                m.legacy = legacy
        # 如果模块是RTDETRDecoder，那么它会在参数列表中插入通道数。
        elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1
            args.insert(1, [ch[x] for x in f])
        # 如果模块是CBLinear，那么它会调整参数列表。
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        # 如果模块是CBFuse，那么它会调整输出通道数。
        elif m is CBFuse:
            c2 = ch[f[-1]]
        else:
            c2 = ch[f]
        # args 输出的就是传入给网络层的参数
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module #网络实例化
        t = str(m)[8:-2].replace("__main__.", "")  # module type #t 代表的是当前模块 m 的类名Conv，去除了不必要的部分。
        m_.np = sum(x.numel() for x in m_.parameters())  # number params # 网络参数计算
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type # 信息补充
        if verbose:
            LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}")  # print
        # 把f中的-1过滤了，留下拼接的层索引
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1 and x != -2)

        layers.append(m_) # layers中加入一个层
        # 确保通道的记录从头开始。这样做有助于保证层与层之间的输出通道信息准确地传递和更新，避免混乱。
        if i == 0:
            ch = []
        ch.append(c2) # 将当前网络最终输出加入到ch中
    return nn.Sequential(*layers), sorted(save)


def yaml_model_load(path):
    """Load a YOLOv8 model from a YAML file."""
    path = Path(path)
    if path.stem in (f"yolov{d}{x}6" for x in "nsmlx" for d in (5, 8)):
        new_stem = re.sub(r"(\d+)([nslmx])6(.+)?$", r"\1\2-p6\3", path.stem)
        LOGGER.warning(f"WARNING ⚠️ Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}.")
        path = path.with_name(new_stem + path.suffix)

    unified_path = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", str(path))  # i.e. yolov8x.yaml -> yolov8.yaml
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = yaml_load(yaml_file)  # model dict
    d["scale"] = guess_model_scale(path)
    d["yaml_file"] = str(path)
    return d


def guess_model_scale(model_path):
    """
    Takes a path to a YOLO model's YAML file as input and extracts the size character of the model's scale. The function
    uses regular expression matching to find the pattern of the model scale in the YAML file name, which is denoted by
    n, s, m, l, or x. The function returns the size character of the model scale as a string.

    Args:
        model_path (str | Path): The path to the YOLO model's YAML file.

    Returns:
        (str): The size character of the model's scale, which can be n, s, m, l, or x.
    """
    try:
        return re.search(r"yolo[v]?\d+([nslmx])", Path(model_path).stem).group(1)  # noqa, returns n, s, m, l, or x
    except AttributeError:
        return ""


def guess_model_task(model):
    """
    Guess the task of a PyTorch model from its architecture or configuration.

    Args:
        model (nn.Module | dict): PyTorch model or model configuration in YAML format.

    Returns:
        (str): Task of the model ('detect', 'segment', 'classify', 'pose').

    Raises:
        SyntaxError: If the task of the model could not be determined.
    """

    def cfg2task(cfg):
        """Guess from YAML dictionary."""
        m = cfg["head"][-1][-2].lower()  # output module name
        if m in {"classify", "classifier", "cls", "fc"}:
            return "classify"
        if "detect" in m:
            return "detect"
        if m == "segment":
            return "segment"
        if m == "pose":
            return "pose"
        if m == "obb":
            return "obb"

    # Guess from model cfg
    if isinstance(model, dict):
        try:
            return cfg2task(model)
        except Exception:
            pass

    # Guess from PyTorch model
    if isinstance(model, nn.Module):  # PyTorch model
        for x in "model.args", "model.model.args", "model.model.model.args":
            try:
                return eval(x)["task"]
            except Exception:
                pass
        for x in "model.yaml", "model.model.yaml", "model.model.model.yaml":
            try:
                return cfg2task(eval(x))
            except Exception:
                pass

        for m in model.modules():
            if isinstance(m, Segment):
                return "segment"
            elif isinstance(m, Classify):
                return "classify"
            elif isinstance(m, Pose):
                return "pose"
            elif isinstance(m, OBB):
                return "obb"
            elif isinstance(m, (Detect, WorldDetect, v10Detect)):
                return "detect"

    # Guess from model filename
    if isinstance(model, (str, Path)):
        model = Path(model)
        if "-seg" in model.stem or "segment" in model.parts:
            return "segment"
        elif "-cls" in model.stem or "classify" in model.parts:
            return "classify"
        elif "-pose" in model.stem or "pose" in model.parts:
            return "pose"
        elif "-obb" in model.stem or "obb" in model.parts:
            return "obb"
        elif "detect" in model.parts:
            return "detect"

    # Unable to determine task from model
    LOGGER.warning(
        "WARNING ⚠️ Unable to automatically guess model task, assuming 'task=detect'. "
        "Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify','pose' or 'obb'."
    )
    return "detect"  # assume detect
