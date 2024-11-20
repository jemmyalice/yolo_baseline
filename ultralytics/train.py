import warnings
from idlelib.configdialog import tracers

from PIL.ImageFont import truetype

from ultralytics import YOLO
warnings.filterwarnings('ignore')

if __name__=='__main__':
    # model = YOLO(r'F:\ultralytics-main\ultralytics\cfg\models\11\yolo11s.yaml')
    model = YOLO(r'F:\ultralytics-main\ultralytics\cfg\models\11\infyolo11s.yaml')
    model.train(data=r'F:\ultralytics-main\data\llvip\data_infusion.yaml',
    # model.train(data=r'F:\ultralytics-main\data\llvip\data.yaml',
        save_period = 20,
        epochs=1,
        single_cls=False,  # 是否是单类别检测
        patience = 20,
        batch=2,
        workers=0,
        cos_lr = True,
        resume = True,
        optimizer='SGD',  # using SGD 优化器 默认为auto建议大家使用固定的.
        fraction =0.01,
        exist_ok = True,
        multi_scale = True,
        device='cpu',
        amp=True,  # 如果出现训练损失为Nan可以关闭amp
        project='runs/train',
        name='exp',
    )
