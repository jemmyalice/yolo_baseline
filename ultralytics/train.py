import warnings
from idlelib.configdialog import tracers
from PIL.ImageFont import truetype
from ultralytics import YOLO
warnings.filterwarnings('ignore')

if __name__=='__main__':

    # model = YOLO(r'F:\ultralytics-main\yolo11s.pt')
    model = YOLO(r'F:\ultralytics-main\ultralytics\cfg\models\11\infyolo11n.yaml')
    model.train(data=r'F:\ultralytics-main\data\llvip\data_infusion.yaml',
    # model.train(data=r'F:\ultralytics-main\data\llvip\data.yaml',
        lr0=0.0001,  # Learning rate
        plots = True,
        imgsz=640,  # Image size
        epochs=2,
        device='cpu',
        patience = 20, # 20轮性能没改善停止
        batch=2,
        single_cls=False,  # 是否是单类别检测
        workers=0, # 设置0让他自己判断
        # cos_lr = True, # 学习率以cos曲线变化，论文中大多没有使用它
        # resume = True, # 从最后一个检查点恢复训练，具体不清楚
        # optimizer='SGD',  # using SGD 优化器 默认为auto建议大家使用固定的.
        optimizer='Adamw',  # using SGD 优化器 默认为auto建议大家使用固定的.
        fraction =0.001, # 在跑通前用于自己测试
        exist_ok = True, # 在跑通前用于自己测试
        multi_scale = False, # 用于增加泛化性，但是会增加训练时常，要注意
        amp=True,  # 如果出现训练损失为Nan可以关闭amp
        save_period=20,  # 20轮保存一个pt，方便下次继续训练
        project='runs/train',
        name='exp',
    )

    # 使用验证集进行评估
    # model = YOLO(r'F:\downloads\chorme\best.pt')
    # results = model.val(
    #     data='/kaggle/input/llvip-converted/data_fusion.yaml',  # 配置文件路径，包含数据集路径和类别信息
    #     imgsz=640,  # 输入图像的大小，和训练时的一致
    #     batch=8,  # 每次评估时的批次大小
    #     device='0',  # 使用的设备，如果有 GPU 可设置为 'cuda'
    #     conf=0.50,  # 置信度阈值，低于该阈值的预测结果会被丢弃
    #     save=True,  # 是否保存预测结果（保存到 runs 文件夹）
    #     save_txt=True,  # 是否保存检测框坐标的文本文件（同一文件夹下会生成 `.txt` 文件）
    #     visualize=True,  # 是否可视化评估过程，显示图像
    #     line_width=3  # 画框时的线宽
    # )