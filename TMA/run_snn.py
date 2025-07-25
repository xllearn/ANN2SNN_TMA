import torch
import torch.nn as nn
import argparse
import os
import random
import numpy as np
import copy
from torch.autograd import Function

# 从您原有的项目中导入必要的模块
from Models.ResNet import *
from Models.VGG import *
from dataprocess import PreProcess_Cifar10, PreProcess_Cifar100, PreProcess_ImageNet
from NetworkFunction import eval_snn, mp_test
# 注意：我们只从utils导入最原始、最安全的函数
from utils import replace_maxpool2d_by_avgpool2d, replace_activation_by_floor, replace_activation_by_MPLayer, replace_MPLayer_by_neuron, isActivation, QCFS, IFNeuron

# =========================================================================================
#  以下是所有新功能代码，完全封装在此脚本内部，与您的主项目隔离
# =========================================================================================

class surrogate_function(Function):
    """
    用于GN神经元的自定义反向传播（代理梯度）函数
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        sg = 1 / (1 + torch.exp(-x))
        return grad_output * sg * (1 - sg)

class GN(nn.Module):
    """
    一个完全独立、安全的组神经元 (Group Neuron) 实现
    """
    def __init__(self, m: int = 4, v_threshold: float = 1.0):
        super().__init__()
        self.m = m
        self.v_th_internal = v_threshold / self.m
        self.surrogate = surrogate_function.apply

        # 使用 register_buffer 安全地注册状态，这些状态不是模型参数
        self.register_buffer('v', torch.tensor(0.0))
        self.register_buffer('bias', torch.arange(1, m + 1, dtype=torch.float32) * self.v_th_internal)
        self.initialized = False

    def forward(self, x: torch.Tensor):
        if not self.initialized:
            # “懒加载”：在第一次前向传播时，根据输入x的形状，初始化状态张量的正确尺寸
            self.v = torch.full_like(x, self.v_th_internal * 0.5)
            v_shape = self.v.shape
            self.v = self.v.unsqueeze(0).repeat(self.m, *([1] * len(v_shape)))
            self.bias = self.bias.view(self.m, *([1] * len(v_shape)))
            self.initialized = True
        
        self.v += x
        spike = self.surrogate(self.v - self.bias)
        spike_sum = torch.sum(spike, dim=0)
        self.v -= spike_sum * self.v_th_internal
        return spike_sum * self.v_th_internal

    def reset(self):
        """
        重置神经元状态，以便在处理新序列时使用
        """
        self.initialized = False
        self.register_buffer('v', torch.tensor(0.0))

def get_module_by_name(parent, name):
    """
    一个辅助函数，用于按名称字符串获取子模块
    """
    name_list = name.split(".")
    for item in name_list[:-1]:
        if hasattr(parent, item):
            parent = getattr(parent, item)
        else:
            return None, None
    if hasattr(parent, name_list[-1]):
        child = getattr(parent, name_list[-1])
        return parent, child
    else:
        return None, None

def replace_qcfs_with_sn(model: nn.Module, members:int, sn_type:str):
    """
    一个稳健的替换函数，它遍历模型，将QCFS层替换为指定的脉冲神经元 (GN或IF)
    """
    module_name_list = []
    for name, module in model.named_modules():
        if isinstance(module, QCFS):
            module_name_list.append(name)
    
    for name in module_name_list:
        parent, child = get_module_by_name(model, name)
        if parent is None or child is None:
            continue

        if sn_type == 'gn':
            new_child = GN(m=members, v_threshold=child.up.item())
        elif sn_type == 'if':
            # IFNeuron 是从您原始的 modules.py 中安全导入的
            new_child = IFNeuron(scale=child.up.item())
        else:
            raise ValueError(f"不支持的神经元类型: {sn_type}。请选择 'gn' 或 'if'。")
        
        setattr(parent, name.split('.')[-1], new_child)
    return model

# =========================================================================================
#  主程序逻辑
# =========================================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SNN Conversion and Evaluation Script')
    
    # 添加运行此脚本所需的核心参数
    parser.add_argument('--dataset', type=str, required=True, help='数据集名称 (例如: CIFAR10, CIFAR100)')
    parser.add_argument('--net_arch', type=str, required=True, help='网络架构 (例如: vgg16, resnet20)')
    parser.add_argument('--load_model_name', type=str, required=True, help='已训练好的ANN模型权重的路径 (不带.pth后缀)')
    parser.add_argument('--sn_type', type=str, default='if', help="要转换的脉冲神经元类型: 'if' 或 'gn'")
    parser.add_argument('--tau', type=int, default=4, help='当sn_type为gn时，组神经元的成员数量')
    parser.add_argument('--sim_len', type=int, default=32, help='SNN的仿真总时长')
    parser.add_argument('--batchsize', type=int, default=128, help='测试时使用的批量大小')
    parser.add_argument('--device', type=str, default='cuda:0', help='运行设备')
    parser.add_argument('--datadir', type=str, default='datasets', help='数据集所在的目录')
    parser.add_argument('--L', type=int, default=4, help='QCFS的量化级别 (需要和训练时保持一致)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    args = parser.parse_args()

    # 设置随机种子以保证可复现性
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("=============================================")
    print("SNN 转换与评估脚本")
    print(f"  模型: {args.net_arch}")
    print(f"  数据集: {args.dataset}")
    print(f"  加载权重: {args.load_model_name}.pth")
    print(f"  转换为: {args.sn_type.upper()} (tau={args.tau} if GN)")
    print("=============================================")

    # 1. 设置模型和数据集
    if args.dataset == 'CIFAR10':
        cls = 10
        cap_dataset = 10000
        _, test_loader = PreProcess_Cifar10(args.datadir, args.batchsize)
    elif args.dataset == 'CIFAR100':
        cls = 100
        cap_dataset = 10000
        _, test_loader = PreProcess_Cifar100(args.datadir, args.batchsize)
    else:
        raise NotImplementedError(f"不支持的数据集: {args.dataset}")

    if args.net_arch == 'vgg16':
        model = vgg16(num_classes=cls)
    elif args.net_arch == 'resnet20':
        model = resnet20(num_classes=cls)
    elif args.net_arch == 'resnet18':
        model = resnet18(num_classes=cls)    
    # 可根据需要添加更多模型
    else:
        raise NotImplementedError(f"不支持的网络架构: {args.net_arch}")

    # 2. 构建与预训练ANN相同的模型结构
    print("\n[步骤 1] 正在构建ANN模型结构...")
    model = replace_maxpool2d_by_avgpool2d(model)
    model = replace_activation_by_floor(model, args.L)
    model.to(args.device)

    # 3. 加载预训练好的ANN权重
    print(f"[步骤 2] 正在加载预训练权重: {args.load_model_name}.pth...")
    model.load_state_dict(torch.load(args.load_model_name + '.pth', map_location=args.device))

    # 4. 执行核心转换：将QCFS层替换为SNN神经元
    print(f"[步骤 3] 正在将QCFS层转换为 {args.sn_type.upper()} 神经元...")
    model = replace_qcfs_with_sn(model, members=args.tau, sn_type=args.sn_type)
    print("转换完成！")
    print(model)

    # 5. 评估转换后的SNN模型的性能
    print("\n[步骤 4] 正在评估SNN模型性能...")
    accuracies = eval_snn(test_loader, model, sim_len=args.sim_len, device=args.device)

    # 6. 打印结果
    print("\n================= SNN 性能评估结果 =================\n")
    for t in range(args.sim_len):
        acc = accuracies[t] / cap_dataset
        print(f"  时间步 (Timestep) {t + 1}: 精度 (Accuracy) = {acc:.4f}")
    
    print("\n=======================================================")
    print("评估结束。")