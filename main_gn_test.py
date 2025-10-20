import argparse
import os
import torch
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import random
import numpy as np
from tqdm import tqdm
import copy

# 从您原有的项目中导入必要的模块
from Models import modelpool
from Preprocess import datapool
from Models.layer import IF  # 从您项目中导入IF神经元，它将被替换

# =========================================================================================
#  以下是所有新功能代码，完全封装在此脚本内部
# =========================================================================================

class surrogate_function(torch.autograd.Function):
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
        # 使用 Sigmoid 函数作为平滑的代理梯度
        sg = torch.sigmoid(x)
        return grad_output * sg * (1 - sg)

class GN(nn.Module):
    """
    一个完全独立、安全的组神经元 (Group Neuron) 实现
    Args:
        m (int): 组中神经元的数量
        v_threshold (float): ANN激活函数对应的原始阈值
    """
    def __init__(self, m: int = 4, v_threshold: float = 1.0):
        super().__init__()
        self.m = m
        self.v_th_internal = v_threshold / self.m
        self.surrogate = surrogate_function.apply
        self.v = None
        bias_values = torch.arange(1, m + 1, dtype=torch.float32) * self.v_th_internal
        self.register_buffer('bias', bias_values)
        self.initialized = False

    def forward(self, x: torch.Tensor):
        if not self.initialized:
            self.v = torch.zeros(self.m, *x.shape, device=x.device)
            self.initialized = True
        
        reshaped_bias = self.bias.view(self.m, *([1] * x.dim()))

        if reshaped_bias.device != x.device:
            reshaped_bias = reshaped_bias.to(x.device)

        self.v += x
        spike = self.surrogate(self.v - reshaped_bias)
        spike_sum = torch.sum(spike, dim=0)
        self.v -= spike_sum * self.v_th_internal
        return spike_sum * self.v_th_internal

    def reset(self):
        """
        重置神经元状态。
        """
        self.initialized = False

def reset_neurons(model: nn.Module):
    """
    遍历模型中的所有模块，并调用其 reset 方法（如果存在）
    """
    for m in model.modules():
        if hasattr(m, 'reset'):
            m.reset()

def replace_if_with_gn(model: nn.Module, m: int):
    """
    一个稳健的替换函数，它遍历模型，将 IF 层替换为指定的 GN 脉冲神经元
    """
    print(f"\n[转换步骤] 开始将 IF 神经元替换为 GN(m={m}) 神经元...")
    module_name_list = []
    for name, module in model.named_modules():
        if isinstance(module, IF):
            module_name_list.append(name)
            
    if not module_name_list:
        print("警告: 在模型中未找到任何 IF 层，转换未执行。")
        return model

    for name in module_name_list:
        path = name.split('.')
        parent_module = model
        for i in range(len(path) - 1):
            parent_module = getattr(parent_module, path[i])
        
        child_name = path[-1]
        child_module = getattr(parent_module, child_name)
        
        original_threshold = child_module.thresh.item()
        
        if original_threshold < 0:
            print(f"  - 警告: '{name}' 的原始阈值为负 ({original_threshold:.4f})，SNN神经元通常使用正阈值。将跳过对此层的替换。")
            continue

        new_module = GN(m=m, v_threshold=original_threshold)
        
        setattr(parent_module, child_name, new_module)
        print(f"  - 替换 '{name}' (原始阈值: {original_threshold:.4f}) -> GN(m={m})")
        
    print("所有合格的 IF 层已成功替换为 GN 层。")
    return model


def eval_snn(model: nn.Module, test_loader, device, sim_len: int):
    """
    评估SNN模型在多个时间步长上的性能 (已优化)
    """
    model.eval()
    num_samples = len(test_loader.dataset)
    # accuracies_per_step[t] 将存储在时间步 t+1 时总的正确预测数
    accuracies_per_step = np.zeros(sim_len)
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="SNN评估中"):
            images = images.to(device)
            labels = labels.to(device)
            
            reset_neurons(model)
            
            # 累加每个时间步的输出
            outputs_accumulator = torch.zeros(images.size(0), 1000, device=device) # 适配ImageNet的1000个类别
            
            for t in range(sim_len):
                out = model(images)
                outputs_accumulator += out
                
                # 计算到当前时间步为止的累计投票结果
                _, predicted = torch.max(outputs_accumulator, 1)
                
                # 累加当前时间步的正确样本数
                accuracies_per_step[t] += (predicted == labels).sum().item()

    # 返回每个时间步的累计正确率
    return accuracies_per_step / num_samples


# =========================================================================================
#  主程序逻辑
# =========================================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SNN Conversion and Evaluation Script with GN Neurons (Adapted for ImageNet)')
    
    # 适配ImageNet和ResNet34
    parser.add_argument('-data', '--dataset', type=str, default='imagenet', help='数据集名称 (例如: cifar10, imagenet)')
    parser.add_argument('-arch','--model', type=str, default='resnet34', help='网络架构 (例如: vgg16, resnet34)')
    parser.add_argument('-id', '--identifier', type=str, required=True, help='已训练好的ANN模型标识符 (例如: ImageNet-ResNet34-t8)')
    
    parser.add_argument('--tau', type=int, default=4, help='当使用GN神经元时，每个组的成员数量(m)')
    parser.add_argument('-T', '--sim_len', type=int, default=16, help='SNN的总仿真时长 (Timesteps)')
    
    parser.add_argument('-b','--batch_size', type=int, default=4, help='测试时使用的批量大小 (根据显存调整)')
    parser.add_argument('-dev','--device', type=str, default='0', help='运行设备ID (例如: 0)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    args = parser.parse_args()
    
    # --- **关键修复**：正确设置设备 ---
    device_str = args.device
    if device_str.isdigit():
        device_str = f"cuda:{device_str}"
    
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device.index)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("=" * 60)
    print("SNN (GN Neuron) 转换与评估脚本 (ImageNet适配版)")
    print(f"  - 模型架构: {args.model}")
    print(f"  - 数据集: {args.dataset}")
    print(f"  - 模型标识符: {args.identifier}")
    print(f"  - 转换为: GN Neuron (组员数 m={args.tau})")
    print(f"  - 仿真时长: {args.sim_len} Timesteps")
    print(f"  - 运行设备: {device}")
    print("=" * 60)

    print("\n[步骤 1] 正在加载数据集...")
    _, test_loader = datapool(args.dataset, args.batch_size)
    print("数据集加载完成。")

    print("\n[步骤 2] 正在构建原始ANN模型...")
    model = modelpool(args.model, args.dataset)
    model.to(device)
    print("模型构建完成。")
    
    model_dir = f'{args.dataset}-checkpoints'
    model_path = os.path.join(model_dir, args.identifier + '.pth')
    print(f"\n[步骤 3] 正在从 '{model_path}' 加载预训练权重...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"错误: 找不到模型权重文件 '{model_path}'。请确保路径和标识符正确。")
    
    state_dict = torch.load(model_path, map_location=device)

    # --- **关键修复**：添加兼容旧版模型权重的逻辑 ---
    keys = list(state_dict.keys())
    for k in keys:
        if "relu.up" in k:
            state_dict[k[:-7] + 'act.thresh'] = state_dict.pop(k)
        elif "up" in k:
            state_dict[k[:-2] + 'thresh'] = state_dict.pop(k)

    model.load_state_dict(state_dict) 
    print("权重加载成功。")

    # 将IF神经元替换为GN神经元
    model = replace_if_with_gn(model, m=args.tau)
    
    print("\n[步骤 4] 正在评估SNN模型性能...")
    accuracies = eval_snn(model, test_loader, device=device, sim_len=args.sim_len)

    print("\n================ SNN 性能评估结果 =================\n")
    for t in range(args.sim_len):
        print(f"  - 时间步 (Timestep) {t + 1}: 精度 (Accuracy) = {accuracies[t]:.4f}")
    
    print(f"\n  - 最终精度 (T={args.sim_len}): {accuracies[-1]:.4f}")
    print("\n======================================================")
    print("评估结束。")