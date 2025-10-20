import torch
import torch.nn as nn
from typing import Dict

class ANNInferenceEnergyMeter:
    """ANN推理能耗测量器"""
    
    def __init__(self, model, input_shape=(1, 3, 32, 32), device='cuda'):
        self.model = model
        self.input_shape = input_shape
        self.device = device
        self.total_mac_ops = 0  
        # 45nm CMOS工艺的基础能耗参数
        self.add_energy_pj = 0.03      # 8bit加法操作能耗
        self.mul_energy_pj = 0.20      # 8bit乘法操作能耗
        self.sram_read_energy_pj = 10.00   # 64b SRAM读取能耗
        
        # 能耗统计变量
        self.reset_energy_stats()
        
        # 注册钩子
        self.hooks = []
        self._register_hooks()
    
    def reset_energy_stats(self):
        self.mac_energy = 0.0  # 乘加运算能耗
        self.weight_movement_energy = 0.0  # 权重读取能耗
        self.activation_movement_energy = 0.0  # 激活值读取/写入能耗
        self.total_mac_ops = 0
    
    def _register_hooks(self):
        def conv_hook(module, input, output):
            if isinstance(module, nn.Conv2d):
                self._calculate_conv_energy(module, input[0], output)
        
        def fc_hook(module, input, output):
            if isinstance(module, nn.Linear):
                self._calculate_fc_energy(module, input[0], output)
        
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                self.hooks.append(module.register_forward_hook(conv_hook))
            elif isinstance(module, nn.Linear):
                self.hooks.append(module.register_forward_hook(fc_hook))
    
    def _calculate_conv_energy(self, conv_layer, input_tensor, output_tensor):
        batch_size = input_tensor.shape[0]
        in_channels = conv_layer.in_channels
        out_channels = conv_layer.out_channels
        kernel_size = conv_layer.kernel_size[0] * conv_layer.kernel_size[1]
        output_height = output_tensor.shape[2]
        output_width = output_tensor.shape[3]
        
        # 计算MAC操作次数
        mac_ops = batch_size * out_channels * output_height * output_width * in_channels * kernel_size
        self.total_mac_ops += mac_ops
        
        # 权重读取：每个卷积核权重读取一次
        weight_reads = in_channels * out_channels * kernel_size
        
        # 激活值读取/写入
        activation_reads = batch_size * in_channels * input_tensor.shape[2] * input_tensor.shape[3]
        activation_writes = batch_size * out_channels * output_height * output_width
        
        # 累加能耗
        self.mac_energy += mac_ops * (self.mul_energy_pj + self.add_energy_pj)
        self.weight_movement_energy += weight_reads * self.sram_read_energy_pj
        self.activation_movement_energy += (activation_reads + activation_writes) * self.sram_read_energy_pj
    
    def _calculate_fc_energy(self, fc_layer, input_tensor, output_tensor):
        batch_size = input_tensor.shape[0]
        in_features = fc_layer.in_features
        out_features = fc_layer.out_features
        
        # 计算MAC操作次数
        mac_ops = batch_size * in_features * out_features
        self.total_mac_ops += mac_ops
        
        # 权重读取
        weight_reads = in_features * out_features
        
        # 激活值读取/写入
        activation_reads = batch_size * in_features
        activation_writes = batch_size * out_features
        
        # 累加能耗
        self.mac_energy += mac_ops * (self.mul_energy_pj + self.add_energy_pj)
        self.weight_movement_energy += weight_reads * self.sram_read_energy_pj
        self.activation_movement_energy += (activation_reads + activation_writes) * self.sram_read_energy_pj
    
    def measure_inference(self, input_data) -> Dict:
        self.reset_energy_stats()
        
        with torch.no_grad():
            _ = self.model(input_data)
        
        # 转换为nJ
        total_energy = (self.mac_energy + self.weight_movement_energy + 
                       self.activation_movement_energy) * 0.001
        
        return {
            "total_energy": total_energy,
            "mac_energy": self.mac_energy * 0.001,
            "weight_movement": self.weight_movement_energy * 0.001,
            "activation_movement": self.activation_movement_energy * 0.001,
            "computation_stats": {
                "total_mac_ops": self.total_mac_ops,
                "total_flops": self.total_mac_ops * 2  # 每个MAC操作包含一次乘法和一次加法
            }
        }
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def __del__(self):
        self.remove_hooks()