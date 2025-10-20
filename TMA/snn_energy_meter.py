import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import copy
from modules import MPLayer
from utils import replace_activation_by_MPLayer, replace_MPLayer_by_neuron

class SNNInferenceEnergyMeter:
    """
    SNN推理能耗测量器
    基于文献《To_Spike_or_Not_to_Spike_A_Digital_Hardware_Perspective_on_Deep_Learning_Acceleration.pdf》
    的能耗计算模型，适配ResNet18 ANN转SNN的推理过程
    """
    
    def __init__(self, model, input_shape=(1, 3, 32, 32), device='cuda'):
        """
        初始化能耗测量器
        
        Args:
            model: 转换后的SNN模型（包含MPLayer）
            input_shape: 输入数据形状 (batch_size, channels, height, width)
            device: 计算设备
        """
        self.model = model
        self.input_shape = input_shape
        self.device = device
        self.total_ac_ops = 0  # 累加操作次数
        self.total_mac_ops = 0 
        # 参考文献Table I：45nm CMOS工艺的基础能耗参数
        self.add_energy_pj = 0.03      # 8bit加法操作能耗为0.03pJ
        self.mul_energy_pj = 0.20      # 8bit乘法操作能耗为0.20pJ
        self.sram_read_energy_pj = 10.00   # 64b SRAM读取能耗为10.00pJ
        self.sram_write_energy_pj = 10.00  # 64b SRAM写入能耗为10.00pJ
        
        # 能耗统计变量
        self.reset_energy_stats()
        
        # 注册前向传播钩子
        self.hooks = []
        self._register_hooks()
        
    def reset_energy_stats(self):
        """重置能耗统计"""
        self.synaptic_accumulation_energy = 0.0
        self.weight_movement_energy = 0.0
        self.threshold_movement_energy = 0.0
        self.potential_movement_energy = 0.0
        self.neuron_operation_energy = 0.0
        self.total_ac_ops = 0
        self.total_mac_ops = 0
    
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

    def _register_hooks(self):
        """注册前向传播钩子以监控能耗"""
        def conv_hook(module, input, output):
            if isinstance(module, nn.Conv2d):
                self._calculate_conv_energy(module, input[0], output)
                
        def mplayer_hook(module, input, output):
            if isinstance(module, MPLayer):
                self._calculate_mplayer_energy(module, input[0], output)
                
        def fc_hook(module, input, output):
            if isinstance(module, nn.Linear):
                self._calculate_fc_energy(module, input[0], output)
        
        # 为所有相关层注册钩子
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                hook = module.register_forward_hook(conv_hook)
                self.hooks.append(hook)
            elif isinstance(module, MPLayer):
                hook = module.register_forward_hook(mplayer_hook)
                self.hooks.append(hook)
            elif isinstance(module, nn.Linear):
                hook = module.register_forward_hook(fc_hook)
                self.hooks.append(hook)
    
    def _calculate_conv_energy(self, conv_layer, input_tensor, output_tensor):
        """计算卷积层的能耗"""
        batch_size = input_tensor.shape[0]
        in_channels = conv_layer.in_channels
        out_channels = conv_layer.out_channels
        kernel_size = conv_layer.kernel_size[0] * conv_layer.kernel_size[1]
        output_height = output_tensor.shape[2]
        output_width = output_tensor.shape[3]
        
        # 权重数量
        weight_count = in_channels * out_channels * kernel_size
        
        # 权重移动能耗：每个时间步都需要读取权重
        # 参考文献：权重读取次数 = 权重总数量 × 时间步数
        timesteps = getattr(conv_layer, 'sim_len', 1)  # 如果没有时间步信息，默认为1
        weight_reads = weight_count * timesteps * batch_size
        self.weight_movement_energy += weight_reads * self.sram_read_energy_pj
        
    def _calculate_mplayer_energy(self, mplayer, input_tensor, output_tensor):
        """计算MPLayer（IF神经元）的能耗"""
        batch_size = input_tensor.shape[0]
        
        # 获取神经元数量
        if input_tensor.dim() == 4:  # 卷积层输出
            neuron_count = input_tensor.shape[1] * input_tensor.shape[2] * input_tensor.shape[3]
        else:  # 全连接层输出
            neuron_count = input_tensor.shape[1]
            
        timesteps = mplayer.sim_len
        ac_ops = neuron_count * timesteps * batch_size
        self.total_ac_ops += ac_ops
        
        # 1. 突触累积能耗
        # 参考文献：累加次数 = 输入特征数 × 时间步数
        if input_tensor.dim() == 4:
            # 对于卷积层，累加次数 = 卷积核尺寸 × 输入通道数 × 输出像素数 × 时间步数
            accumulation_ops = neuron_count * timesteps * batch_size
        else:
            # 对于全连接层
            accumulation_ops = neuron_count * timesteps * batch_size
            
        self.synaptic_accumulation_energy += accumulation_ops * self.add_energy_pj
        
        # 2. 阈值移动能耗
        # 参考文献：阈值读取次数 = 神经元数量 × 时间步数
        threshold_reads = neuron_count * timesteps * batch_size
        self.threshold_movement_energy += threshold_reads * self.sram_read_energy_pj
        
        # 3. 膜电位移动能耗
        # 参考文献：每次时间步需读取上一时刻电位、写入当前时刻电位
        potential_reads = neuron_count * timesteps * batch_size
        potential_writes = neuron_count * timesteps * batch_size
        self.potential_movement_energy += (potential_reads * self.sram_read_energy_pj + 
                                         potential_writes * self.sram_write_energy_pj)
        
        # 4. 神经元模型操作能耗（适配IF神经元特性）
        # 包括：膜电位更新（加法）、阈值比较、电位重置（减法）
        # IF神经元无泄漏系数，无需乘法操作
        
        # 膜电位更新：输入累加后更新电位
        potential_update_ops = neuron_count * timesteps * batch_size
        
        # 阈值比较：判断是否触发脉冲
        threshold_compare_ops = neuron_count * timesteps * batch_size
        
        # 电位重置：若触发则清零（假设平均50%的神经元触发）
        reset_ops = neuron_count * timesteps * batch_size * 0.5
        
        neuron_ops_total = potential_update_ops + threshold_compare_ops + reset_ops
        self.neuron_operation_energy += neuron_ops_total * self.add_energy_pj
        
    def _calculate_fc_energy(self, fc_layer, input_tensor, output_tensor):
        """计算全连接层的能耗"""
        batch_size = input_tensor.shape[0]
        in_features = fc_layer.in_features
        out_features = fc_layer.out_features
        
        # 权重数量
        weight_count = in_features * out_features
        
        # 权重移动能耗
        timesteps = getattr(fc_layer, 'sim_len', 1)
        weight_reads = weight_count * timesteps * batch_size
        self.weight_movement_energy += weight_reads * self.sram_read_energy_pj
        
    def measure_single_inference(self, input_data, timesteps=8):
        """
        测量单张图像的推理能耗
        
        Args:
            input_data: 输入数据张量
            timesteps: SNN推理时间步数
            
        Returns:
            Dict: 能耗统计结果（单位：nJ）
        """
        self.reset_energy_stats()
        
        # 设置模型的时间步数
        self._set_model_timesteps(timesteps)
        
        # 执行前向传播
        with torch.no_grad():
            _ = self.model(input_data)
            
        # 计算总能耗并转换为nJ（1pJ = 0.001nJ）
        total_energy = (self.synaptic_accumulation_energy + 
                       self.weight_movement_energy + 
                       self.threshold_movement_energy + 
                       self.potential_movement_energy + 
                       self.neuron_operation_energy) * 0.001
        
        return {
            "total_energy": total_energy,
            "synaptic_accumulation": self.synaptic_accumulation_energy * 0.001,
            "data_movement": {
                "weight": self.weight_movement_energy * 0.001,
                "threshold": self.threshold_movement_energy * 0.001,
                "potential": self.potential_movement_energy * 0.001
            },
            "neuron_operation": self.neuron_operation_energy * 0.001,
            "computation_stats": {
                "total_ac_ops": self.total_ac_ops,
                "total_mac_ops": self.total_mac_ops,
                "total_ops": self.total_ac_ops + self.total_mac_ops
            }
        }
        
    def measure_batch_inference(self, dataloader, timesteps_list=[8, 16, 32], max_samples=100):
        """
        测量批量推理的平均能耗
        
        Args:
            dataloader: 数据加载器
            timesteps_list: 要测试的时间步数列表
            max_samples: 最大测试样本数
            
        Returns:
            Dict: 不同时间步下的平均能耗统计
        """
        results = {}
        
        for timesteps in timesteps_list:
            print(f"\n测量 T={timesteps} 时间步的能耗...")
            
            total_energies = []
            sample_count = 0
            
            for batch_idx, (images, labels) in enumerate(dataloader):
                if sample_count >= max_samples:
                    break
                    
                images = images.to(self.device)
                
                # 逐张图像测量能耗
                for i in range(min(images.shape[0], max_samples - sample_count)):
                    single_image = images[i:i+1]
                    energy_result = self.measure_single_inference(single_image, timesteps)
                    total_energies.append(energy_result["total_energy"])
                    sample_count += 1
                    
                    if sample_count >= max_samples:
                        break
            
            # 计算平均能耗
            avg_energy = sum(total_energies) / len(total_energies)
            results[f"T={timesteps}"] = {
                "average_total_energy_nJ": avg_energy,
                "samples_tested": len(total_energies)
            }
            
            print(f"T={timesteps}: 平均能耗 = {avg_energy:.3f} nJ (测试样本: {len(total_energies)})")
            
        return results
        
    def _set_model_timesteps(self, timesteps):
        """设置模型中所有MPLayer的时间步数"""
        for module in self.model.modules():
            if isinstance(module, MPLayer):
                module.sim_len = timesteps
                
    def remove_hooks(self):
        """移除所有注册的钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
    def __del__(self):
        """析构函数，自动清理钩子"""
        self.remove_hooks()


def create_energy_measurement_demo():
    """
    创建能耗测量演示代码
    """
    # 示例使用代码
    demo_code = '''
# 使用示例：ResNet18 SNN能耗测量
from Models.ResNet import resnet18
from dataprocess import PreProcess_Cifar10
from snn_energy_meter import SNNInferenceEnergyMeter
import torch

# 1. 加载预训练的ANN模型
model = resnet18(num_classes=10)
model.load_state_dict(torch.load('model/your_trained_model.pth'))

# 2. 转换为SNN
from utils import replace_activation_by_MPLayer
snn_model = replace_activation_by_MPLayer(model, presim_len=4, sim_len=8, batchsize=1)
snn_model.eval()

# 3. 准备测试数据
test_dataloader = PreProcess_Cifar10(batch_size=1, test_only=True)

# 4. 创建能耗测量器
energy_meter = SNNInferenceEnergyMeter(
    model=snn_model,
    input_shape=(1, 3, 32, 32),
    device='cuda'
)

# 5. 测量单张图像能耗
test_image = torch.randn(1, 3, 32, 32).cuda()
single_result = energy_meter.measure_single_inference(test_image, timesteps=8)
print("单张CIFAR10图像推理能耗 (T=8):")
print(f"总能耗: {single_result['total_energy']:.3f} nJ")
print(f"突触累积: {single_result['synaptic_accumulation']:.3f} nJ")
print(f"权重移动: {single_result['data_movement']['weight']:.3f} nJ")
print(f"阈值移动: {single_result['data_movement']['threshold']:.3f} nJ")
print(f"电位移动: {single_result['data_movement']['potential']:.3f} nJ")
print(f"神经元操作: {single_result['neuron_operation']:.3f} nJ")

# 6. 测量不同时间步的批量平均能耗
batch_results = energy_meter.measure_batch_inference(
    dataloader=test_dataloader,
    timesteps_list=[8, 16, 32],
    max_samples=50
)

print("\n批量推理平均能耗:")
for timestep, result in batch_results.items():
    print(f"{timestep}: {result['average_total_energy_nJ']:.3f} nJ")

# 7. 清理资源
energy_meter.remove_hooks()
'''
    
    return demo_code


if __name__ == "__main__":
    # 打印使用示例
    print("SNN推理能耗测量器使用示例:")
    print(create_energy_measurement_demo())