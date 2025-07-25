import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import copy
from typing import Callable, List, Tuple, Dict, Optional
from spikingjelly.clock_driven import functional


class BPTTCalibrator:
    """
    基于 BPTT 的 SNN 校准器，用于提高 SNN 在低延迟情况下的性能
    """

    def __init__(self,
                 device: torch.device = torch.device('cuda:0'),
                 loss_function: nn.Module = nn.MSELoss(),
                 optimizer_class: optim.Optimizer = optim.Adam,
                 lr: float = 0.001,
                 T: int = 32,
                 epochs: int = 5,
                 batch_size: int = 64,
                 weight_decay: float = 1e-5,
                 scheduler_type: str = 'cosine',
                 kl_weight: float = 0.1,  # KL散度权重参数
                 mse_weight: float = 2,  # MSE损失权重参数
                 reg_weight: float = 0.5,  # 权重正则化权重参数
                 dynamic_weights: bool = True,  # 是否使用动态权重调整
                 use_warmup: bool = True,  # 预热参数
                 warmup_epochs: int = 2,   # 预热轮数
                 freeze_bn: bool = True):  # 冻结BN参数
        """
        初始化 BPTT 校准器

        参数:
            device: 训练设备
            loss_function: 损失函数，默认为 MSE
            optimizer_class: 优化器类，默认为 Adam
            lr: 学习率
            T: 时间步长
            epochs: 训练轮数
            batch_size: 批大小
            weight_decay: 权重衰减
            scheduler_type: 学习率调度器类型，可选 'step', 'cosine', 'none'
            kl_weight: KL散度损失权重
            mse_weight: MSE损失权重
            reg_weight: 权重正则化权重
            dynamic_weights: 是否使用动态权重调整
            use_warmup: 是否使用学习率预热
            warmup_epochs: 预热轮数
            freeze_bn: 是否冻结批归一化层
        """
        self.device = device
        self.loss_function = loss_function
        self.optimizer_class = optimizer_class
        self.lr = lr
        self.T = T
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        self.kl_weight = kl_weight
        self.mse_weight = mse_weight
        self.reg_weight = reg_weight
        self.dynamic_weights = dynamic_weights
        self.use_warmup = use_warmup
        self.warmup_epochs = warmup_epochs
        self.freeze_bn = freeze_bn

        # 添加评估相关属性
        self.do_per_epoch_eval = False
        self.test_dataloader = None
        self.eval_timesteps = None
        self.cap_dataset = None
        self.presim_len = None
        self.net_arch = None
       # 添加早停相关属性
        self.early_stopping = True
        self.patience = 8
        self.best_acc = 0
        self.no_improve_count = 0

    def enable_per_epoch_eval(self, test_dataloader, eval_timesteps, cap_dataset, presim_len, net_arch=''):
        """
        启用每轮校准后的评估

        参数:
            test_dataloader: 测试数据加载器
            eval_timesteps: 评估的时间步长列表
            cap_dataset: 数据集大小
            presim_len: 预模拟长度
            net_arch: 网络架构
        """
        self.do_per_epoch_eval = True
        self.test_dataloader = test_dataloader
        self.eval_timesteps = eval_timesteps
        self.cap_dataset = cap_dataset
        self.presim_len = presim_len
        self.net_arch = net_arch

    def evaluate_model(self, model):
        """
        评估模型在不同时间步的精度

        参数:
            model: 待评估的模型

        返回:
            包含每个时间步精度的字典
        """
        # 导入评估函数
        from main import evaluate_snn_at_timesteps

        # 复制模型以避免修改原始模型
        eval_model = copy.deepcopy(model)

        # 评估模型
        accuracies = evaluate_snn_at_timesteps(
            eval_model,
            self.test_dataloader,
            self.eval_timesteps,
            self.device,
            self.cap_dataset,
            self.presim_len,
            self.net_arch
        )

        return accuracies

    def calibrate(self,
                 ann_model: nn.Module,
                 snn_model: nn.Module,
                 train_samples: torch.Tensor,
                 val_samples: torch.Tensor = None,
                 encoder: Callable = None) -> nn.Module:
        """
        使用 BPTT 校准 SNN 模型

        参数:
            ann_model: ANN 模型
            snn_model: 待校准的 SNN 模型
            train_samples: 训练样本
            val_samples: 验证样本
            encoder: 编码器，用于将输入转换为脉冲序列

        返回:
            校准后的 SNN 模型
        """
        print("开始基于 BPTT 的 SNN 校准...")

        # 确保 ANN 和 SNN 都在评估模式
        ann_model.eval()
        snn_model.train()

        # 如果需要冻结BN层
        if self.freeze_bn:
            for name, module in snn_model.named_modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
                    print(f"冻结批归一化层: {name}")

        # 获取 ANN 的输出作为目标
        with torch.no_grad():
            ann_outputs = ann_model(train_samples)

        # 准备数据集
        dataset = torch.utils.data.TensorDataset(train_samples, ann_outputs)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        # 配置优化器，添加权重衰减
        if self.optimizer_class == optim.Adam:
            optimizer = self.optimizer_class(snn_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            optimizer = self.optimizer_class(snn_model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)

        # 配置学习率调度器
        if self.scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
        elif self.scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        else:
            scheduler = None

        # 添加学习率预热
        if self.use_warmup:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=self.warmup_epochs * len(dataloader)
            )

        # 训练循环
        best_loss = float('inf')
        best_model = None
        best_epoch_acc = 0
        best_min_acc = 0  # 记录最佳最小精度
        best_consistency = float('-inf')  # 记录最佳一致性得分

        # 保存初始精度，用于比较
        initial_accuracies = None

        # 保存初始模型权重，用于正则化
        initial_weights = {}
        for name, param in snn_model.named_parameters():
            if param.requires_grad:
                initial_weights[name] = param.data.clone()

        for epoch in range(self.epochs):
            total_loss = 0.0
            batch_count = 0

            # 设置学习率
            if epoch == 0:
                print(f"初始学习率: {self.lr:.6f}")

            # 计算当前训练进度，用于动态权重调整
            progress = epoch / (self.epochs - 1) if self.epochs > 1 else 0

            for inputs, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                batch_size = inputs.size(0)  # 获取当前批次大小

                # 重置 SNN 的状态
                functional.reset_net(snn_model)

                # 前向传播
                if encoder is not None:
                    spike_inputs = encoder(inputs)
                    outputs = 0
                    for t in range(self.T):
                        outputs += snn_model(spike_inputs[t])
                    outputs = outputs / self.T
                else:
                    outputs = 0
                    for t in range(self.T):
                        curr_output = snn_model(inputs)
                        # 确保输出形状正确
                        if curr_output.size(0) != batch_size:
                            # 如果输出形状不匹配，进行调整
                            curr_output = curr_output.view(batch_size, -1, curr_output.size(-1)).mean(dim=1)
                        outputs += curr_output
                    outputs = outputs / self.T

                # 确保输出和目标形状匹配
                if outputs.size(0) != targets.size(0):
                    print(f"警告: 输出形状 {outputs.shape} 与目标形状 {targets.shape} 不匹配，尝试调整...")
                    # 尝试调整输出形状以匹配目标
                    outputs = outputs.view(targets.size(0), -1, outputs.size(-1)).mean(dim=1)

                # 计算组合损失：MSE + KL散度 + 权重正则化
                mse_loss = F.mse_loss(outputs, targets)

                # 对输出进行softmax处理，以便计算KL散度
                log_outputs = F.log_softmax(outputs, dim=1)
                targets_prob = F.softmax(targets, dim=1)
                kl_loss = F.kl_div(log_outputs, targets_prob, reduction='batchmean')

                # 添加权重正则化损失，防止权重偏离初始值太远
                l2_reg_loss = 0
                for name, param in snn_model.named_parameters():
                    if name in initial_weights:
                        l2_reg_loss += torch.sum((param - initial_weights[name])**2)

                # 动态调整权重（如果启用）
                if self.dynamic_weights:
                    # 随着训练进行，增加MSE权重，减少正则化权重
                    current_mse_weight = self.mse_weight * (1.0 + 0.5 * progress)  # 从mse_weight增加到1.5*mse_weight
                    current_reg_weight = self.reg_weight * (1.0 - 0.5 * progress)  # 从reg_weight减少到0.5*reg_weight
                    current_kl_weight = self.kl_weight * (1.0 - 0.3 * progress)    # 从kl_weight减少到0.7*kl_weight
                else:
                    current_mse_weight = self.mse_weight
                    current_reg_weight = self.reg_weight
                    current_kl_weight = self.kl_weight

                # 组合损失，使用优化后的权重
                loss = current_mse_weight * mse_loss + current_kl_weight * kl_loss + current_reg_weight * l2_reg_loss

                # 打印当前使用的权重（每个epoch的第一个batch）
                if batch_count == 0:
                    print(f"当前损失权重 - MSE: {current_mse_weight:.4f}, KL: {current_kl_weight:.4f}, REG: {current_reg_weight:.4f}")

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(snn_model.parameters(), max_norm=1.0)

                optimizer.step()

                # 学习率预热
                if self.use_warmup and epoch < self.warmup_epochs:
                    warmup_scheduler.step()

                total_loss += loss.item()
                batch_count += 1

            # 更新学习率
            if scheduler is not None:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
                print(f"当前学习率: {current_lr:.6f}")

            avg_loss = total_loss / batch_count
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}")

            # 保存最佳模型（基于损失）
            if avg_loss < best_loss:
                best_loss = avg_loss
                if not self.do_per_epoch_eval:  # 如果不评估精度，则基于损失保存最佳模型
                    best_model = copy.deepcopy(snn_model)
                    print(f"发现更好的模型，损失: {best_loss:.6f}")

            # 每轮校准后评估模型精度
            if self.do_per_epoch_eval:
                print(f"\n=== 第 {epoch+1} 轮校准后的模型精度 ===")
                accuracies = self.evaluate_model(snn_model)

                print(f"{'时间步':<10}{'精度':<15}")
                print("-" * 25)

                # 保存第一轮的精度作为基准
                if epoch == 0 and initial_accuracies is None:
                    initial_accuracies = accuracies.copy()
                    print("记录初始精度作为基准")

                # 计算各项指标
                acc_values = []
                min_acc = float('inf')
                max_acc = 0

                for t in sorted(self.eval_timesteps):
                    if t in accuracies:
                        acc = accuracies[t]
                        print(f"{t:<10}{acc:.4f}")
                        acc_values.append(acc)
                        min_acc = min(min_acc, acc)
                        max_acc = max(max_acc, acc)

                # 计算平均精度
                avg_acc = sum(acc_values) / len(acc_values) if acc_values else 0

                # 计算一致性得分 (标准差的负值，越接近0越好)
                std_dev = np.std(acc_values) if len(acc_values) > 1 else 0
                consistency_score = -std_dev  # 负标准差，越高越好

                # 计算相对于初始精度的提升
                if initial_accuracies:
                    improvements = []
                    for t in sorted(self.eval_timesteps):
                        if t in accuracies and t in initial_accuracies:
                            imp = accuracies[t] - initial_accuracies[t]
                            improvements.append(imp)

                    avg_improvement = sum(improvements) / len(improvements) if improvements else 0
                    min_improvement = min(improvements) if improvements else 0

                    print(f"\n相对于初始精度:")
                    print(f"平均提升: {avg_improvement:.4f}")
                    print(f"最小提升: {min_improvement:.4f}")

                print(f"\n平均精度: {avg_acc:.4f}")
                print(f"最小精度: {min_acc:.4f}")
                print(f"精度标准差: {std_dev:.4f}")

                # 综合评分：最小精度 * 0.5 + 平均精度 * 0.3 + 一致性得分 * 0.2
                # 这个公式可以根据需要调整权重
                combined_score = min_acc * 0.5 + avg_acc * 0.3 + consistency_score * 0.2
                print(f"综合评分: {combined_score:.4f}")

                # 检查是否有提升 - 使用综合评分而不仅仅是平均精度
                if combined_score > best_consistency:
                    best_consistency = combined_score
                    best_model = copy.deepcopy(snn_model)
                    print(f"发现更好的模型，综合评分: {best_consistency:.4f}")
                    self.no_improve_count = 0
                elif min_acc > best_min_acc and min_acc > 0:
                    # 如果综合评分没有提高，但最小精度提高了，也认为是更好的模型
                    best_min_acc = min_acc
                    best_model = copy.deepcopy(snn_model)
                    print(f"发现更好的模型，最小精度: {best_min_acc:.4f}")
                    self.no_improve_count = 0
                else:
                    self.no_improve_count += 1
                    print(f"精度未提升，已连续 {self.no_improve_count} 轮")

                    # 早停检查
                    if self.early_stopping and self.no_improve_count >= self.patience:
                        print(f"早停：连续 {self.patience} 轮精度未提升，停止训练")
                        break

                print()

        print(f"BPTT 校准完成，最佳损失: {best_loss:.6f}")

        # 如果有验证集，评估校准效果
        if val_samples is not None:
            try:
                self._evaluate_calibration(ann_model, best_model, val_samples, encoder)
            except Exception as e:
                print(f"评估校准效果时发生错误: {str(e)}")
                print("继续执行，返回最佳模型...")

        return best_model


    def _evaluate_calibration(self,
                              ann_model: nn.Module,
                              snn_model: nn.Module,
                              val_samples: torch.Tensor,
                              encoder: Callable = None):
        """
        评估校准效果

        参数:
            ann_model: ANN 模型
            snn_model: 校准后的 SNN 模型
            val_samples: 验证样本
            encoder: 编码器
        """
        ann_model.eval()
        snn_model.eval()

        # 打印输入样本形状，帮助调试
        print(f"验证样本形状: {val_samples.shape}")

        # 确保使用较小的批次大小，避免内存问题
        batch_size = min(32, val_samples.size(0))
        print(f"使用批次大小: {batch_size}")

        # 完全重新处理，不依赖之前的实现
        try:
            with torch.no_grad():
                # 将验证样本分成小批次
                num_samples = val_samples.size(0)
                num_batches = (num_samples + batch_size - 1) // batch_size  # 向上取整

                all_ann_outputs = []
                all_snn_outputs = []

                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min(start_idx + batch_size, num_samples)
                    batch_samples = val_samples[start_idx:end_idx].to(self.device)

                    # 获取当前批次的 ANN 输出
                    batch_ann_outputs = ann_model(batch_samples)

                    # 重置 SNN 状态
                    functional.reset_net(snn_model)

                    # 获取当前批次的 SNN 输出
                    if encoder is not None:
                        spike_inputs = encoder(batch_samples)
                        batch_snn_outputs = 0
                        for t in range(self.T):
                            batch_snn_outputs += snn_model(spike_inputs[t])
                        batch_snn_outputs = batch_snn_outputs / self.T
                    else:
                        batch_snn_outputs = 0
                        for t in range(self.T):
                            batch_snn_outputs += snn_model(batch_samples)
                        batch_snn_outputs = batch_snn_outputs / self.T

                    # 保存当前批次的输出
                    all_ann_outputs.append(batch_ann_outputs)
                    all_snn_outputs.append(batch_snn_outputs)

                    # 打印当前批次的形状，帮助调试
                    print(f"批次 {i+1}/{num_batches}: ANN输出形状 {batch_ann_outputs.shape}, SNN输出形状 {batch_snn_outputs.shape}")

                # 确保所有批次的输出形状一致，然后再合并
                first_ann_shape = all_ann_outputs[0].shape[1:]  # 忽略批次维度
                first_snn_shape = all_snn_outputs[0].shape[1:]

                for i in range(1, len(all_ann_outputs)):
                    if all_ann_outputs[i].shape[1:] != first_ann_shape:
                        print(f"警告: 第 {i+1} 批次的 ANN 输出形状与第一批次不一致")
                    if all_snn_outputs[i].shape[1:] != first_snn_shape:
                        print(f"警告: 第 {i+1} 批次的 SNN 输出形状与第一批次不一致")

                # 合并所有批次的输出
                ann_outputs = torch.cat(all_ann_outputs, dim=0)
                snn_outputs = torch.cat(all_snn_outputs, dim=0)

                print(f"合并后: ANN输出形状 {ann_outputs.shape}, SNN输出形状 {snn_outputs.shape}")

                # 计算 MSE
                if ann_outputs.shape == snn_outputs.shape:
                    mse = nn.MSELoss()(snn_outputs, ann_outputs).item()
                    print(f"校准后 SNN 与 ANN 的 MSE: {mse:.6f}")
                else:
                    print(f"错误: 无法计算 MSE，因为 ANN 输出形状 {ann_outputs.shape} 与 SNN 输出形状 {snn_outputs.shape} 不匹配")
                    # 尝试调整形状以计算近似 MSE
                    print("尝试调整形状以计算近似 MSE...")

                    # 如果只是批次大小不同，可以截断或填充
                    min_batch_size = min(ann_outputs.size(0), snn_outputs.size(0))
                    ann_outputs_truncated = ann_outputs[:min_batch_size]
                    snn_outputs_truncated = snn_outputs[:min_batch_size]

                    if ann_outputs_truncated.shape == snn_outputs_truncated.shape:
                        mse = nn.MSELoss()(snn_outputs_truncated, ann_outputs_truncated).item()
                        print(f"截断后的 MSE (使用前 {min_batch_size} 个样本): {mse:.6f}")
                    else:
                        print("即使截断后，形状仍然不匹配，无法计算 MSE")

        except Exception as e:
            print(f"评估校准效果时发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            print("继续执行，返回最佳模型...")

        return


def calibrate_snn_with_bptt(ann_model: nn.Module,
                            snn_model: nn.Module,
                            train_samples: torch.Tensor,
                            val_samples: torch.Tensor = None,
                            device: torch.device = torch.device('cuda:0'),
                            T: int = 32,
                            epochs: int = 5,
                            batch_size: int = 64,
                            lr: float = 0.001,
                            encoder: Callable = None,
                            test_dataloader=None,
                            eval_timesteps=None,
                            cap_dataset=10000,
                            presim_len=0,
                            net_arch='',
                            weight_decay=1e-5,
                            scheduler_type='cosine',
                            kl_weight=0.1,
                            mse_weight=1.5,  # 添加MSE损失权重参数
                            reg_weight=0.05,  # 添加权重正则化权重参数
                            dynamic_weights=True,  # 添加动态权重调整参数
                            use_warmup=True,
                            warmup_epochs=2,
                            freeze_bn=True) -> nn.Module:
    """
    使用 BPTT 校准 SNN 模型的便捷函数

    参数:
        ann_model: ANN 模型
        snn_model: 待校准的 SNN 模型
        train_samples: 训练样本
        val_samples: 验证样本
        device: 训练设备
        T: 时间步长
        epochs: 训练轮数
        batch_size: 批大小
        lr: 学习率
        encoder: 编码器
        test_dataloader: 测试数据加载器，用于评估每轮校准后的精度
        eval_timesteps: 评估的时间步长列表
        cap_dataset: 数据集大小，用于计算精度
        presim_len: 预模拟长度
        net_arch: 网络架构
        weight_decay: 权重衰减
        scheduler_type: 学习率调度器类型
        kl_weight: KL散度损失权重
        mse_weight: MSE损失权重
        reg_weight: 权重正则化权重
        dynamic_weights: 是否使用动态权重调整
        use_warmup: 是否使用学习率预热
        warmup_epochs: 预热轮数
        freeze_bn: 是否冻结批归一化层

    返回:
        校准后的 SNN 模型
    """
    calibrator = BPTTCalibrator(
        device=device,
        T=T,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        scheduler_type=scheduler_type,
        kl_weight=kl_weight,
        mse_weight=mse_weight,
        reg_weight=reg_weight,
        dynamic_weights=dynamic_weights,
        use_warmup=use_warmup,
        warmup_epochs=warmup_epochs,
        freeze_bn=freeze_bn
    )

    # 如果提供了测试数据，启用每轮评估
    if test_dataloader is not None and eval_timesteps is not None:
        calibrator.enable_per_epoch_eval(
            test_dataloader=test_dataloader,
            eval_timesteps=eval_timesteps,
            cap_dataset=cap_dataset,
            presim_len=presim_len,
            net_arch=net_arch
        )

    return calibrator.calibrate(
        ann_model=ann_model,
        snn_model=snn_model,
        train_samples=train_samples,
        val_samples=val_samples,
        encoder=encoder
    )