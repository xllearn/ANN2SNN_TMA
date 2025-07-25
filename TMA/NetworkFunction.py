from torch import nn
import torch
from tqdm import tqdm
from utils import *
import numpy as np
# Import AMP libraries
from torch.cuda.amp import autocast
# Import spikingjelly functional module
from spikingjelly.clock_driven import functional # 确保导入

# 修改 train_ann 函数以接受 scaler 和 use_amp 参数
def train_ann(train_dataloader, test_dataloader, model, epochs, lr, wd, device, save_name, scaler=None, use_amp=False):
    """
    训练 ANN 模型，支持混合精度训练。

    参数:
        train_dataloader: 训练数据加载器
        test_dataloader: 测试数据加载器
        model: ANN 模型
        epochs: 训练轮数
        lr: 学习率
        wd: 权重衰减
        device: 设备
        save_name: 模型保存路径
        scaler: GradScaler 实例 (如果 use_amp=True)
        use_amp: 是否启用混合精度
    """
    model = model.cuda(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    # 余弦退火学习率，T_max设为总epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    best_acc = 0

    for epoch in range(epochs):
        epoch_loss = 0
        lenth = 0
        model.train() # 确保模型处于训练模式
        for img, label in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            img = img.to(device, non_blocking=True) # 使用 non_blocking=True 加速数据传输
            label = label.to(device, non_blocking=True)
            optimizer.zero_grad()

            # 使用 autocast 和 GradScaler 进行混合精度训练
            with autocast(enabled=use_amp):
                out = model(img)
                loss = loss_fn(out, label)

            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                # 可以添加梯度裁剪 (可选)
                # scaler.unscale_(optimizer) # 在裁剪前需要 unscale
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                # 可以添加梯度裁剪 (可选)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            epoch_loss += loss.item() * img.size(0) # 乘以 batch size 获取总损失
            lenth += img.size(0)

        # 计算平均损失
        avg_epoch_loss = epoch_loss / lenth if lenth > 0 else 0

        # 验证
        acc = eval_ann(test_dataloader, model, device)
        print(f"ANNs training Epoch {epoch+1}: Val_loss: {avg_epoch_loss:.6f} Acc: {acc:.4f}")

        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_name)
            print(f"New best accuracy ({best_acc:.4f}) achieved. Model saved to {save_name}")

        # 更新学习率
        scheduler.step()

    print(f"Finished ANN training. Best validation accuracy: {best_acc:.4f}")
    return model


# eval_ann 函数保持不变，因为它不需要混合精度
def eval_ann(test_dataloader, model, device):
    tot = 0
    total_samples = 0
    model.eval() # 确保模型处于评估模式
    model.to(device) # 确保模型在正确的设备上

    with torch.no_grad(): # 评估时不需要计算梯度
        for img, label in tqdm(test_dataloader, desc="Evaluating ANN"):
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            out = model(img)
            tot += (label == out.max(1)[1]).sum().item()
            total_samples += img.size(0) # 使用 img.size(0) 获取当前批次大小

    accuracy = tot / total_samples if total_samples > 0 else 0
    return accuracy


# SNN 相关的函数 (eval_snn, mp_test, cal_message) 保持不变，
# 因为混合精度主要应用于 ANN 的训练阶段以加速。
# 如果需要对 SNN 的推理也进行加速，可以考虑在 SNN 模型上应用 torch.compile，
# 但混合精度对 SNN 推理的加速效果通常不如对 ANN 训练明显。

def eval_snn(test_dataloader, model, sim_len, device):
    tot = torch.zeros(sim_len).to(device) # 确保 tot 在正确的设备上
    model = model.to(device) # 确保模型在正确的设备上
    model.eval() # 确保模型处于评估模式
    total_samples = 0
    with torch.no_grad():
        for img, label in tqdm(test_dataloader, desc="Evaluating SNN"):
            spikes = 0
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            current_batch_size = img.size(0)
            total_samples += current_batch_size

            functional.reset_net(model) # 在每个 batch 开始前重置网络状态
            for t in range(sim_len):
                out = model(img) # 假设模型每次调用模拟一步
                spikes += out
                # 确保 tot 的索引不超过其长度
                if t < sim_len:
                    # 确保 spikes 的形状是 [batch_size, num_classes]
                    if spikes.shape[0] == current_batch_size and spikes.dim() == 2:
                        prediction = spikes.max(1)[1] # 形状应该是 [batch_size]
                        # 比较 prediction 和 label
                        if prediction.shape == label.shape:
                            tot[t] += (label == prediction).sum().item()
                        else:
                            # 如果形状不匹配，打印错误信息
                            # print(f"Shape mismatch in eval_snn at t={t}: label {label.shape}, prediction {prediction.shape}, spikes {spikes.shape}") # 注释掉或删除
                            pass
                    else:
                        # print(f"Unexpected shape for spikes in eval_snn at t={t}: {spikes.shape}") # 注释掉或删除
                        pass

    return tot # 返回的是每个时间步的正确数量累积，需要在外部除以总样本数


# --- 修改 mp_test 函数 ---
def mp_test(test_dataloader, model, net_arch, presim_len, sim_len, device):
    new_tot = torch.zeros(sim_len).to(device) # 确保 new_tot 在正确的设备上
    model = model.to(device) # 确保模型在正确的设备上
    model.eval() # 确保模型处于评估模式
    total_samples = 0

    with torch.no_grad():
        for img, label in tqdm(test_dataloader, desc="MP Testing SNN"):
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            current_batch_size = img.size(0)
            total_samples += current_batch_size

            functional.reset_net(model) # 在每个 batch 开始前重置网络状态

            # --- 调用模型一次，获取所有时间步的输出 ---
            # 假设 model(img) 返回 [sim_len * batch_size, num_classes]
            out_flat = model(img)

            # --- 验证并重塑输出 ---
            num_classes = out_flat.shape[-1] # 获取类别数
            expected_elements = sim_len * current_batch_size * num_classes
            if out_flat.numel() != expected_elements or out_flat.dim() != 2:
                 # print(f"Warning: Unexpected output shape from model(img) in mp_test: {out_flat.shape}. Expected [{sim_len * current_batch_size}, {num_classes}]") # 注释掉或删除
                 continue # 如果形状不对，跳过这个批次

            # 重塑为 [T, N, C]
            try:
                out_seq = out_flat.view(sim_len, current_batch_size, num_classes)
            except RuntimeError as e:
                print(f"Error reshaping output in mp_test: {e}. Output shape: {out_flat.shape}, Expected elements: {expected_elements}")
                continue # 跳过这个批次

            # --- 计算累积精度 ---
            # 沿时间维度累积脉冲（或电压）
            spikes_accumulator_seq = torch.cumsum(out_seq, dim=0) # Shape [T, N, C]

            for t in range(sim_len):
                current_spikes = spikes_accumulator_seq[t] # Shape [N, C]
                prediction = current_spikes.max(1)[1]      # Shape [N]
                if prediction.shape == label.shape:
                    new_tot[t] += (label == prediction).sum().item()
                else:
                    # print(f"Shape mismatch in mp_test (cumulative) at t={t}: label {label.shape}, prediction {prediction.shape}") # 注释掉或删除
                    pass

    return new_tot # 返回每个时间步的正确预测数量


def cal_message(test_dataloader, model, net_arch, presim_len, sim_len, device):
    model = model.to(device) # 确保模型在正确的设备上
    model.eval() # 确保模型处于评估模式
    ans = []

    with torch.no_grad():
        # 这个函数的目标似乎是收集 MPLayer 的内部状态，但 MPLayer 可能已被移除或修改
        # 需要确认 MPLayer 的当前状态和作用
        has_mplayer = any(isinstance(m, MPLayer) for m in model.modules())

        if not has_mplayer:
            print("警告: 模型中未找到 MPLayer，cal_message 可能无法按预期工作。")
            return np.array([]) # 返回空数组或进行其他处理

        for img, label in tqdm(test_dataloader, desc="Calculating Message"):
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            # 这里的逻辑依赖于 set_MPLayer 和 calculate_MPLayer 函数以及 MPLayer 的实现
            # 需要确保这些函数和类仍然有效
            try:
                set_MPLayer(model, True) # 切换到 SNN 模式？
                _ = model(img) # 运行一次以触发可能的内部状态记录

                set_MPLayer(model, False) # 切换回 ANN 模式？
                _ = model(img) # 再次运行

                # 收集记录的数据
                calculate_MPLayer(model, ans) # 假设此函数修改 ans 列表

            except AttributeError as e:
                 print(f"错误: 在 cal_message 中调用 set_MPLayer 或 calculate_MPLayer 时出错: {e}")
                 print("这可能是因为模型结构已更改或相关函数/类已移除。")
                 return np.array([]) # 返回空数组

        # 重塑结果
        if ans:
            try:
                ans = np.array(ans).reshape(-1, 4)
            except ValueError as e:
                print(f"错误: 无法将收集到的数据重塑为 (-1, 4): {e}")
                print(f"收集到的数据: {ans}")
                ans = np.array([]) # 返回空数组
        else:
            ans = np.array([])

    return ans
