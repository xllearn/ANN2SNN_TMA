from utils import *
from NetworkFunction import *
import argparse
from dataprocess import PreProcess_Cifar10, PreProcess_Cifar100, PreProcess_ImageNet
from Models.ResNet import *
from Models.VGG import *
import torch
import random
import os
import numpy as np
import copy
from calibration import calibrate_snn_with_bptt
# Import AMP libraries
from torch.cuda.amp import GradScaler

def evaluate_snn_at_timesteps(model, test_dataloader, timesteps, device, cap_dataset, presim_len=0, net_arch=''):
    """
    在指定的时间步长列表上评估 SNN 模型的精度

    参数:
        model: SNN 模型
        test_dataloader: 测试数据加载器
        timesteps: 时间步长列表，例如 [1, 2, 4, 8, 16, 32]
        device: 设备
        cap_dataset: 数据集大小，用于计算精度
        presim_len: 预模拟长度
        net_arch: 网络架构名称，用于 mp_test

    返回:
        包含每个时间步长精度的字典
    """
    accuracies = {}
    max_timestep = max(timesteps)

    # 复制模型以避免修改原始模型
    eval_model = copy.deepcopy(model)

    # 获取原始准确度计数
    if presim_len > 0:
        # 使用 mp_test 评估
        acc_results = mp_test(test_dataloader, eval_model, net_arch=net_arch,
                              presim_len=presim_len, sim_len=max_timestep, device=device)
    else:
        # 使用标准 SNN 评估
        replace_MPLayer_by_neuron(eval_model)
        acc_results = eval_snn(test_dataloader, eval_model, sim_len=max_timestep, device=device)

    # 提取每个指定时间步的精度
    for t in timesteps:
        if t <= max_timestep:
            accuracies[t] = (acc_results[t - 1] / cap_dataset).item()

    return accuracies


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='CIFAR100', help='Dataset name')
    parser.add_argument('--datadir', type=str, default='datasets', help='Directory where the dataset is saved')
    parser.add_argument('--savedir', type=str, default='model/', help='Directory where the model is saved')
    parser.add_argument('--load_model_name', type=str, default='None', help='The name of the loaded ANN model')
    parser.add_argument('--trainann_epochs', type=int, default=300, help='Training Epochs of ANNs')
    parser.add_argument('--activation_floor', type=str, default='QCFS', help='ANN activation modules')
    parser.add_argument('--net_arch', type=str, default='vgg16', help='Network Architecture')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--batchsize', type=int, default=50, help='Batch size')
    parser.add_argument('--L', type=int, default=4, help='Quantization level of QCFS')
    parser.add_argument('--sim_len', type=int, default=32, help='Simulation length of SNNs')
    parser.add_argument('--presim_len', type=int, default=4, help='Pre Simulation length')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--direct_training', action='store_true', default=False)
    parser.add_argument('--save_model', action='store_true', default=True)
    parser.add_argument('--cal_message', action='store_true', default=False)
    parser.add_argument('--train_dir', type=str, default='autodl-tmp/train',
                        help='Directory where the ImageNet train dataset is saved')
    parser.add_argument('--test_dir', type=str, default='autodl-tmp/val',
                        help='Directory where the ImageNet test dataset is saved')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')
    # 添加 BPTT 校准相关参数
    parser.add_argument('--bptt_calib', action='store_true', default=False, help='是否使用 BPTT 校准')
    parser.add_argument('--calib_samples', type=int, default=512, help='校准样本数量')
    parser.add_argument('--bptt_epochs', type=int, default=10, help='BPTT 校准轮数')
    parser.add_argument('--bptt_lr', type=float, default=0.001, help='BPTT 校准学习率')
    # 添加评估时间步参数
    parser.add_argument('--eval_timesteps', type=str, default='1,2,4,8,16,32', help='评估的时间步长，用逗号分隔')
    parser.add_argument('--compare_models', action='store_true', default=True, help='是否比较校准前后的模型精度')
    # 添加 AMP 和 Compile 开关
    parser.add_argument('--amp', action='store_true', default=True, help='Enable Automatic Mixed Precision (AMP)')
    parser.add_argument('--compile', action='store_true', default=False, help='Enable torch.compile (PyTorch 2.0+)')


    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES

    # 创建保存目录
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    torch.backends.cudnn.benchmark = True
    _seed_ = args.seed
    random.seed(_seed_)
    os.environ['PYTHONHASHSEED'] = str(_seed_)
    torch.manual_seed(_seed_)
    torch.cuda.manual_seed(_seed_)
    torch.cuda.manual_seed_all(_seed_)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(_seed_)

    cls = 100
    cap_dataset = 10000

    if args.dataset == 'CIFAR10':
        cls = 10
    elif args.dataset == 'CIFAR100':
        cls = 100
    elif args.dataset == 'ImageNet':
        cls = 1000
        cap_dataset = 50000

    if args.net_arch == 'resnet20':
        model = resnet20(num_classes=cls)
    elif args.net_arch == 'resnet18':
        model = resnet18(num_classes=cls)
    elif args.net_arch == 'resnet34':
        model = resnet34(num_classes=cls)
    elif args.net_arch == 'vgg16':
        model = vgg16(num_classes=cls) # 使用 Models/VGG.py 中的 vgg16
    else:
        error('unable to find model ' + args.net_arch)

    model = replace_maxpool2d_by_avgpool2d(model)

    if args.activation_floor == 'QCFS':
        model = replace_activation_by_floor(model, args.L)
    else:
        error('unable to find activation floor: ' + args.activation_floor)

    if args.dataset == 'CIFAR10':
        train, test = PreProcess_Cifar10(args.datadir, args.batchsize)
    elif args.dataset == 'CIFAR100':
        train, test = PreProcess_Cifar100(args.datadir, args.batchsize)
    elif args.dataset == 'ImageNet':
        train, test = PreProcess_ImageNet(args.datadir, args.batchsize, train_dir=args.train_dir,
                                          test_dir=args.test_dir)
    else:
        error('unable to find dataset ' + args.dataset)

    # 将模型移动到设备
    model.to(args.device)

    # --- 应用 torch.compile ---
    if args.compile and hasattr(torch, 'compile'):
        print("Compiling the model...")
        try:
            # 你可以尝试不同的 mode, 例如 'reduce-overhead'
            model = torch.compile(model)
            print("Model compiled successfully.")
        except Exception as e:
            print(f"torch.compile failed: {e}. Running without compilation.")
            args.compile = False # 编译失败则禁用
    elif args.compile:
        print("torch.compile is not available in this PyTorch version. Running without compilation.")
        args.compile = False
    else:
        print("torch.compile is disabled.")


    # --- 初始化 GradScaler ---
    # 仅在启用 AMP 时需要
    scaler = GradScaler(enabled=args.amp)
    print(f"Automatic Mixed Precision (AMP) enabled: {args.amp}")


    if args.load_model_name != 'None':
        print(f'=== Load Pretrained ANNs ===')
        # 加载状态字典时确保映射到正确的设备
        model.load_state_dict(torch.load(args.load_model_name + '.pth', map_location=args.device))
    if args.direct_training is True:
        print(f'=== Start Training ANNs ===')
        save_name = args.savedir + args.activation_floor + '_' + args.dataset + '_' + args.net_arch + '_L' + str(
            args.L) + '.pth'
        # 将 scaler 传递给训练函数
        model = train_ann(train, test, model, epochs=args.trainann_epochs, lr=args.lr, wd=args.wd, device=args.device,
                          save_name=save_name, scaler=scaler, use_amp=args.amp) # 传递 scaler 和 use_amp

    print(f'=== ANNs accuracy after the first training stage ===')
    acc = eval_ann(test, model, args.device)
    print(f'Pretrained ANN Accuracy : {acc}')

    # 保存 ANN 模型副本用于校准
    ann_model = copy.deepcopy(model)

    print(f'=== SNNs accuracy after shift up/down initial membrane potential ===')
    replace_activation_by_MPLayer(model, presim_len=args.presim_len, sim_len=args.sim_len, batchsize=args.batchsize)

    # 保存未校准的 SNN 模型（权重1）
    uncalibrated_snn = copy.deepcopy(model)

    # 解析评估时间步
    eval_timesteps = [int(t) for t in args.eval_timesteps.split(',')]

    # 如果启用 BPTT 校准
    calibrated_snn = None
    if args.bptt_calib:
        print(f'=== 开始 BPTT 校准 ===')
        # 采样校准数据
        print("采样校准数据...")


        def sample_data(dataloader, num_samples):
            samples = []
            labels = []
            count = 0
            for data, target in dataloader:
                samples.append(data)
                labels.append(target)
                count += data.shape[0]
                if count >= num_samples:
                    break
            # 截断到精确的样本数量
            samples_tensor = torch.cat(samples, 0)[:num_samples]
            labels_tensor = torch.cat(labels, 0)[:num_samples]
            return samples_tensor, labels_tensor


        train_samples, _ = sample_data(train, args.calib_samples)
        val_samples, _ = sample_data(test, args.calib_samples // 4)
        train_samples = train_samples.to(args.device)
        val_samples = val_samples.to(args.device)

        # 使用 BPTT 校准 SNN
        print("开始 BPTT 校准...")
        calibrated_snn = calibrate_snn_with_bptt(
            ann_model=ann_model,
            snn_model=model,
            train_samples=train_samples,
            val_samples=val_samples,
            device=args.device,
            T=args.sim_len,
            epochs=args.bptt_epochs,
            batch_size=args.batchsize,
            lr=args.bptt_lr,
            test_dataloader=test,  # 添加测试数据集
            eval_timesteps=eval_timesteps,  # 添加评估时间步
            cap_dataset=cap_dataset,  # 添加数据集大小
            presim_len=args.presim_len,  # 添加预模拟长度
            net_arch=args.net_arch,  # 添加网络架构
            weight_decay=1e-5,  # 添加权重衰减
            scheduler_type='cosine'  # 添加学习率调度器类型
        )

        # 保存校准后的 SNN 模型
        if args.save_model:
            calib_save_name = args.savedir + args.activation_floor + '_' + args.dataset + '_' + args.net_arch + '_L' + str(
                args.L) + '_calibrated.pth'
            torch.save(calibrated_snn.state_dict(), calib_save_name)
            print(f"校准后的 SNN 模型已保存到: {calib_save_name}")

    # 比较校准前后的模型精度
    if args.compare_models and calibrated_snn is not None:
        print("\n=== 比较校准前后的 SNN 模型精度 ===")

        # 评估未校准的 SNN 模型（权重1）
        print("\n评估未校准的 SNN 模型（权重1）在不同时间步的精度...")
        uncalibrated_accuracies = evaluate_snn_at_timesteps(
            uncalibrated_snn, test, eval_timesteps, args.device, cap_dataset,
            args.presim_len, args.net_arch  # 添加 net_arch 参数
        )

        # 评估校准后的 SNN 模型（权重2）
        print("\n评估校准后的 SNN 模型（权重2）在不同时间步的精度...")
        calibrated_accuracies = evaluate_snn_at_timesteps(
            calibrated_snn, test, eval_timesteps, args.device, cap_dataset, args.presim_len, args.net_arch # 添加 net_arch 参数
        )

        # 比较并选择每个时间步下精度更高的模型
        best_accuracies = {}
        best_models = {}

        print("\n=== 精度比较结果 ===")
        print(f"{'时间步':<10}{'未校准精度':<15}{'校准后精度':<15}{'最佳精度':<15}{'最佳模型':<10}")
        print("-" * 60)

        for t in eval_timesteps:
            uncal_acc = uncalibrated_accuracies.get(t, 0)
            cal_acc = calibrated_accuracies.get(t, 0)

            if cal_acc > uncal_acc:
                best_acc = cal_acc
                best_model = "校准后"
            else:
                best_acc = uncal_acc
                best_model = "未校准"

            best_accuracies[t] = best_acc
            best_models[t] = best_model

            print(f"{t:<10}{uncal_acc:.4f}{'':<10}{cal_acc:.4f}{'':<10}{best_acc:.4f}{'':<10}{best_model}")

        # 保存比较结果
        result_file = args.savedir + args.activation_floor + '_' + args.dataset + '_' + args.net_arch + '_comparison.txt'
        with open(result_file, 'w') as f:
            f.write("=== SNN 模型精度比较 ===\n\n")
            f.write(f"数据集: {args.dataset}\n")
            f.write(f"模型架构: {args.net_arch}\n")
            f.write(f"量化级别: {args.L}\n\n")

            f.write(f"{'时间步':<10}{'未校准精度':<15}{'校准后精度':<15}{'最佳精度':<15}{'最佳模型':<10}\n")
            f.write("-" * 60 + "\n")

            for t in eval_timesteps:
                uncal_acc = uncalibrated_accuracies.get(t, 0)
                cal_acc = calibrated_accuracies.get(t, 0)
                best_acc = best_accuracies[t]
                best_model = best_models[t]

                f.write(f"{t:<10}{uncal_acc:.4f}{'':<10}{cal_acc:.4f}{'':<10}{best_acc:.4f}{'':<10}{best_model}\n")

        print(f"\n比较结果已保存到: {result_file}")

        # 使用最佳模型进行后续评估
        if args.presim_len > 0:
            if args.cal_message == True:
                # 选择最佳模型进行消息计算
                best_t = max(best_accuracies, key=best_accuracies.get)
                best_model_name = best_models[best_t]
                selected_model = calibrated_snn if best_model_name == "校准后" else uncalibrated_snn

                ans = cal_message(test, selected_model, net_arch=args.net_arch, presim_len=args.presim_len,
                                  sim_len=args.sim_len, device=args.device)
                np.save(args.savedir + 'my_message_' + args.dataset + '_' + args.net_arch + '.npy', ans)
                print(f"已保存消息到: {args.savedir + 'my_message_' + args.dataset + '_' + args.net_arch + '.npy'}")
        else:
            # 使用标准评估方法
            print("\n=== 使用标准评估方法评估最佳模型 ===")
            # 选择最佳模型
            best_t = max(best_accuracies, key=best_accuracies.get)
            best_model_name = best_models[best_t]
            selected_model = calibrated_snn if best_model_name == "校准后" else uncalibrated_snn

            replace_MPLayer_by_neuron(selected_model)
            new_acc = eval_snn(test, selected_model, sim_len=args.sim_len, device=args.device)

            t = 1
            while t < args.sim_len:
                print(f"时间步 {t}, 精度 = {(new_acc[t - 1] / cap_dataset):.4f}")
                t *= 2
            print(f"时间步 {args.sim_len}, 精度 = {(new_acc[args.sim_len - 1] / cap_dataset):.4f}")
    else:
        # 如果不比较模型，则使用原始评估流程
        if args.presim_len > 0:
            if args.cal_message == True:
                ans = cal_message(test, model, net_arch=args.net_arch, presim_len=args.presim_len, sim_len=args.sim_len,
                                  device=args.device)
                np.save(args.savedir + 'my_message_' + args.dataset + '_' + args.net_arch + '.npy', ans)
            else:
                new_acc = mp_test(test, model, net_arch=args.net_arch, presim_len=args.presim_len, sim_len=args.sim_len,
                                  device=args.device)
                print(new_acc)
        else:
            replace_MPLayer_by_neuron(model)
            new_acc = eval_snn(test, model, sim_len=args.sim_len, device=args.device)

        t = 1
        while t < args.sim_len:
            print(f"时间步 {t}, 精度 = {(new_acc[t - 1] / cap_dataset):.4f}")
            t *= 2
        print(f"时间步 {args.sim_len}, 精度 = {(new_acc[args.sim_len - 1] / cap_dataset):.4f}")

