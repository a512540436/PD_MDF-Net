import copy
import math
import os
import random
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    mean_absolute_error, mean_squared_error, r2_score, roc_curve, auc
from torch.utils.data import TensorDataset, DataLoader
from scipy.special import softmax

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 102
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # numpy 的随机种子
random.seed(seed)


def lr_change(optimizer, epoch_num, max_lr):
    if epoch_num <= 5:
        return CustomLRScheduler(optimizer, 5, 5, 5, max_lr, max_lr, max_lr)
    else:
        warmup_epochs = int(epoch_num / 5)
        constant_epochs = int(2 * epoch_num / 5)
        decay_epochs = int(2 * epoch_num / 5)

        initial_lr = max_lr * 1e-4
        final_lr = max_lr * 1e-2
        scheduler_in = CustomLRScheduler(optimizer, warmup_epochs, constant_epochs, decay_epochs, initial_lr, max_lr,
                                         final_lr)
        return scheduler_in


class CustomLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, constant_epochs, decay_epochs, initial_lr, max_lr, final_lr):
        self.warmup_epochs = warmup_epochs
        self.constant_epochs = constant_epochs
        self.decay_epochs = decay_epochs
        self.initial_lr = initial_lr
        self.max_lr = max_lr
        self.final_lr = final_lr
        super().__init__(optimizer)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            # Gradual warmup phase
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * (epoch / self.warmup_epochs)
        elif epoch < self.warmup_epochs + self.constant_epochs:
            # Constant learning rate phase
            lr = self.max_lr
        else:
            # Gradual decay phase
            decay_factor = (self.final_lr / self.max_lr) ** (1 / self.decay_epochs)
            lr = self.max_lr * decay_factor ** (epoch - self.warmup_epochs - self.constant_epochs)
        return [lr]


def print_eval_info(p_label_in, out_dict, printing_options):
    print("-" * 40, f"{f'{p_label_in}':^20}", "-" * 40)

    if printing_options == "ALL":
        for line in out_dict.get("summary_p", []):
            print(line)

    print(f"loss:{out_dict['loss']:.9f}")

    # 判断是回归还是分类
    if "mae_s" in out_dict or "mse_s" in out_dict:  # 回归任务
        print(f"样本MAE={out_dict.get('mae_s', float('nan')):^15.9f} "
              f"样本MSE={out_dict.get('mse_s', float('nan')):^15.9f} "
              f"样本RMSE={out_dict.get('rmse_s', float('nan')):^15.9f} "
              f"样本R2={out_dict.get('r2_s', float('nan')):^15.9f} "
              f"样本MAPE={out_dict.get('mape_s', float('nan')):^15.9f}"
              f"样本WMAPE={out_dict.get('wmape_s', float('nan')):^15.9f}"
              f"样本bias={out_dict.get('bias_s', float('nan')):^15.9f} "
              f"样本max_error={out_dict.get('max_error_s', float('nan')):^15.9f}")
        print(f"人数MAE={out_dict.get('mae_p', float('nan')):^15.9f} "
              f"人数MSE={out_dict.get('mse_p', float('nan')):^15.9f} "
              f"人数RMSE={out_dict.get('rmse_p', float('nan')):^15.9f} "
              f"人数R2={out_dict.get('r2_p', float('nan')):^15.9f} "
              f"人数MAPE={out_dict.get('mape_p', float('nan')):^15.9f}"
              f"人数WMAPE={out_dict.get('wmape_p', float('nan')):^15.9f}"
              f"人数bias={out_dict.get('bias_p', float('nan')):^15.9f} "
              f"人数max_error={out_dict.get('max_error_p', float('nan')):^15.9f}")
    else:  # 分类任务
        print(f"样本准确率={out_dict['acc_s']:^15.9f} 样本精确率={out_dict['precision_s']:^15.9f} "
              f"样本召回率={out_dict['recall_s']:^15.9f} 样本F1分数={out_dict['f1_s']:^15.9f} "
              f"样本AUC={out_dict['roc_auc_s']:^15.9f}")
        print(f"人数准确率={out_dict['acc_p']:^15.9f} 人数精确率={out_dict['precision_p']:^15.9f} "
              f"人数召回率={out_dict['recall_p']:^15.9f} 人数F1分数={out_dict['f1_p']:^15.9f} "
              f"人数AUC={out_dict['roc_auc_p']:^15.9f}")

    print("-" * 102)


def check_and_update_model(p_label_in, out_dict, target_in, printing_options,
                           model_in=None, optimizer_in=None, model_save=None):
    if not target_in:
        target_max = 0
        print_eval_info(p_label_in, out_dict, printing_options)
        return target_max
    elif target_in[0] == "quiet":
        return 0

    key, value = target_in

    # --- 定义指标方向 ---
    smaller_better = {
        "loss", "mae_s", "mse_s", "rmse_s", "mape_s",
        "mae_p", "mse_p", "rmse_p", "mape_p",
        "bias_s", "bias_p", "max_error_s", "max_error_p"
    }
    larger_better = {
        "acc_s", "precision_s", "recall_s", "f1_s", "roc_auc_s",
        "acc_p", "precision_p", "recall_p", "f1_p", "roc_auc_p",
        "r2_s", "r2_p"
    }

    # --- 判断是否更新 ---
    if key in smaller_better:
        should_update = out_dict[key] <= value
    elif key in larger_better:
        should_update = out_dict[key] >= value
    elif key == "epoch":
        should_update = (p_label_in % value == 0)
    else:
        raise ValueError(f"未知的指标: {key}")

    # --- 更新逻辑 ---
    if should_update:
        print(f"由 {key} 更新模型：")
        print_eval_info(p_label_in, out_dict, printing_options)
        if model_save:
            torch.save({
                'epoch': p_label_in,
                'model_state_dict': model_in.state_dict(),
                'optimizer_state_dict': optimizer_in.state_dict(),
                'best_key': out_dict[key]
            }, model_save)
            print("{:^11}".format("Epoch " + str(p_label_in)) + ":" + model_save +
                  f" 模型已保存，最新 {key} 为: {out_dict[key]:.9f}")
        return out_dict[key] if key != "epoch" else 0
    else:
        return value


def evaluate_by_id(data_in, all_p_in):
    # 深拷贝数据
    data_in = copy.deepcopy(data_in)
    data_in += [all_p_in]  # 追加预测值

    # 加前缀构造 ID
    for i in range(len(data_in[0])):
        if data_in[1][i][7] == 1:
            data_in[0][i] = "1_" + data_in[0][i]
        else:
            data_in[0][i] = "0_" + data_in[0][i]
        data_in[0][i] = f"{torch.argmax(data_in[3][i])}_" + data_in[0][i]

    id_list = data_in[0]
    true_labels = data_in[3]
    pred_labels = data_in[-1]

    # 聚合
    pred_dict = defaultdict(list)
    true_dict = defaultdict(list)
    count_dict = defaultdict(int)

    for i, id_str in enumerate(id_list):
        pred_dict[id_str].append(pred_labels[i])
        true_dict[id_str].append(true_labels[i])
        count_dict[id_str] += 1

    # 打印每个 ID 信息
    summary_lines = []
    header = f"{'ID':<16}{'Count':^12}{'TLabel':^12}{'PLabel':^12}{'Expected':^12}{'PredProb':^60}" \
             f"{'Correct':^8}"
    summary_lines.append(header)

    all_pred_avg = []
    all_true_avg = []

    for id_str in sorted(pred_dict.keys()):
        pred_tensor = torch.stack(pred_dict[id_str])  # shape: [N, num_classes]
        true_tensor = torch.stack(true_dict[id_str])  # shape: [N, num_classes]

        pred_avg = pred_tensor.mean(dim=0)
        true_avg = true_tensor.mean(dim=0)

        all_pred_avg.append(pred_avg)
        all_true_avg.append(true_avg)

        pred_classes = torch.argmax(pred_tensor, dim=1)  # 每个样本的类别索引
        pred_class = torch.mode(pred_classes).values.item()  # 多数投票类别
        true_class = true_avg.argmax().item()
        is_correct = pred_class == true_class

        num_classes = pred_avg.size(0)
        class_indices = torch.arange(num_classes, dtype=pred_avg.dtype, device=pred_avg.device)
        if len(class_indices) > 4:
            class_indices[-1] = 0
        expected_class = (class_indices * pred_avg).sum().item()

        # ⏬ 构造行文本
        pred_values_str = " ".join([f"{v:.5f}" for v in pred_avg.tolist()])
        line = f"{id_str[4:]:<16}{count_dict[id_str]:^12}{true_class:^12}{pred_class:^12}" \
               f"{expected_class:^12.4f}{pred_values_str:^60}{str(is_correct):^8}"
        summary_lines.append(line)

    acc_in, precision_in, recall_in, f1_in, roc_auc_in, fpr_in, tpr_in = \
        get_sta_info(torch.stack(all_pred_avg), torch.stack(all_true_avg))

    return acc_in, precision_in, recall_in, f1_in, roc_auc_in, fpr_in, tpr_in, summary_lines


def evaluate_by_id_liner(data_in, all_p_in):
    # 深拷贝数据
    data_in = copy.deepcopy(data_in)
    data_in += [all_p_in]  # 追加预测值

    # 加前缀构造 ID
    for i in range(len(data_in[0])):
        if data_in[1][i][7] == 1:
            data_in[0][i] = "1_" + data_in[0][i]
        else:
            data_in[0][i] = "0_" + data_in[0][i]
        data_in[0][i] = f"{torch.argmax(data_in[3][i])}_" + data_in[0][i]

    id_list = data_in[0]
    true_labels = data_in[3]
    pred_labels = data_in[-1]

    # 聚合
    pred_dict = defaultdict(list)
    true_dict = defaultdict(list)
    count_dict = defaultdict(int)

    for i, id_str in enumerate(id_list):
        pred_dict[id_str].append(pred_labels[i])
        true_dict[id_str].append(true_labels[i])
        count_dict[id_str] += 1


    # 打印每个 ID 信息
    summary_lines = []
    header = f"{'ID':<16}{'Count':^12}{'Ture_num':^12}{'pred_num':^12}{'diff':^12}{'diff%':^12}"
    summary_lines.append(header)

    all_pred_avg = []
    all_true_avg = []

    for id_str in sorted(pred_dict.keys()):
        pred_tensor = torch.stack(pred_dict[id_str]).cpu().reshape(-1)
        true_tensor = torch.stack(true_dict[id_str]).cpu().reshape(-1)

        pred_avg = pred_tensor.mean(dim=0)
        true_avg = true_tensor.mean(dim=0)

        all_pred_avg.append(pred_avg)
        all_true_avg.append(true_avg)

        diff = (pred_avg - true_avg).item()
        if true_avg.item() != 0:
            diff_per = (pred_avg - true_avg).item() / true_avg.item() * 100
            line = f"{id_str[4:]:<16}{count_dict[id_str]:^12}" \
                   f"{true_avg.item():^12.4f}{pred_avg.item():^12.4f}{diff:^12.4f}{diff_per:^12.4f}%"
        else:
            line = f"{id_str[4:]:<16}{count_dict[id_str]:^12}" \
                   f"{true_avg.item():^12.4f}{pred_avg.item():^12.4f}{diff:^12.4f}{'-':^12}%"

        summary_lines.append(line)

    mae, mse, rmse, mape, wmape, r2, bias, max_error = get_sta_info_liner(torch.stack(all_pred_avg),
                                                                          torch.stack(all_true_avg))

    return mae, mse, rmse, mape, wmape, r2, bias, max_error, summary_lines


def get_sta_info(p_in, t_in):
    yp_in = p_in.detach().cpu().numpy()  # predicted probabilities, shape (N, C)
    yt_in = t_in.detach().cpu().numpy()
    input_in = yp_in.argmax(axis=1)  # predicted class indices
    label_in = yt_in.argmax(axis=1)  # true class indices
    if yp_in.shape[1] == 2:  # binary classification
        acc = accuracy_score(label_in, input_in)
        precision = precision_score(label_in, input_in, average='binary', zero_division=0, pos_label=1)
        recall = recall_score(label_in, input_in, average='binary', zero_division=0, pos_label=1)
        f1 = f1_score(label_in, input_in, average='binary', zero_division=0, pos_label=1)
        fpr, tpr, thresholds = roc_curve(yt_in[:, 1], yp_in[:, 1])  # Assuming labels are one-hot encoded
        roc_auc = auc(fpr, tpr)
        fpr = {}
        tpr = {}
        for i, c in enumerate([0, 1]):
            fpr[c], tpr[c], _ = roc_curve(yt_in[:, i], yp_in[:, i])
    else:  # multi-class classification
        aver_opt = "weighted"  # 'macro'  "weighted"
        present_classes_in = np.where(yt_in.sum(axis=0) > 0)[0]  # remove classes not present
        # present_classes_in = [0, 1, 2]
        ytp_in = yt_in[:, present_classes_in]
        ypp_in = softmax(yp_in[:, present_classes_in], axis=1)  # ypp_in = yp_in[:, present_classes_in]
        acc = accuracy_score(label_in, input_in)
        precision = precision_score(label_in, input_in, average=aver_opt, zero_division=0, labels=present_classes_in)
        recall = recall_score(label_in, input_in, average=aver_opt, zero_division=0, labels=present_classes_in)
        f1 = f1_score(label_in, input_in, average=aver_opt, zero_division=0, labels=present_classes_in)
        roc_auc = roc_auc_score(ytp_in, ypp_in, multi_class='ovr', average=aver_opt)
        # ROC curve data for each class
        fpr = {}
        tpr = {}
        for i, c in enumerate(present_classes_in):
            fpr[c], tpr[c], _ = roc_curve(ytp_in[:, i], ypp_in[:, i])
    return acc, precision, recall, f1, roc_auc, fpr, tpr


def get_sta_info_liner(p_in, t_in):
    yp = p_in.detach().cpu().numpy()
    yt = t_in.detach().cpu().numpy()

    # 1. MAE
    mae = mean_absolute_error(yt, yp)
    # 2. MSE
    mse = mean_squared_error(yt, yp)
    # 3. RMSE
    rmse = np.sqrt(mse)
    # 4. MAPE (平均百分比误差)
    mask = yt != 0
    if np.any(mask):
        mape = np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100
    else:
        mape = np.nan  # 如果全为0，则返回 NaN
    # 5. R^2 Score
    r2 = r2_score(yt, yp)
    # 6. 平均误差（Bias）
    bias = np.mean(yp - yt)
    # 7. 最大误差
    max_error = np.max(np.abs(yp - yt))
    # 8. WMAPE (加权平均绝对百分比误差 / 相对绝对误差)

    denom = np.sum(np.abs(yt))
    if denom > 0:
        wmape = np.sum(np.abs(yt - yp)) / denom * 100
    else:
        wmape = np.nan

    return mae, mse, rmse, mape, wmape, r2, bias, max_error


def model_eval(model_in, data_in, dataloader_in, criterion_in, p_label_in, target_in=None,
               printing_options=None, optimizer_in=None, model_save=None):
    model_in.eval()  # 评估模式
    total_loss = 0.0
    all_t = torch.empty(0, ).to(device)
    all_p = torch.empty(0, ).to(device)

    with torch.no_grad():  # 禁用梯度计算
        for x1, x2, yt in dataloader_in:
            yp = model_in(x2)
            loss_in = criterion_in(yp, yt)

            total_loss += loss_in.item() * len(x1)
            all_p = torch.cat([all_p, yp], dim=0)
            all_t = torch.cat([all_t, yt], dim=0)

    avg_loss = total_loss / len(data_in[0])
    if all_p.shape[1] > 1:
        all_p = F.softmax(all_p, dim=1)
        acc_s, precision_s, recall_s, f1_s, roc_auc_s, fpr_s, tpr_s = get_sta_info(all_p, all_t)
        acc_p, precision_p, recall_p, f1_p, roc_auc_p, fpr_p, tpr_p, summary_p = evaluate_by_id(data_in, all_p)

        out_dict = {
            "loss": avg_loss,
            "acc_s": acc_s,
            "precision_s": precision_s,
            "recall_s": recall_s,
            "f1_s": f1_s,
            "roc_auc_s": roc_auc_s,
            "fpr_s": fpr_s,
            "tpr_s": tpr_s,
            "acc_p": acc_p,
            "precision_p": precision_p,
            "recall_p": recall_p,
            "f1_p": f1_p,
            "roc_auc_p": roc_auc_p,
            "fpr_p": fpr_p,
            "tpr_p": tpr_p,
            "summary_p": summary_p
        }

        target_max = check_and_update_model(p_label_in, out_dict, target_in, printing_options, model_in, optimizer_in,
                                            model_save)
        return out_dict, target_max
    else:
        all_p = torch.clamp(all_p, 0, torch.max(all_t).item())
        mae_s, mse_s, rmse_s, mape_s, wmape_s, r2_s, bias_s, max_error_s = get_sta_info_liner(all_p, all_t)
        mae_p, mse_p, rmse_p, mape_p, wmape_p, r2_p, bias_p, max_error_p, summary_p = \
            evaluate_by_id_liner(data_in, all_p)

        out_dict = {
            "loss": avg_loss,
            "mae_s": mae_s,
            "mse_s": mse_s,
            "rmse_s": rmse_s,
            "mape_s": mape_s,
            "wmape_s": wmape_s,
            "r2_s": r2_s,
            "bias_s": bias_s,
            "max_error_s": max_error_s,
            "mae_p": mae_p,
            "mse_p": mse_p,
            "rmse_p": rmse_p,
            "mape_p": mape_p,
            "wmape_p": wmape_p,
            "r2_p": r2_p,
            "bias_p": bias_p,
            "max_error_p": max_error_p,
            "summary_p": summary_p
        }

        target_min = check_and_update_model(p_label_in, out_dict, target_in, printing_options, model_in, optimizer_in,
                                            model_save)
        return out_dict, target_min


def get_date_tensor(pkl_data_in, label_num):
    names_list = []
    info_list = []
    data_list = []
    label_list = []

    for item in pkl_data_in:
        person_info, class_info, data = item
        name = person_info[0]
        numeric_info = []

        for val in person_info[1:]:
            if isinstance(val, (int, float)):
                numeric_info.append(val)
            elif isinstance(val, list):
                numeric_info.extend(val)
        names_list.append(name)
        info_list.append(numeric_info)
        data_list.append(data)
        label_list.append(class_info[label_num])

    info_tensor = torch.tensor(info_list, dtype=torch.float32)
    data_tensor = torch.tensor(np.array(data_list), dtype=torch.float32)
    label_tensor = torch.tensor(label_list, dtype=torch.float32)

    data_out = [names_list, info_tensor, data_tensor, label_tensor]
    return data_out


def get_date_tensor_age(pkl_data_in, label_num, age_d, s_N):
    names_list = []
    info_list = []
    data_list = []
    label_list = []

    for item in pkl_data_in:
        if age_d[0] <= item[0][3] <= age_d[1]:
            person_info, class_info, data = item
            name = person_info[0]
            numeric_info = []

            for val in person_info[1:]:
                if isinstance(val, (int, float)):
                    numeric_info.append(val)
                elif isinstance(val, list):
                    numeric_info.extend(val)
            names_list.append(name)
            info_list.append(numeric_info)
            data_list.append(data)
            label_list.append(class_info[label_num])

    N = len(names_list)
    sample_size = s_N

    # 生成随机索引（不重复）

    state = random.getstate()
    # 局部使用固定种子
    random.seed(2)
    indices = random.sample(range(N), sample_size)
    # 恢复全局 random 状态
    random.setstate(state)

    # 根据索引抽样
    names_sample = [names_list[i] for i in indices]
    info_sample = [info_list[i] for i in indices]
    data_sample = [data_list[i] for i in indices]
    label_sample = [label_list[i] for i in indices]

    info_tensor = torch.tensor(info_sample, dtype=torch.float32)
    data_tensor = torch.tensor(np.array(data_sample), dtype=torch.float32)
    label_tensor = torch.tensor(label_sample, dtype=torch.float32)

    data_out = [names_sample, info_tensor, data_tensor, label_tensor]
    return data_out


def print_class_distribution(label_tensor, title=""):
    class_indices = label_tensor.argmax(dim=1)
    num_classes = label_tensor.size(1)
    class_counts = torch.bincount(class_indices, minlength=num_classes)
    total = class_counts.sum().item()

    print(f"\n📊 {title} class distribution (Total: {total} samples):")
    for i, count in enumerate(class_counts.tolist()):
        ratio = count / total * 100
        print(f"  Class {i}: {count:>4} samples ({ratio:>5.2f}%)")


def get_dataloader_all(data_in, bs_size, shuffle=True):
    x_info = data_in[1].to(device)
    x_data = data_in[2].to(device)
    y_t = data_in[3].to(device)

    # 自动检测回归任务：y_t 是一维，x_data 也可能是二维
    # 如果 y_t 是一维，则认为是回归，把它变为 (N,1)
    if y_t.dim() == 1:
        y_t = y_t.unsqueeze(1)  # (N,) -> (N,1)

    ds_out = TensorDataset(x_info, x_data, y_t)
    dl_out = DataLoader(ds_out, batch_size=bs_size, shuffle=shuffle, drop_last=shuffle)
    return dl_out


def init_weights_(
        module,
        n_layer,
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1):
    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def load_checkpoint(model, optimizer, filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        """
        :param gamma: focusing parameter
        :param weight: manual rescaling weight given to each class
        :param reduction: 'none' | 'mean' | 'sum'
        """
        super().__init__()
        self.gamma = gamma
        self.weight = weight  # 类别权重
        self.reduction = reduction

    def forward(self, input_in, target):
        if target.ndim == 2:
            target = target.argmax(dim=1)  # 自动处理 one-hot

        log_prob = F.log_softmax(input_in, dim=1)
        prob = torch.exp(log_prob)
        focal_weight = (1 - prob).pow(self.gamma)

        loss = F.nll_loss(focal_weight * log_prob, target, weight=self.weight, reduction=self.reduction)
        return loss


class ZeroTolerantMSELoss(nn.Module):
    @staticmethod
    def forward(y_pred, y_true):
        y_true = y_true.to(dtype=y_pred.dtype)

        mask_zero = (y_true == 0).float()  # float mask
        mask_nonzero = 1.0 - mask_zero  # 非0 mask

        # 直接用 mask 相乘，不用索引赋值
        loss = mask_nonzero * (y_pred - y_true) ** 2 + mask_zero * F.relu(y_pred) ** 2
        return loss.mean()


class BoundedMSELoss(nn.Module):
    def __init__(self, min_val=0, max_val=68, boundary_weight=2.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.boundary_weight = boundary_weight

    def forward(self, y_pred, y_true):
        # 基础MSE损失
        base_loss = F.mse_loss(y_pred, y_true, reduction='none')

        # 边界惩罚项
        lower_bound_penalty = F.relu(self.min_val - y_pred) ** 2
        upper_bound_penalty = F.relu(y_pred - self.max_val) ** 2

        # 组合损失
        loss = base_loss + self.boundary_weight * (lower_bound_penalty + upper_bound_penalty)
        return loss.mean()


class ScaledSigmoid(nn.Module):
    def __init__(self, min_val, max_val):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return self.min_val + (self.max_val - self.min_val) * torch.sigmoid(x)


def save_fold_result_to_excel(out_data_tr, out_data_te, fold_idx, metric_path, curve_path):
    os.makedirs(os.path.dirname(metric_path), exist_ok=True)

    # ========== 1. 保存 scalar 指标（如 acc、f1、loss）到 output_metrics.xlsx ==========
    exclude_keys = ["fpr_s", "tpr_s", "fpr_p", "tpr_p", "summary_p"]
    keys_to_save = [k for k in out_data_tr.keys() if k not in exclude_keys]

    row = {"fold": fold_idx}
    for k in keys_to_save:
        row[f"tr_{k}"] = out_data_tr[k]
    for k in keys_to_save:
        row[f"te_{k}"] = out_data_te[k]

    new_row_df = pd.DataFrame([row])

    if os.path.exists(metric_path):
        df_old = pd.read_excel(metric_path)
        df = pd.concat([df_old, new_row_df], ignore_index=True)
    else:
        df = new_row_df

    df.to_excel(metric_path, index=False)

    # ========== 2. 保存 fpr/tpr/summary 到 result_fpr_tpr_summary_fold{n}.xlsx ==========
    curve_path = curve_path.format(fold_idx)
    with pd.ExcelWriter(curve_path) as writer:
        # summary_p
        pd.DataFrame({"Summary": out_data_tr["summary_p"]}).to_excel(writer, sheet_name="tr_Summary", index=False)
        pd.DataFrame({"Summary": out_data_te["summary_p"]}).to_excel(writer, sheet_name="te_Summary", index=False)

        # fpr/tpr for training
        if "fpr_s" in out_data_tr and out_data_tr["fpr_s"]:
            for cls in sorted(out_data_tr["fpr_s"].keys()):
                df = pd.DataFrame({
                    "fpr": out_data_tr["fpr_s"][cls],
                    "tpr": out_data_tr["tpr_s"][cls]
                })
                df.to_excel(writer, sheet_name=f"tr_Class_{cls}", index=False)

            # fpr/tpr for testing
            for cls in sorted(out_data_te["fpr_s"].keys()):
                df = pd.DataFrame({
                    "fpr": out_data_te["fpr_s"][cls],
                    "tpr": out_data_te["tpr_s"][cls]
                })
                df.to_excel(writer, sheet_name=f"te_Class_{cls}", index=False)

    print(f"✅ Fold {fold_idx} 指标保存成功：{curve_path}")


def process_binary(folder, out_name):
    """处理二分类版本"""
    all_data = []

    for i in range(1, 6):
        file_path = os.path.join(folder, f"result_fpr_tpr_summary_fold{i}.xlsx")
        df = pd.read_excel(file_path, sheet_name=1, header=None)
        df = df.iloc[2:, :]

        for val in df.iloc[:, 0].dropna():
            parts = [p for p in str(val).split(" ") if p.strip() != ""]
            if len(parts) >= 6:
                row = {
                    "ID": parts[0],
                    "Count": parts[1],
                    "Ture_num": parts[2],
                    "pred_num": parts[3],
                    "diff": parts[4],
                    "diff%": "".join(parts[5:])
                }
                all_data.append(row)

    result = pd.DataFrame(all_data)

    # 转数值
    for col in ["Ture_num", "pred_num", "diff", "diff%"]:
        result[col] = (
            result[col].astype(str)
            .str.replace("%", "", regex=False)
            .replace("-", np.nan)
        )
        result[col] = pd.to_numeric(result[col], errors="coerce")

    result["Count"] = pd.to_numeric(result["Count"], errors="coerce", downcast="integer")

    # diff_abs
    insert_at = result.columns.get_loc("diff") + 1
    result.insert(insert_at, "diff_abs", result["diff"].abs())

    out_path = os.path.join(folder, out_name)
    result.to_excel(out_path, index=False)
    return result


def process_multiclass(folder, out_name):
    """处理多分类版本"""
    all_data = []

    for i in range(1, 6):
        file_path = os.path.join(folder, f"result_fpr_tpr_summary_fold{i}.xlsx")
        df = pd.read_excel(file_path, sheet_name=1, header=None)
        df = df.iloc[2:, :]

        for val in df.iloc[:, 0].dropna():
            parts = [p for p in str(val).split(" ") if p.strip() != ""]
            if len(parts) >= 6:
                ID, Count = parts[0], parts[1]
                TLabel, PLabel, Expected = parts[2], parts[3], parts[4]
                Correct = parts[-1]
                PredProb = " ".join(parts[6:-1])

                row = {
                    "ID": ID,
                    "Count": Count,
                    "TLabel": TLabel,
                    "PLabel": PLabel,
                    "Expected": Expected,
                    "PredProb": PredProb,
                    "Correct": Correct,
                }
                all_data.append(row)

    result = pd.DataFrame(all_data)

    for col in ["TLabel", "PLabel", "Expected"]:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    result["Count"] = pd.to_numeric(result["Count"], errors="coerce", downcast="integer")

    out_path = os.path.join(folder, out_name)
    result.to_excel(out_path, index=False)
    return result


def collect_te_summary(folder, out_name):
    """自动识别二分类/多分类 te_summary"""
    # 先读第一个文件前几行
    test_file = os.path.join(folder, "result_fpr_tpr_summary_fold1.xlsx")
    df = pd.read_excel(test_file, sheet_name=1, header=None)
    sample_val = str(df.iloc[2, 0])  # 第3行第1列的示例
    parts = [p for p in sample_val.split(" ") if p.strip() != ""]

    # 判断
    if len(parts) >= 4 and parts[3].isdigit():
        # 第4列是数字 → 多分类 (TLabel)
        return process_multiclass(folder, out_name)
    elif len(parts) >= 4 and not parts[3].isdigit():
        # 第4列不是数字 → 二分类 (Ture_num)
        return process_binary(folder, out_name)
    else:
        raise ValueError("无法识别该 te_summary 格式")


def insert_blank_cols_for_auc(df):
    cols_to_insert_after = ["tr_roc_auc_s", "tr_roc_auc_p", "te_roc_auc_s", "te_roc_auc_p"]

    # 由于插入列会改变列索引，先记录所有位置
    insert_positions = []
    for col in cols_to_insert_after:
        if col in df.columns:
            insert_positions.append((df.columns.get_loc(col) + 1, f"空列_{col}"))

    # 按位置从小到大排序，依次插入空列
    insert_positions.sort()
    offset = 0
    for pos, new_col_name in insert_positions:
        df.insert(pos + offset, new_col_name, "")
        offset += 1

    return df
