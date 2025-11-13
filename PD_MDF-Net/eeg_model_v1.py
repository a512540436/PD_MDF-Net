import copy
import os
import random
import numpy as np
import pandas as pd
import pywt
import torch
from torch import nn, optim
import shutil
import sys
from datetime import datetime
import joblib
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import utils.model_base_function as bf
import utils.model_net as mn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed = 102
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # numpy 的随机种子
random.seed(seed)


def get_label_weight(model_in):
    if model_in.out_dim == 1:
        data_label = 8
        w = "liner"
    elif model_in.out_dim == 2:
        data_label = 1
        w = torch.tensor([0.8, 1.2])
    elif model_in.out_dim == 4:
        data_label = 7
        w = torch.tensor([0.5, 1, 1, 1])
    # elif model_in.out_dim == 4:
    #     data_label = 2
    #     w = torch.tensor([0.5, 1.5, 1.5, 1.5])
    elif model_in.out_dim == 5:
        data_label = 3
        w = torch.tensor([0.5, 1.5, 3, 6, 1])
    elif model_in.out_dim == 7:
        data_label = 4
        w = torch.tensor([0.2, 0.6, 2.5, 2.5, 2.5, 2.5, 1])
    else:
        data_label = 5
        w = torch.tensor([0.2, 0.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 1, 2.5])

    return data_label, w


def get_data_5v(num_in, data_type, data_str, label_num, HPZO_label):
    train_data = []
    test_data = []

    for i in range(1, 6):
        path = f"./datasets/eeg_data/CV_Fold_{data_type}{data_str}/eeg_data_cv_f{i}_{''.join(HPZO_label)}.pkl"
        data = joblib.load(path)

        if i == num_in:
            test_data = data
        else:
            train_data.extend(data)

    train_data = bf.get_date_tensor(train_data, label_num)
    test_data = bf.get_date_tensor(test_data, label_num)

    channel_indices = [9, 20, 26, 28, 4, 6]
    # # 按索引提取通道
    train_data[2] = train_data[2][:, channel_indices, :]
    test_data[2] = test_data[2][:, channel_indices, :]

    data_mean = train_data[2].mean(dim=(0, 2), keepdim=True)
    data_std = train_data[2].std(dim=(0, 2), keepdim=True)

    torch.save({
        'data_mean': data_mean,
        'data_std': data_std
    }, f"./datasets/eeg_data/CV_Fold_{data_type}{data_str}/eeg_cv_{num_in}_normalization_params.pt")

    train_data[2] = (train_data[2] - data_mean) / data_std
    test_data[2] = (test_data[2] - data_mean) / data_std

    if len(train_data[3].shape) == 2:
        bf.print_class_distribution(train_data[3], title="Train")
        bf.print_class_distribution(test_data[3], title="Test")
    print(f"✅ Loaded from .pt: {len(train_data[0])} training samples, {len(test_data[0])} testing samples")
    return train_data, test_data


def get_data_5v_v2(num_in, data_type, data_str, label_num, HPZO_label, wavelet='db4'):
    train_data = []
    test_data = []
    for i in range(1, 6):
        path = f".\MultiClassGaitFC_v2\datasets\eeg_data/CV_Fold_{data_type}{data_str}/eeg_data_cv_f{i}_{''.join(HPZO_label)}.pkl"
        data = joblib.load(path)

        if i == num_in:
            test_data = data
        else:
            train_data.extend(data)

    train_data = bf.get_date_tensor(train_data, label_num)
    test_data = bf.get_date_tensor(test_data, label_num)

    data_mean = train_data[2].mean(dim=(0, 2), keepdim=True)
    data_std = train_data[2].std(dim=(0, 2), keepdim=True)

    train_data[2] = (train_data[2] - data_mean) / data_std
    test_data[2] = (test_data[2] - data_mean) / data_std

    norm_path = f".\MultiClassGaitFC_v2\datasets\eeg_data/CV_Fold_{data_type}{data_str}/eeg_cv_{num_in}_normalization_params.pt"
    torch.save({'data_mean': data_mean, 'data_std': data_std}, norm_path)

    def to_wavelet_domain(x, wave='db4'):
        N, C, T = x.shape
        x_flat = x.reshape(N * C, T).cpu().numpy()  # 合并 batch 和 channel
        coeffs = []

        for i in range(N * C):  # 只循环一次
            cA, cD = pywt.dwt(x_flat[i], wave)
            coeffs.append(np.concatenate([cA, cD]))

        coeffs = np.stack(coeffs).reshape(N, C, -1)
        return torch.tensor(coeffs, dtype=x.dtype, device=x.device)

    x_train_freq = torch.abs(torch.fft.rfft(train_data[2], dim=-1))
    x_test_freq = torch.abs(torch.fft.rfft(test_data[2], dim=-1))
    x_train_wave = to_wavelet_domain(train_data[2], wavelet)
    x_test_wave = to_wavelet_domain(test_data[2], wavelet)

    # ====== 6. 替换原始数据结构 ======
    x_train_all = torch.cat([train_data[2], x_train_freq, x_train_wave], dim=2)
    x_test_all = torch.cat([test_data[2], x_test_freq, x_test_wave], dim=2)

    train_data[2] = x_train_all.to(torch.float32)
    test_data[2] = x_test_all.to(torch.float32)

    if len(train_data[3].shape) == 2:
        bf.print_class_distribution(train_data[3], title="Train")
        bf.print_class_distribution(test_data[3], title="Test")

    print(f"✅ Loaded {len(train_data[0])} train, {len(test_data[0])} test samples")

    return train_data, test_data


def train_func(model_in, num_5v, HPZO_label, data_type, data_str, model_save_path, model_str, out_temp_path, lr=0.0001,
               bs=2000,
               load_para=False,
               epoch_num=1000):
    torch.backends.cudnn.benchmark = True
    model_in = model_in.to(device)
    model_in = model_in.to(memory_format=torch.channels_last)
    data_label, w = get_label_weight(model_in)

    tr_data, te_data = get_data_5v_v2(num_5v, data_type, data_str, label_num=data_label, HPZO_label=HPZO_label)
    dataloader_tr = bf.get_dataloader_all(tr_data, bs, True)
    dataloader_tr_ = bf.get_dataloader_all(tr_data, 10000, False)
    dataloader_te = bf.get_dataloader_all(te_data, 10000, False)

    model_path = f"{model_save_path}/model_{model_str}_class{model_in.out_dim}_cv{num_5v}.pth"
    if load_para:
        state_dict = torch.load(model_path)
        model_in.load_state_dict(state_dict['model_state_dict'])
    else:
        bf.init_weights_(model_in, n_layer=4)
        # model_in.apply(init_weights)

    if w == "liner":
        # criterion = nn.MSELoss().to(device)
        criterion = bf.ZeroTolerantMSELoss().to(device)
    else:
        criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=0.0).to(device)

    # criterion = FocalLoss(weight=w.to(device), gamma=2).to(device)
    optimizer = optim.AdamW(model_in.parameters(), lr=lr, weight_decay=0.001)  # weight_decay=0.001
    scheduler = bf.lr_change(optimizer, epoch_num, lr)

    f1_max = -1
    mse_min = 10000
    no_improve_epochs = 0
    scaler = GradScaler()
    for epoch in range(epoch_num):
        if epoch % 1 == 0:
            # # out_dict = {"loss": avg_loss, "acc_s": acc_s, "precision_s": precision_s, "recall_s": recall_s,
            # # "f1_s": f1_s, "roc_auc_s": roc_auc_s, "fpr_s": fpr_s, "tpr_s": tpr_s, "acc_p": acc_p,
            # # "precision_p": precision_p, "recall_p": recall_p, "f1_p": f1_p, "roc_auc_p": roc_auc_p, "fpr_p": fpr_p,
            # # "tpr_p": tpr_p, "summary_p": summary_p}
            if w == "liner":
                output, mse_out = bf.model_eval(model_in, te_data, dataloader_te, criterion, epoch, ["mse_p", mse_min],
                                                "AL", optimizer, model_path)

                if mse_out < mse_min:  # MSE 越小越好
                    mse_min = mse_out
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                    if no_improve_epochs >= 1000000:
                        print("⚠️ 连续无提升，加载最优模型重试")
                        model_in, optimizer = bf.load_checkpoint(model_in, optimizer, model_path)
                        no_improve_epochs = 0

                sys.stdout.write('\r' + ' ' * 50 + '\r')  # 清空整行
                print(f"epoch:{epoch}, Loss:{output['loss']:.6f}, MSE:{output['mse_s']:.6f}, MAE:{output['mae_s']:.6f}"
                      f", MSE_p:{output['mse_p']:.6f}, MAE_p:{output['mae_p']:.6f}"
                      f", R2:{output['r2_s']:.6f}", end='', flush=True)
            else:
                output, f1_out = bf.model_eval(model_in, te_data, dataloader_te, criterion, epoch, ["f1_p", f1_max],
                                               "AL", optimizer, model_path)
                if f1_out >= f1_max:
                    f1_max = f1_out
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                    if no_improve_epochs >= 1000000:
                        print("⚠️ 连续无提升，加载最优模型重试")
                        model_in, optimizer = bf.load_checkpoint(model_in, optimizer, model_path)
                        no_improve_epochs = 0

                sys.stdout.write('\r' + ' ' * 50 + '\r')  # 清空整行
                print(f"epoch:{epoch}, Loss:{output['loss']:.6f}, Acc:{output['acc_s']:.6f}, F1:{output['f1_s']:.6f}"
                      f", Acc_p:{output['acc_p']:.6f}, F1_p:{output['f1_p']:.6f}", end='', flush=True)

        for x_info, x_data, y_t in dataloader_tr:
            model_in.train()
            optimizer.zero_grad()
            with autocast():
                y_p = model_in(x_data)
                loss = criterion(y_p, y_t)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

    state_dict = torch.load(model_path)
    model_in.load_state_dict(state_dict['model_state_dict'])

    print("TR" + "-" * 70)
    out_data_tr, _ = bf.model_eval(model_in, tr_data, dataloader_tr_, criterion, "TR最终验证", printing_options="ALL")
    print("TE" + "-" * 70)
    out_data_te, _ = bf.model_eval(model_in, te_data, dataloader_te, criterion, "TE最终验证", printing_options="ALL")

    metric_path = out_temp_path + "/output_metrics.xlsx"
    curve_path = out_temp_path + "/result_fpr_tpr_summary_fold{}.xlsx"
    bf.save_fold_result_to_excel(out_data_tr, out_data_te, num_5v, metric_path=metric_path,
                                 curve_path=curve_path)


def train_and_save_all(model, HPZO_label, batch_size, epoch, learn_rate, data_type, data_str,
                       train=True, temp_dir="./result_save/eeg_result/temp",
                       base_save_dir="./result_save/eeg_result",
                       model_save_dir="./model_save/eeg_model",
                       cv_folds=5):
    if train:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        print(f"✅ '{temp_dir}' 已创建并清空。")

        for fold in range(1, cv_folds + 1):
            model_copy = copy.deepcopy(model)
            print("─" * 100)
            trainable_params = sum(p.numel() for p in model_copy.parameters() if p.requires_grad)
            print(f"第{fold}折，{model.__class__.__name__}网络：可训练参数总量: {trainable_params}")
            train_func(model_copy, num_5v=fold, HPZO_label=HPZO_label, data_type=data_type, data_str=data_str,
                       model_save_path=model_save_dir,
                       model_str=model.__class__.__name__, out_temp_path=temp_dir,
                       lr=learn_rate, bs=batch_size, epoch_num=epoch)

    time_str = datetime.now().strftime("%Y%m%d%H%M")
    folder_name = f"{model.__class__.__name__}_class{model.out_dim}_{data_type}_{data_str}_{''.join(HPZO_label)}" \
                  f"_bs{batch_size}_ep{epoch}_lr{learn_rate}_{time_str}"
    full_path = os.path.join(base_save_dir, folder_name)
    os.makedirs(full_path, exist_ok=True)
    print(f"✅ 输出文件夹已创建：{full_path}")

    df = pd.read_excel(os.path.join(temp_dir, "output_metrics.xlsx"))
    avg_row = df.iloc[:cv_folds, 1:].mean()
    avg_label = pd.Series(["avg"], index=["fold"])
    avg_full_row = pd.concat([avg_label, avg_row])
    df.loc[len(df)] = avg_full_row
    df = bf.insert_blank_cols_for_auc(df)
    df.to_excel(os.path.join(full_path, "output_metrics_avg.xlsx"), index=False)
    print(f"✅ 平均值已追加并保存至：{full_path}")

    summary_dir = os.path.join(full_path, "summary_fpr_tpr")
    os.makedirs(summary_dir, exist_ok=True)
    for i in range(1, cv_folds + 1):
        src = os.path.join(temp_dir, f"result_fpr_tpr_summary_fold{i}.xlsx")
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(summary_dir, f"result_fpr_tpr_summary_fold{i}.xlsx"))

    model_dir = os.path.join(full_path, "model")
    os.makedirs(model_dir, exist_ok=True)
    for i in range(1, cv_folds + 1):
        src = os.path.join(model_save_dir, f"model_{model.__class__.__name__}_class{model.out_dim}_cv{i}.pth")
        dst = os.path.join(model_dir, f"model_{model.__class__.__name__}_class{model.out_dim}_cv{i}.pth")
        if os.path.exists(src):
            shutil.copy2(src, dst)

    bf.collect_te_summary(summary_dir, "eeg_te_summary_all.xlsx")


if __name__ == "__main__":
    # SingleClassifier(in_dim=39 * 2 * 175, out_dim=5)

    train_and_save_all(
        model=mn.MDFNet(n_channels=32, shape_T=20, n_classes=2),
        HPZO_label=["H", "P"],
        batch_size=128,
        epoch=500,
        learn_rate=0.0001,
        data_type="S_",
        data_str="d1",
        train=True,
        temp_dir="./result_save/eeg_result/temp",
        base_save_dir="./result_save/eeg_result",
        model_save_dir="./model_save/eeg_model",
        cv_folds=5,
    )


