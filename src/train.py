# 模型训练
import torch
from dataset import get_dataloader
from model import InputMethodModel
import config
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter

def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    """
    训练一个 epoch。

    Args:
        model (InputMethodModel): 输入法模型。
        dataloader (torch.utils.data.DataLoader): 训练数据加载器。
        loss_fn (torch.nn.Module): 损失函数。
        optimizer (torch.optim.Optimizer): 优化器。
        device (torch.device): 计算设备（CPU 或 GPU）。

    Returns:
        float: 该 epoch 的平均损失。
    """

    # 把模型切换为"训练模式"。
    # 训练模式下会启用 Dropout、BatchNorm 等只在训练时生效的机制，
    # 让模型在训练时"放开手脚"学习，推理时则要调用 model.eval() 关闭它们。
    model.train()

    # total_loss 用来累加本 epoch 中所有 batch 的损失，用于最后求平均值。
    total_loss = 0.0

    # dataloader 每次迭代会返回一个 batch 的数据：
    #   inputs  : 一批上文序列输入，形状类似 (batch_size, seq_len, ...)
    #   targets : 与输入对应的正确输出（标签/目标汉字），用于计算损失
    for inputs, targets in tqdm(dataloader, desc='Training'):
        # 把数据搬运到指定设备（GPU/CPU）上。
        # 模型在哪个设备上，数据就必须在哪个设备上，否则会报错或无法计算。
        inputs = inputs.to(device)
        targets = targets.to(device)

        # ---------- 前向传播 ----------
        # 把 inputs 送入模型，得到预测结果 logits。
        # logits 是每个位置/每个候选字"分数"（未经过 softmax 的原始输出）。
        logits = model(inputs)

        # 用损失函数计算"预测结果"与"真实标签"之间的差距。
        # loss 是一个标量（单个数字），数值越大说明模型预测得越差。
        loss = loss_fn(logits, targets)

        # ---------- 反向传播 ----------
        # 1. 根据 loss 自动计算模型中所有参数的梯度（gradient）。
        #    梯度告诉我们"参数往哪个方向调整可以让 loss 变小"。
        loss.backward()

        # 2. 根据刚才算出的梯度，更新一次模型参数（一步梯度下降）。
        optimizer.step()

        # 3. 把之前累积的梯度清零。
        #    因为 backward() 是"累加"梯度的，不清零的话，
        #    下一个 batch 的梯度会和当前 batch 的梯度叠加，导致训练错误。
        optimizer.zero_grad()

        # 把当前 batch 的损失累加到 total_loss 中。
        # loss.item() 取出 loss 中的纯数字（Python float），方便累加。
        total_loss += loss.item()

    # 返回该 epoch 的平均损失 = 所有 batch 损失之和 ÷ batch 数量。
    return total_loss / len(dataloader)

def train():
    # cpu or gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 数据集
    dataloader = get_dataloader()
    # 词表
    with open(config.MODELS_DIR / 'vocab.txt', 'r', encoding='utf-8') as f:
        vocab_lsit = f.read().splitlines()
    # print(vocab_lsit[:10])
    # 模型
    model = InputMethodModel(len(vocab_lsit))
    model.to(device)
    # 损失函数
    loss_fn = torch.nn.CrossEntropyLoss()
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # tensorboard
    writer = SummaryWriter(log_dir=config.LOGS_DIR)

    # 记录训练开始时间
    start_time = time.perf_counter()  # perf_counter 提供更高精度的计时

    # 开始训练
    for epoch in range(config.EPOCHS):
        print('=' * 10, f' Epoch: {epoch + 1} ', '=' * 10)
        loss = train_one_epoch(model, dataloader, loss_fn, optimizer, device)
        print(f'loss: {loss}')

        # 记录训练结果
        writer.add_scalar('loss', loss, epoch) # tag, y, x

    writer.close()

    # 计算总用时
    end_time = time.perf_counter()
    elapsed_seconds = end_time - start_time

    # 格式化为 时:分:秒
    hours, rem = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f'训练结束，用时：{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}')  # 例：01:23:45.67

        
    return

if __name__ == '__main__':
    train()    


# git push main