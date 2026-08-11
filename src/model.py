# 模型结构定义
from torch import nn
import config

class InputMethodModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # 词嵌入层: 将离散的 token 索引映射为稠密的低维向量表示。
        # 每个 token 对应一个可学习的 embedding 向量，
        # 初始值随机初始化，并在训练过程中逐步学习到词的语义信息
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,          # 词表大小（token 索引取值范围）
            embedding_dim=config.EMBEDDING_DIM  # 每个 token 映射后的向量维度
        )

        # RNN层: 用于捕捉输入序列中的上下文依赖关系，
        # 通过循环结构将历史信息保留在隐藏状态中，
        # 从而为每个位置生成蕴含上下文的语义表示
        self.rnn = nn.RNN(
            input_size=config.EMBEDDING_DIM,  # 输入特征维度
            hidden_size=config.HIDDEN_SIZE,   # 隐藏层状态维度
            batch_first=True                  # 输入张量形状为 (batch, seq_len, feature)
        )

        # 全连接层：将 RNN 的隐藏状态映射为词表大小的得分
        # 每个位置输出一个概率分布，表示下一个字的候选概率
        self.linear = nn.Linear(in_features=config.HIDDEN_SIZE,  # 输入为 RNN 隐藏层维度
                                out_features=vocab_size)         # 输出为词表大小（各候选字得分）

    def forward(self, x):
        # 输入 x.shape: [batch_size, seq_len]，每个元素是输入序列中对应位置的 token 索引

        # 通过词嵌入层将 token 索引映射为稠密向量
        embed = self.embedding(x)
        # embed.shape: [batch_size, seq_len, embedding_dim]

        # 将嵌入序列送入 RNN，捕捉序列中的上下文依赖关系
        # output 为每个时间步的隐藏状态，hn 为最后一个时间步的隐藏状态
        output, hn = self.rnn(embed)
        # output.shape: [batch_size, seq_len, hidden_size]

        # 取最后一个时间步的隐藏状态作为整句的语义表示
        last_hidden_state = output[:, -1, :]
        # last_hidden_state.shape: [batch_size, hidden_size]

        # 通过全连接层将隐藏状态映射为词表大小的得分，
        # 表示下一个字的各候选概率（后续配合 softmax 得到概率分布）
        logits = self.linear(last_hidden_state)
        # logits.shape: [batch_size, vocab_size]

        return logits


'''
<快速查阅>
nn.Module模型常用方法:
1. 前向传播: model(x)  # 返回模型输出
2. 打印模型结构: print(model)  # 查看各层结构及参数形状
3. 获取模型参数: model.parameters()  # 返回所有可学习参数迭代器
4. 切换训练模式: model.train()  # 启用 Dropout/BatchNorm 的训练行为
5. 切换评估模式: model.eval()  # 关闭 Dropout、固定 BatchNorm 统计量
6. 保存模型: torch.save(model.state_dict(), path)  # 仅保存参数
7. 加载模型: model.load_state_dict(torch.load(path))  # 加载参数到模型
8. 获取状态字典: model.state_dict()  # 返回 {参数名: 张量} 字典
9. 梯度清零: optimizer.zero_grad()  # 清空上一步累积的梯度
10. 反向传播: loss.backward()  # 计算各参数梯度
11. 参数更新: optimizer.step()  # 根据梯度更新参数
12. 冻结参数: model.requires_grad_(False)  # 停止所有参数梯度更新

<训练循环模板>
for epoch in range(config.EPOCHS):
    model.train()                      # 切到训练模式
    for x, y in train_loader:
        optimizer.zero_grad()          # 梯度清零
        logits = model(x)              # 前向传播
        loss = criterion(logits, y)    # 计算损失
        loss.backward()                # 反向传播
        optimizer.step()               # 更新参数
    model.eval()                       # 切到评估模式
    with torch.no_grad():              # 评估时关闭梯度计算
        # 验证逻辑...
'''