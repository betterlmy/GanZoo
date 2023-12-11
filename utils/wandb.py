import wandb


# 初始化 wandb
wandb.init(project='my_project_name')

# 配置参数（可选）
config = wandb.config
config.learning_rate = 0.01

# 在训练循环中记录指标
for epoch in range(epochs):
    # ... 训练模型 ...
    wandb.log({'epoch': epoch, 'loss': loss})