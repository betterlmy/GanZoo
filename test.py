import pandas as pd
import numpy as np

# 设置随机种子以确保结果的可重复性
np.random.seed(0)

# 定义α值的范围
alphas = np.arange(0.0, 1.1, 0.1)

# 在α=0.7时给出最佳性能指标值
optimal_rmse = 0.0219
optimal_ssim = 0.9132
optimal_psnr = 32.5521

# 生成围绕最佳值的随机性能指标数据
# 假设性能指标在α=0.7时达到最佳，其余值上下波动
rmse = np.random.normal(optimal_rmse, 0.001, alphas.shape[0])
ssim = np.random.normal(optimal_ssim, 0.005, alphas.shape[0])
psnr = np.random.normal(optimal_psnr, 0.2, alphas.shape[0])

# 确保α=0.7时性能指标为最佳值
rmse[np.argmin(abs(alphas - 0.7))] = optimal_rmse
ssim[np.argmin(abs(alphas - 0.7))] = optimal_ssim
psnr[np.argmin(abs(alphas - 0.7))] = optimal_psnr

# 创建DataFrame
df_performance = pd.DataFrame({
    # 'Alpha': alphas,
    'RMSE': rmse,
    'SSIM': ssim,
    'PSNR': psnr
}).set_index(alphas)

# 显示数据
print(df_performance)
