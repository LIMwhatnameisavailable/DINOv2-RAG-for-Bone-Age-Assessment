"""
LIM/code/utils/gw_mae.py
生长加权 MAE 损失函数（GW-MAE）
设计说明：连续高斯加权函数，峰值位于 120 月（10 岁），
对应青春期前发育加速阶段。与现有代码 eval_utils.py 中的
分段常数版本（GrowthWindowWeights）临床动机相同，但实现不同。
LIM 项目统一使用本文件定义的版本。
"""
import torch
import torch.nn as nn


class GrowthWeightedMAELoss(nn.Module):
    """
    GW-MAE：生长加权平均绝对误差损失函数。

    权重函数：
        w(age) = w_min + (w_max - w_min) * exp(-0.5 * ((age - mu) / sigma)^2)

    参数：
        mu    : 高斯峰值位置（月），默认 120
        sigma : 高斯宽度（月），默认 36
        w_min : 最小权重，默认 0.5
        w_max : 最大权重，默认 2.0
    """

    def __init__(self, mu: float = 120.0, sigma: float = 36.0,
                 w_min: float = 0.5, w_max: float = 2.0):
        super().__init__()
        self.mu    = mu
        self.sigma = sigma
        self.w_min = w_min
        self.w_max = w_max

    def weight(self, age_months: torch.Tensor) -> torch.Tensor:
        gaussian = torch.exp(
            -0.5 * ((age_months - self.mu) / self.sigma) ** 2
        )
        return self.w_min + (self.w_max - self.w_min) * gaussian

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        w = self.weight(target)
        return (w * torch.abs(pred - target)).mean()


class CombinedLoss(nn.Module):
    """
    联合损失：alpha_mae * MAE + beta_gw_mae * GW-MAE

    三种使用模式（通过系数控制）：
        纯 MAE      : alpha_mae=1.0, beta_gw_mae=0.0
        纯 GW-MAE   : alpha_mae=0.0, beta_gw_mae=1.0
        联合损失    : alpha_mae=1.0, beta_gw_mae=1.0
    """

    def __init__(self, alpha_mae: float = 1.0,
                 beta_gw_mae: float = 1.0, **gw_kwargs):
        super().__init__()
        self.alpha_mae   = alpha_mae
        self.beta_gw_mae = beta_gw_mae
        self.mae_loss    = nn.L1Loss()
        self.gw_mae_loss = GrowthWeightedMAELoss(**gw_kwargs)

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        if self.alpha_mae != 0.0:
            loss = loss + self.alpha_mae * self.mae_loss(pred, target)
        if self.beta_gw_mae != 0.0:
            loss = loss + self.beta_gw_mae * self.gw_mae_loss(pred, target)
        return loss


if __name__ == "__main__":
    # 快速验证：构造一批假数据，检查 loss 能正常前向传播
    import torch

    pred   = torch.tensor([100.0, 120.0, 150.0, 200.0])
    target = torch.tensor([ 95.0, 125.0, 145.0, 210.0])

    gw   = GrowthWeightedMAELoss()
    comb = CombinedLoss(alpha_mae=1.0, beta_gw_mae=1.0)

    print(f"GW-MAE loss       : {gw(pred, target):.4f}")
    print(f"Combined loss     : {comb(pred, target):.4f}")

    # 验证权重曲线：0月和120月的权重应分别接近 w_min 和 w_max
    ages = torch.tensor([0.0, 60.0, 120.0, 180.0, 228.0])
    w    = gw.weight(ages)
    print("\n权重验证（mu=120, sigma=36, w_min=0.5, w_max=2.0）：")
    for age, wi in zip(ages.tolist(), w.tolist()):
        print(f"  age={age:5.0f} 月  →  w={wi:.4f}")