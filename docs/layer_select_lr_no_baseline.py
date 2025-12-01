import torch
import torch.nn as nn
import torch.optim as optim
import random

class LayerPolicy(nn.Module):
    """一个简单的 policy 网络，用于从 num_layers 中选一层"""
    def __init__(self, num_layers):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_layers))  # 可学习参数，相当于每层的 preference

    def forward(self):
        probs = torch.softmax(self.logits, dim=-1)  # 采样概率
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()  # 采样层
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, probs.detach()


def simulate_reward(action, num_layers):
    """假数据: 定义 reward = -(action-最佳层)^2 + noise"""
    best_layer = 3  # 假设第 3 层是最优层
    reward = - (action - best_layer) ** 2 + random.gauss(0, 0.2)  # 加点噪声
    return reward


def train_policy(num_layers=33, epochs=1000, lr=0.1):
    policy = LayerPolicy(num_layers)
    optimizer = optim.Adam([policy.logits], lr=lr)

    reward_history = []
    for epoch in range(epochs):
        action, log_prob, probs = policy()
        reward = simulate_reward(action, num_layers)

        # REINFORCE 损失
        loss = -log_prob * reward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        reward_history.append(reward)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d} | "
                  f"选择的层: {action} | "
                  f"Reward: {reward:.3f} | "
                  f"策略分布: {probs.numpy()}")

    return policy, reward_history


if __name__ == "__main__":
    policy, reward_history = train_policy()

    # 测试：打印最终选择概率
    final_probs = torch.softmax(policy.logits, dim=-1).detach().numpy()
    print("\n最终策略分布:", final_probs)
