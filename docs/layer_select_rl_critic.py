import torch
import torch.nn as nn
import torch.optim as optim
import random

# ---------------- Policy 网络 ----------------
class LayerPolicy(nn.Module):
    """Policy 网络: 从 num_layers 中为每个样本独立选一层"""
    def __init__(self, num_layers):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_layers))  # 可学习参数

    def forward(self, batch_size, temperature=1.0, hard=True):
        probs = torch.softmax(self.logits, dim=-1)  # 全局 logits -> probs
        # 每个样本独立采样
        dist = torch.distributions.Categorical(probs.expand(batch_size, -1))
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, probs.detach()


# ---------------- Critic 网络 ----------------
class Critic(nn.Module):
    """Critic 网络: 预测 reward 作为 baseline"""
    def __init__(self, num_layers, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_layers, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, logits, batch_size):
        probs = torch.softmax(logits, dim=-1)
        probs = probs.expand(batch_size, -1)  # 每个样本独立
        baseline = self.net(probs).squeeze(-1)
        return baseline


# ---------------- 模拟 reward ----------------
def simulate_reward(actions, best_layer=3):
    """每个样本独立 reward = -(action - best_layer)^2 + noise"""
    rewards = []
    for a in actions:
        r = - (a.item() - best_layer)**2 + random.gauss(0, 0.2)
        rewards.append(r)
    return torch.tensor(rewards, dtype=torch.float32)


# ---------------- 训练函数 ----------------
def train_policy_batch(num_layers=33, batch_size=8, epochs=1000, lr_policy=0.1, lr_critic=0.01):
    policy = LayerPolicy(num_layers)
    critic = Critic(num_layers)
    
    optimizer_policy = optim.Adam([policy.logits], lr=lr_policy)
    optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic)

    reward_history = []

    for epoch in range(epochs):
        # ---- Policy forward ----
        actions, log_probs, probs = policy(batch_size)
        
        # ---- Simulate reward ----
        rewards = simulate_reward(actions)

        # ---- Critic forward ----
        baseline = critic(policy.logits, batch_size)
        advantage = rewards - baseline.detach()

        # ---- Losses ----
        policy_loss = -(log_probs * advantage).mean()
        critic_loss = ((baseline - rewards)**2).mean()

        optimizer_policy.zero_grad()
        optimizer_critic.zero_grad()
        total_loss = policy_loss + critic_loss
        total_loss.backward()
        optimizer_policy.step()
        optimizer_critic.step()

        reward_history.extend(rewards.tolist())

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:3d} | "
                  f"Actions: {actions.tolist()} | "
                  f"Rewards: {rewards.tolist()} | "
                  f"Baseline: {baseline.tolist()} | "
                  f"Policy probs (first 5): {probs[:5].numpy()}")

    return policy, critic, reward_history


if __name__ == "__main__":
    policy, critic, reward_history = train_policy_batch()

    final_probs = torch.softmax(policy.logits, dim=-1).detach().numpy()
    print("\n最终策略分布:", final_probs)
