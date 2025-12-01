import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List

from tools.simi_loss import SimiLoss, essential_layers


class LayerPolicy(nn.Module):
    """Policy over layers (shared across samples): logits over num_layers.
    At training step, for a batch of N samples, sample one layer per sample.
    """
    def __init__(self, num_layers: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_layers))

    def forward(self, batch_size: int):
        dist = torch.distributions.Categorical(logits=self.logits.expand(batch_size, -1))
        actions = dist.sample()                  # [N]
        log_probs = dist.log_prob(actions)       # [N]
        return actions, log_probs, dist


def collect_sample_ids(predictions_root: str) -> List[str]:
    import os
    layer40_dir = os.path.join(predictions_root, "layer40")
    import os
    ids = []
    for f in os.listdir(layer40_dir):
        p = os.path.join(layer40_dir, f)
        if os.path.isfile(p) and f.lower().endswith(".npy"):
            ids.append(os.path.splitext(f)[0])
    ids.sort()
    return ids


def build_loss_table(predictions_root: str, dataset_root: str, sample_ids: List[str]) -> torch.Tensor:
    """Precompute per-sample per-layer losses: loss = BCE_Gauss + Dice_Gauss.
    Returns: tensor of shape [N_samples, num_layers].
    """
    simi = SimiLoss()
    losses = []
    for sid in sample_ids:
        try:
            bce_losses, bce_gauss_losses, dice_losses, dice_gauss_losses = simi.compute_per_layer_losses_for_id(
                predictions_root=predictions_root,
                dataset_root=dataset_root,
                sample_id=sid,
                viz_out_dir="",
                pos_weight_user=-1.0,
                pos_weight_gauss_user=-1.0,
            )
            # Use only Gaussian variants as requested
            per_layer = []
            for i in range(len(essential_layers)):
                per_layer.append(float(bce_gauss_losses[i] + dice_gauss_losses[i]))
            losses.append(per_layer)
        except Exception as e:
            print(f"[Skip] {sid}: {e}")
            continue
    if len(losses) == 0:
        raise RuntimeError("No valid samples collected for RL training.")
    loss_table = torch.tensor(losses, dtype=torch.float32)
    return loss_table


def train_policy_with_losses(loss_table: torch.Tensor, epochs: int = 300, lr: float = 1e-1, seed: int = 42):
    torch.manual_seed(seed)
    num_samples, num_layers = loss_table.shape
    policy = LayerPolicy(num_layers)
    optimizer = optim.Adam([policy.logits], lr=lr)

    for ep in range(1, epochs + 1):
        actions, log_probs, _ = policy(num_samples)
        # rewards = -(BCE_Gauss + Dice_Gauss)
        rewards = -loss_table[torch.arange(num_samples), actions]
        # simple baseline: per-sample mean over layers (detached)
        baseline = -loss_table.mean(dim=1)
        advantage = (rewards - baseline).detach()

        policy_loss = -(log_probs * advantage).mean()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if ep % 20 == 0:
            with torch.no_grad():
                avg_reward = rewards.mean().item()
                print(f"Epoch {ep:03d} | policy_loss: {policy_loss.item():.6f} | avg_reward: {avg_reward:.6f}")

    with torch.no_grad():
        probs = torch.softmax(policy.logits, dim=-1)
        chosen = torch.argmax(probs).item()
        print("Learned layer probabilities:", probs.cpu().numpy())
        print("Greedy chosen layer:", chosen)
    return policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_root",  type=str, default="/home/xinyin/qianrui/lmm/UGround_26/uground-13B@reason_seg_val")
    parser.add_argument("--dataset_root", type=str, default="/home/xinyin/qianrui/lmm/dataset_sesame/reason_seg/ReasonSeg/val")
    parser.add_argument("--max_samples", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-1)
    args = parser.parse_args()

    # 1) Collect sample ids
    sample_ids = collect_sample_ids(args.predictions_root)
    if args.max_samples > 0:
        sample_ids = sample_ids[:args.max_samples]
    print(f"Using {len(sample_ids)} samples for RL layer selection training")

    # 2) Precompute per-layer losses per sample
    loss_table = build_loss_table(args.predictions_root, args.dataset_root, sample_ids)
    print("Loss table shape:", loss_table.shape)

    # 3) Train policy to maximize reward = -(BCE_Gauss + Dice_Gauss)
    train_policy_with_losses(loss_table, epochs=args.epochs, lr=args.lr)


if __name__ == "__main__":
    main()
