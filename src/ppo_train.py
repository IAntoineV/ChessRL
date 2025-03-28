import torch
from collections import defaultdict


def ppo_loss(
    old_log_probs,
    old_values,
    advantages,
    states,
    returns,
    model,
    optimizer,
    epochs,
    clip_ratio,
):
    def compute_loss(log_probs, values, returns, old_log_probs, advantages):
        # Compute ratio and apply PPO clipping
        ratio = torch.exp(log_probs) / (torch.exp(old_log_probs) + 1e-10)
        clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
        policy_loss = -torch.mean(
            torch.min(ratio * advantages, clipped_ratio * advantages)
        )

        # Value loss (mean squared error)
        value_loss = torch.mean((values.squeeze(-1) - returns) ** 2)

        # Entropy bonus (to encourage exploration)
        entropy_bonus = -torch.mean(
            torch.sum(
                torch.exp(log_probs) * log_probs,
                dim=-1,
            )
        )

        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus
        return total_loss

    for _ in range(epochs):
        for i, state in enumerate(states):
            log_prob, values = model(state)
            loss = compute_loss(
                log_prob, values, returns, old_log_probs[i], advantages[i]
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss


def get_advantages(returns, values):
    advantages = returns - values.squeeze(-1)
    if advantages.numel() > 1 and advantages.std() != 0:
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages


def train_ppo(
    env,
    policy_net,
    optimizer,
    device,
    max_episodes=1000,
    max_steps_per_episode=100,
    clip_ratio=0.8,
    gamma=0.3,
    epochs=100,
):
    def discount_rewards(rewards, gamma):
        discounted_sum = 0
        returns = []
        for r in reversed(rewards):
            discounted_sum = r + gamma * discounted_sum
            returns.append(discounted_sum)
        returns.reverse()
        return returns

    for episode in range(max_episodes):
        states, actions, rewards, values, old_log_probs = [], [], [], [], []

        # For Gym versions that return a tuple from reset()
        state = env.reset()
        done = False
        list_infos = defaultdict(lambda: 0)
        step = 0
        while not done and step < max_steps_per_episode:
            log_probs, value = policy_net(state)
            action_probs = torch.clamp(torch.exp(log_probs), min=1e-12)
            action = torch.multinomial(action_probs, 1).item()

            next_state, reward, terminated, truncated, infos = env.step(action)
            for key in infos.keys():
                list_infos[key] += 1 / max_steps_per_episode

            done = terminated or truncated
            # Handle new Gym API if reset returns a tuple

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            old_log_probs.append(log_probs.detach())
            values.append(value.detach())

            state = next_state
            step += 1

        # Compute discounted rewards and advantages
        returns_batch = discount_rewards(rewards, gamma)
        returns_batch = torch.tensor(returns_batch, dtype=torch.float32, device=device)
        values = torch.tensor(values, dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.int64, device=device)

        advantages = get_advantages(returns_batch, values)

        loss = ppo_loss(
            old_log_probs,
            values,
            advantages,
            states,
            returns_batch,
            policy_net,
            optimizer,
            epochs,
            clip_ratio,
        )
        print(f"action {action}")
        print(env.board.unicode())
        print(list_infos)
        print(f"Episode: {episode + 1}, Loss: {loss.item()}")
