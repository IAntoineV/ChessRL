import torch
import torch.nn as nn


def ppo_loss(
    old_logits,
    old_values,
    advantages,
    states,
    actions,
    returns,
    model,
    optimizer,
    epochs,
    clip_ratio,
):
    def compute_loss(logits, values, actions, returns, old_logits, advantages):
        action_probs = torch.gather(
            torch.softmax(logits, dim=-1), 1, actions.unsqueeze(1)
        ).squeeze(1)
        old_action_probs = torch.gather(
            torch.softmax(old_logits, dim=-1), 1, actions.unsqueeze(1)
        ).squeeze(1)

        # Policy loss
        ratio = torch.exp(
            torch.log(action_probs + 1e-10) - torch.log(old_action_probs + 1e-10)
        )
        clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
        policy_loss = -torch.mean(
            torch.min(ratio * advantages, clipped_ratio * advantages)
        )

        # Value loss
        value_loss = torch.mean((values.squeeze(-1) - returns) ** 2)

        # Entropy bonus (optional)
        entropy_bonus = torch.mean(
            torch.sum(
                torch.softmax(logits, dim=-1) * torch.log_softmax(logits, dim=-1),
                dim=-1,
            )
        )

        total_loss = (
            policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus
        )  # Entropy regularization
        return total_loss

    for _ in range(epochs):
        logits, values = model(states)
        loss = compute_loss(logits, values, actions, returns, old_logits, advantages)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss


def get_advantages(returns, values):
    advantages = returns - values.squeeze(-1)
    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)


# Main training loop


def train_ppo(
    env,
    policy_net,
    optimizer,
    device,
    max_episodes=10,
    lr_actor=3e-4,
    clip_ratio=0.2,
    gamma=0.99,
    epochs=4,
):
    """Train the policy network using Proximal Policy Optimization (PPO)."""

    # Main training loop
    max_episodes = 1000
    max_steps_per_episode = 1000

    def discount_rewards(rewards, gamma):
        discounted_sum = 0
        returns = []
        for r in rewards[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.append(discounted_sum)
        returns.reverse()
        return returns

    for episode in range(max_episodes):
        states, actions, rewards, values = [], [], [], []
        state = env.reset()
        for _ in range(max_steps_per_episode):
            logits, value = policy_net(state)

            action = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value)

            state = next_state

            if done:
                returns_batch = discount_rewards(rewards, gamma)
                returns_batch = torch.tensor(returns_batch, dtype=torch.float32).to(
                    device
                )

                states = torch.cat(states)
                actions = torch.tensor(actions, dtype=torch.int64).to(device)
                values = torch.cat(values).squeeze(-1)

                advantages = get_advantages(returns_batch, values)
                old_logits, _ = policy_net(states)
                old_logits = old_logits.detach()

                loss = ppo_loss(
                    old_logits,
                    values,
                    advantages,
                    states,
                    actions,
                    returns_batch,
                    policy_net,
                    optimizer,
                    epochs,
                    clip_ratio,
                )
                print(f"Episode: {episode + 1}, Loss: {loss.item()}")

                break
