import torch


def ppo_loss(
    old_log_probs,
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
    def compute_loss(log_probs, values, actions, returns, old_log_probs, advantages):
        # Compute probabilities for actions taken
        action_probs = torch.gather(
            torch.exp(log_probs), 1, actions.unsqueeze(1)
        ).squeeze(1)
        old_action_probs = torch.gather(
            torch.exp(old_log_probs), 1, actions.unsqueeze(1)
        ).squeeze(1)

        # Compute ratio and apply PPO clipping
        ratio = action_probs / (old_action_probs + 1e-10)
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
        log_probs, values = model(states)
        loss = compute_loss(
            log_probs, values, actions, returns, old_log_probs, advantages
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
    max_steps_per_episode=1000,
    clip_ratio=0.8,
    gamma=0.3,
    epochs=4,
):
    def discount_rewards(rewards, gamma):
        discounted_sum = 0
        returns = []
        for r in reversed(rewards):
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)
        return returns

    for episode in range(max_episodes):
        states, actions, rewards, values = [], [], [], []

        # For Gym versions that return a tuple from reset()
        state = env.reset()

        done = False
        step = 0
        while not done and step < max_steps_per_episode:
            log_probs, value = policy_net(state)
            action_probs = torch.clamp(torch.exp(log_probs), min=1e-12)
            action = torch.multinomial(action_probs, 1).item()

            next_state, reward, terminated, truncated, infos = env.step(action)
            print(infos)
            done = terminated or truncated
            # Handle new Gym API if reset returns a tuple

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value.detach())

            state = next_state
            step += 1

        # Compute discounted rewards and advantages
        returns_batch = discount_rewards(rewards, gamma)
        returns_batch = torch.tensor(returns_batch, dtype=torch.float32, device=device)

        states = torch.cat(states, dim=0)
        actions = torch.tensor(actions, dtype=torch.int64, device=device)
        values = torch.cat(values).squeeze(-1)

        advantages = get_advantages(returns_batch, values)
        old_log_prob, _ = policy_net(states)
        old_log_prob = old_log_prob.detach()

        loss = ppo_loss(
            old_log_prob,
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
        print(f"action {action}")
        print(env.board.unicode())
        print(f"Episode: {episode + 1}, Loss: {loss.item()}")
