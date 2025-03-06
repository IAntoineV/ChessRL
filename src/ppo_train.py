import torch
import torch.nn as nn


class PPOBuffer:
    def __init__(self, max_size, state_dim):
        """
        Initialize a PPO buffer to store trajectories.
        Args:
            max_size: Maximum number of transitions to store.
            state_dim: Dimension of the state space.
        """
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        # Initialize storage for states, actions, rewards, etc.
        self.states = torch.zeros((max_size, state_dim), dtype=torch.float32)
        self.actions = torch.zeros(max_size, dtype=torch.int64)
        self.rewards = torch.zeros(max_size, dtype=torch.float32)
        self.dones = torch.zeros(max_size, dtype=torch.float32)
        self.log_probs = torch.zeros(max_size, dtype=torch.float32)
        self.values = torch.zeros(max_size, dtype=torch.float32)

    def store(self, state, action, reward, done, log_prob, value):
        """
        Store a transition in the buffer.
        Args:
            state: The observed state.
            action: The taken action.
            reward: The received reward.
            done: Whether the episode ended.
            log_prob: Log probability of the action (for PPO).
            value: State value predicted by the model.
        """
        if self.ptr < self.max_size:
            self.states[self.ptr] = torch.tensor(state, dtype=torch.float32)
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.dones[self.ptr] = done
            self.log_probs[self.ptr] = log_prob
            self.values[self.ptr] = value

            self.ptr += 1
            self.size = min(self.size + 1, self.max_size)

    def clear(self):
        """
        Clear the buffer for the next episode.
        """
        self.ptr = 0
        self.size = 0


def compute_advantages(rewards, values, dones, gamma, lam):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return torch.tensor(advantages, dtype=torch.float32), torch.tensor(
        returns, dtype=torch.float32
    )


def ppo_loss(policy_net, states, actions, old_log_probs, advantages, returns, epsilon):
    # Forward pass through deep model
    logits, values = policy_net(states)
    probs = torch.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    new_log_probs = dist.log_prob(actions)

    # Policy loss
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

    # Value loss
    value_loss = nn.MSELoss()(values.squeeze(), returns)

    # Entropy regularization
    entropy_loss = dist.entropy().mean()

    total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss
    return total_loss


def train_ppo(
    env,
    policy_net,
    buffer,
    optimizer,
    max_episodes=10,
    gamma=0.1,
    lam=1,
    epochs=10,
    batch_size=16,
    epsilon=0.5,
):
    """Train the policy network using Proximal Policy Optimization (PPO)."""
    for _ in range(max_episodes):
        state = env.reset()
        done = False

        # Collect trajectories for the episode
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            # Get action, log probability, and value estimate from the model
            action, log_prob, value = policy_net.get_action_and_value(state_tensor)

            # Step the environment
            next_state, reward, done, _ = env.step(action.item())

            # Store transition in the replay buffer
            buffer.store(
                state, action.item(), reward, done, log_prob.item(), value.item()
            )
            state = next_state

        # Compute advantages and returns after the episode
        advantages, returns = compute_advantages(
            rewards=buffer.rewards,
            values=buffer.values,
            dones=buffer.dones,
            gamma=gamma,
            lam=lam,
        )

        # Prepare data for PPO training
        states = torch.tensor(buffer.states, dtype=torch.float32)
        actions = torch.tensor(buffer.actions, dtype=torch.int64)
        old_log_probs = torch.tensor(buffer.log_probs, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        # Train policy and value function using mini-batches
        for _ in range(epochs):
            # Shuffle and create mini-batches
            dataset = torch.utils.data.TensorDataset(
                states, actions, old_log_probs, advantages, returns
            )
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True
            )

            for mini_batch in loader:
                mb_states, mb_actions, mb_old_log_probs, mb_advantages, mb_returns = (
                    mini_batch
                )

                # Compute the PPO loss
                loss = ppo_loss(
                    policy_net,
                    mb_states,
                    mb_actions,
                    mb_old_log_probs,
                    mb_advantages,
                    mb_returns,
                    epsilon,
                )

                # Update the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Clear the buffer after training
        buffer.clear()
