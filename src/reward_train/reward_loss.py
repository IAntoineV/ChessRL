"""
We want to implement different methods that use probability shaping based on rewards.

For instance,  given a position :
* we get the policy of our model in this position
* we sample moves based on the policy (play 1 move from the starting position)
* We compute stockfish evaluation for each resulting position
* We tell the model how to update its policy based on the rewards. (What is implemented here)
"""
import numpy as np
import torch


class MoveTraining:
    @staticmethod
    def compute_loss_bestmove(policy, indexes_evaluated, rewards):
        """
            :param policy: logits of shape (b,policy_length)
            :param indexes_evaluated: (b,G)
            :param rewards: (b,G)
            :return:
        """
        b,G = rewards.shape
        best_evaluated_index = np.argmax(rewards, axis=-1)
        best_index = indexes_evaluated[np.arange(b), best_evaluated_index]
        best_move_policy = torch.zeros_like(policy)
        best_move_policy[np.arange(b), best_index] = 1
        loss = torch.nn.functional.cross_entropy(policy, best_move_policy)
        return loss, {}

    @staticmethod
    def compute_loss_grpo(policy, policy_ref, indexes_evaluated, rewards, epsilon=0.2, stable_eps = 1e-6, kl_coef=e1-5):
        """
            :param policy: logits of shape (b,policy_length)
            :param policy_ref: logits of shape (b,policy_length)
            :param indexes_evaluated: (b,G)
            :param rewards: (b,G)
            :return:
        """
        # Convert logits to log-probabilities
        current_log_probs = torch.log_softmax(policy, dim=-1)
        ref_log_probs = torch.log_softmax(policy_ref, dim=-1)

        current_logp = current_log_probs.gather(-1, (indexes_evaluated))
        ref_logp = ref_log_probs.gather(-1, indexes_evaluated)

        # Calculate probability ratio using log-space stability
        log_ratio = current_logp - ref_logp
        ratio = torch.exp(log_ratio)

        advantage = (rewards - np.mean(rewards, axis=-1, keepdims=True)) / (np.std(rewards, axis=-1, keepdims=True) + stable_eps)
        advantage = torch.from_numpy(advantage).to(device=ratio.device)
        # Policy gradient loss with clipping
        clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
        surrogate = torch.min(ratio * advantage, clipped_ratio * advantage)

        kl_term = torch.kl_div()
        loss = -surrogate.mean()

        # Diagnostic metrics
        info = {
            "avg_ratio": ratio.mean().item(),
            "clip_frac": ((ratio < (1 - epsilon)) | (ratio > (1 + epsilon))).float().mean().item(),
            "avg_logp_diff": ((2*(advantage>0)-1) * log_ratio).mean().item(),
            "surrogate": surrogate.mean().item()
        }

        return loss, info


if __name__ == "__main__":
    policy = torch.log(torch.Tensor([
        [0.3,0.2,0.5], [0.75,0.2,0.05]
    ])) # (2,3)

    policy_ref = torch.log(torch.Tensor([
        [0.3, 0.1, 0.6], [0.65, 0.3, 0.05]
    ]))  # (2,3)

    indexes_evaluated = np.array([
        [0,1,0,2], [0,1,0,0]
    ]) # (2,4)

    rewards = np.array([
        [100,10,100,1000], [15,20,15,15]
    ]) # (2,4)

    loss = MoveTraining.compute_loss_bestmove(policy, indexes_evaluated, rewards)
    loss_grpo = MoveTraining.compute_loss_grpo(policy, policy_ref, indexes_evaluated, rewards)



