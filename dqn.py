import logging
import sys

import ale_py
import chex
import gymnasium
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from flax import nnx
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

from buffers import ReplayBuffer, Transition

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DQN(nnx.Module):
    def __init__(self, n_actions: int, rngs: nnx.Rngs) -> None:
        # Input: (batch_size, 84, 84, 4)
        self.conv1 = nnx.Conv(
            in_features=4,
            out_features=32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            rngs=rngs,
        )
        self.conv2 = nnx.Conv(
            in_features=32,
            out_features=64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            rngs=rngs,
        )
        self.conv3 = nnx.Conv(
            in_features=64,
            out_features=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            rngs=rngs,
        )
        # After conv3, output shape for 84x84 input:
        # Conv1: (84-8)/4 + 1 = 19 + 1 = 20 -> (20, 20, 32)
        # Conv2: (20-4)/2 + 1 = 8 + 1 = 9 -> (9, 9, 64)
        # Conv3: (9-3)/1 + 1 = 6 + 1 = 7 -> (7, 7, 64)
        self.dense1 = nnx.Linear(in_features=7 * 7 * 64, out_features=512, rngs=rngs)
        self.dense2 = nnx.Linear(in_features=512, out_features=n_actions, rngs=rngs)

    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        # obs shape: (batch_size, 84, 84, 4), dtype: uint8
        chex.assert_shape(obs, (None, 84, 84, 4))  # Allow for batch dimension

        # Normalize observations
        x = obs.astype(jnp.float32) / 255.0

        x = nnx.relu(self.conv1(x))
        x = nnx.relu(self.conv2(x))
        x = nnx.relu(self.conv3(x))
        x = x.reshape((x.shape[0], -1))  # flatten, preserve batch dimension
        x = nnx.relu(self.dense1(x))
        x = self.dense2(x)
        return x


def loss_fn(q_network: DQN, target_q_network: DQN, batch: Transition, gamma: float):
    q_pred_all_actions = q_network(batch.obs.transpose(0, 2, 3, 1))  # NHWC
    q_pred = jnp.take_along_axis(
        q_pred_all_actions, batch.action[:, None], axis=1
    ).squeeze(axis=1)

    q_next_target_all_actions = target_q_network(
        batch.next_obs.transpose(0, 2, 3, 1)  # NHWC
    )
    q_next_target_max = jnp.max(q_next_target_all_actions, axis=1)

    q_target = batch.reward + gamma * q_next_target_max * (1.0 - batch.done.astype(jnp.float32))  # fmt: skip
    q_target = jax.lax.stop_gradient(q_target)

    loss = jnp.mean(optax.huber_loss(q_pred, q_target))
    return loss


if __name__ == "__main__":
    validate: bool = True
    eval_mode: bool = False

    if validate:
        chex.enable_asserts()
    else:
        chex.disable_asserts()

    # start: hyperparameters
    seed: int = 42
    key = jr.PRNGKey(seed)

    n_steps: int = 50_000  # _000
    target_update_freq: int = 10  # _000
    batch_size: int = 32
    gamma: float = 0.99
    frame_skip: int = 4

    replay_size: int = 1_000  # _000
    replay_start_size: int = 50  # _000

    epsilon: float = 1.0
    epsilon_end: float = 0.1
    epsilon_lifetime: int = 1_000  # _000
    epsilon_decay: float = (epsilon_end - epsilon) / epsilon_lifetime

    lr: float = 0.00025
    momentum: float = 0.95
    opt_eps: float = 0.01
    # end: hyperparameters

    env_id = "ALE/Breakout-v5"
    env = gymnasium.make(env_id, frameskip=1)
    env = AtariPreprocessing(
        env,
        frame_skip=frame_skip,
        screen_size=84,
        grayscale_obs=True,
        scale_obs=False,
        terminal_on_life_loss=True,
    )
    env = FrameStackObservation(env, stack_size=4)

    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape  # CHW
    obs_dtype = env.observation_space.dtype

    key, q_key, target_key = jr.split(key, 3)
    q_network = DQN(n_actions, rngs=nnx.Rngs(params=q_key))
    target_q_network = DQN(n_actions, rngs=nnx.Rngs(params=target_key))

    q_params = nnx.state(q_network, nnx.Param)
    nnx.update(target_q_network, q_params)

    opt = nnx.Optimizer(
        q_network,
        optax.rmsprop(learning_rate=lr, momentum=momentum, eps=opt_eps, centered=True),
    )

    replay_buffer = ReplayBuffer(
        capacity=replay_size,
        obs_shape=obs_shape,
        obs_dtype=obs_dtype,
    )

    obs, _ = env.reset(seed=seed)
    obs = jnp.array(obs)
    episode_return = 0.0

    # Main loop of (1) Sampling and (2) Training
    for step in range(n_steps):
        # Sampling
        key, decision_key = jr.split(key)
        if not eval_mode and jr.uniform(decision_key) < epsilon:
            action = jr.randint(decision_key, (), 0, n_actions).item()
        else:
            obs_batched = jnp.expand_dims(obs, 0)
            q_values = q_network(obs_batched.transpose(0, 2, 3, 1))  # NHWC
            action = jnp.argmax(q_values[0]).item()

        epsilon = max(epsilon_end, epsilon + epsilon_decay)

        next_obs, reward_val, terminated, truncated, _ = env.step(action)
        next_obs = jnp.array(next_obs)

        clipped_reward = jnp.clip(reward_val, -1.0, 1.0).item()

        done = terminated or truncated

        replay_buffer.add(
            obs,
            action,
            clipped_reward,
            next_obs,
            done,
        )

        obs = next_obs

        episode_return += clipped_reward

        if done:
            obs, _ = env.reset()
            obs = jnp.array(obs)

            logger.info(f"Episode finished at step {step}: return={episode_return:.4f}")
            episode_return = 0.0

        # Training
        if step >= replay_start_size:
            key, sample_key = jr.split(key)
            batch: Transition = replay_buffer.sample(batch_size, sample_key)

            loss, grads = nnx.value_and_grad(loss_fn)(
                q_network, target_q_network, batch, gamma
            )
            opt.update(grads)

            if step % target_update_freq == 0:
                q_params = nnx.state(q_network, nnx.Param)
                nnx.update(target_q_network, q_params)
                logger.info(f"Step {step}: loss={loss:.4f}, epsilon={epsilon:.3f}")

    env.close()
    logger.info("Training finished")
