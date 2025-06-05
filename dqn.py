import functools
import logging
import os
import sys
import time
from collections.abc import Callable

import ale_py
import chex
import flashbax as fbx
import gymnasium
import jax
import jax.numpy as jnp
import jax.random as jr
import mlflow
import numpy as np
import optax
from flax import nnx
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import (
    AtariPreprocessing,
    FrameStackObservation,
    RecordEpisodeStatistics,
)

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@chex.dataclass
class Transition:
    obs: jax.Array  # Expects CHW format (stack, height, width)
    action: jax.Array  # If batched, (batch_size,), else scalar
    reward: jax.Array  # If batched, (batch_size,), else scalar
    next_obs: jax.Array  # Expects CHW format
    done: jax.Array  # If batched, (batch_size,), else scalar


class DQN(nnx.Module):
    def __init__(self, n_actions: int, rngs: nnx.Rngs) -> None:
        # Input to __call__ is expected to be NHWC: (batch_size, 84, 84, 4)
        self.conv1 = nnx.Conv(
            in_features=4,  # Number of stacked frames
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
        self.dense1 = nnx.Linear(in_features=7 * 7 * 64, out_features=512, rngs=rngs)
        self.dense2 = nnx.Linear(in_features=512, out_features=n_actions, rngs=rngs)

    def __call__(self, obs_nhwc: jax.Array) -> jax.Array:
        # obs_nhwc shape: (batch_size, 84, 84, 4), dtype: float32 (normalized)
        chex.assert_shape(obs_nhwc, (None, 84, 84, 4))

        x = obs_nhwc  # Already normalized and in NHWC
        x = nnx.relu(self.conv1(x))
        x = nnx.relu(self.conv2(x))
        x = nnx.relu(self.conv3(x))
        x = x.reshape((x.shape[0], -1))  # flatten, preserve batch dimension
        x = nnx.relu(self.dense1(x))
        x = self.dense2(x)
        return x


def loss_fn(
    q_network: DQN,
    target_q_network: DQN,
    batch: Transition,
    gamma: float,
) -> jax.Array:
    # Observations in batch are CHW, convert to NHWC and normalize
    obs_nhwc = (batch.obs.astype(jnp.float32) / 255.0).transpose(0, 2, 3, 1)
    next_obs_nhwc = (batch.next_obs.astype(jnp.float32) / 255.0).transpose(0, 2, 3, 1)

    q_pred_all_actions = q_network(obs_nhwc)
    q_pred = jnp.take_along_axis(
        q_pred_all_actions, batch.action[:, None], axis=1
    ).squeeze(axis=1)

    q_next_target_all_actions = target_q_network(next_obs_nhwc)
    q_next_target_max = jnp.max(q_next_target_all_actions, axis=1)

    q_target = batch.reward + gamma * q_next_target_max * (
        1.0 - batch.done.astype(jnp.float32)
    )
    q_target = jax.lax.stop_gradient(q_target)

    loss = jnp.mean(optax.huber_loss(q_pred, q_target))
    return loss


def decide_action(
    q_network: DQN,
    obs_chw: jax.Array,  # Single observation (C, H, W)
    key: jr.PRNGKey,
    epsilon: float,
    n_actions: int,
    eval_mode: bool,
) -> jax.Array:
    """Selects an action for a single observation using an epsilon-greedy policy."""
    explore_key, random_action_key = jr.split(key)

    def explore_fn() -> jax.Array:
        return jr.randint(random_action_key, (), 0, n_actions, dtype=jnp.int32)

    def exploit_fn() -> jax.Array:
        # Normalize, add batch dim, transpose NCHW to NHWC
        obs_nhwc = (jnp.expand_dims(obs_chw, 0).astype(jnp.float32) / 255.0).transpose(
            0, 2, 3, 1
        )
        q_values = q_network(obs_nhwc)
        return jnp.argmax(q_values[0]).astype(jnp.int32)

    action = jax.lax.cond(
        jnp.logical_and(jnp.logical_not(eval_mode), jr.uniform(explore_key) < epsilon),
        explore_fn,
        exploit_fn,
    )
    return action


def train_step(
    opt_graphdef: nnx.GraphDef,
    target_graphdef: nnx.GraphDef,
    opt_state: nnx.State,
    target_state: nnx.State,
    batch: Transition,
    gamma: float,
) -> tuple[jax.Array, nnx.State]:
    opt = nnx.merge(opt_graphdef, opt_state)
    target_network = nnx.merge(target_graphdef, target_state)
    q_network = opt.model

    loss, grads = nnx.value_and_grad(loss_fn)(q_network, target_network, batch, gamma)

    grads = jax.lax.pmean(grads, axis_name="devices")
    loss = jax.lax.pmean(loss, axis_name="devices")

    opt.update(grads)
    return loss, nnx.state(opt)


def update_target(
    opt_graphdef: nnx.GraphDef,
    target_graphdef: nnx.GraphDef,
    opt_state: nnx.State,
    target_state: nnx.State,
) -> nnx.State:
    target_network = nnx.merge(target_graphdef, target_state)

    opt = nnx.merge(opt_graphdef, opt_state)
    q_network = opt.model
    q_params = nnx.state(q_network, nnx.Param)

    nnx.update(target_network, q_params)
    return nnx.state(target_network)


def shard_pytree(
    pytree: nnx.State,
    n_devices: int,
) -> nnx.State:
    def shard_leaf(leaf: nnx.State) -> nnx.State:
        return leaf.reshape((n_devices, -1) + leaf.shape[1:])

    return jax.tree_util.tree_map(shard_leaf, pytree)


if __name__ == "__main__":
    mlflow.set_experiment("dqn_optimized_async_vec_env")
    validate: bool = True
    eval_mode: bool = False

    if validate:
        chex.enable_asserts()
    else:
        chex.disable_asserts()

    simulate_n_devices: int = 4
    os.environ["XLA_FLAGS"] = (
        f"--xla_force_host_platform_device_count={simulate_n_devices}"
    )
    n_devices = jax.local_device_count()
    logger.info(f"Running on {n_devices} devices.")

    seed: int = 42
    key = jr.PRNGKey(seed)

    n_total_steps: int = 5_000_000
    update_freq: int = 10_000
    batch_size: int = 32
    global_batch_size: int = batch_size * n_devices
    gamma: float = 0.99
    frame_skip: int = 4

    buffer_size: int = 1_000_000
    buffer_start_size: int = 50_000

    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay_steps: int = 1_000_000
    epsilon_decay_rate: float = (epsilon_start - epsilon_end) / epsilon_decay_steps
    epsilon: float = epsilon_start

    lr: float = 0.00025
    momentum: float = 0.95
    opt_eps: float = 0.0001

    n_envs: int = 8
    env_id_base = "ALE/Breakout-v5"

    logger.info(f"Initializing {n_envs} parallel environments using AsyncVectorEnv...")

    def make_env(env_idx: int, env_seed: int) -> Callable[[], gymnasium.Env]:
        def init() -> gymnasium.Env:
            env = gymnasium.make(env_id_base, frameskip=1)
            env = AtariPreprocessing(
                env,
                frame_skip=frame_skip,
                screen_size=84,
                grayscale_obs=True,
                scale_obs=False,
                terminal_on_life_loss=True,
            )
            env = FrameStackObservation(env, stack_size=4)
            env = RecordEpisodeStatistics(env)
            env.action_space.seed(env_seed)
            return env

        return init

    env_fns = [make_env(i, seed + i) for i in range(n_envs)]
    env = AsyncVectorEnv(env_fns)

    n_actions = env.single_action_space.n
    single_obs_shape = env.single_observation_space.shape  # CHW
    obs_dtype = env.single_observation_space.dtype

    logger.info(
        f"Action space: {n_actions}, Observation shape (single env): {single_obs_shape} {obs_dtype}"
    )

    key, q_key, target_key = jr.split(key, 3)
    q_network = DQN(n_actions, rngs=nnx.Rngs(q_key))
    target_network = DQN(n_actions, rngs=nnx.Rngs(target_key))
    q_params = nnx.state(q_network, nnx.Param)
    nnx.update(target_network, q_params)

    opt = nnx.Optimizer(
        q_network,
        optax.rmsprop(
            learning_rate=lr,
            momentum=momentum,
            eps=opt_eps,
        ),
    )

    opt_graphdef, opt_state = nnx.split(opt)
    target_graphdef, target_state = nnx.split(target_network)

    logger.info("Replicating model and optimizer states to devices...")
    replicated_opt_state = jax.device_put_replicated(
        opt_state,
        jax.local_devices(),
    )
    replicated_target_state = jax.device_put_replicated(
        target_state,
        jax.local_devices(),
    )

    decide_action: Callable = jax.vmap(
        decide_action,
        in_axes=(None, 0, 0, None, None, None),
        out_axes=0,
    )  # axes indicates the axis along which the function is mapped for that variable
    decide_action: Callable[
        [DQN, jax.Array, jr.PRNGKey, float, int, bool],
        jax.Array,
    ] = nnx.jit(
        decide_action,
        static_argnames=("n_actions", "eval_mode"),
    )

    train_step: Callable[
        [nnx.State, nnx.State, Transition, float],
        tuple[jax.Array, nnx.State],
    ] = jax.pmap(
        functools.partial(train_step, opt_graphdef, target_graphdef),
        axis_name="devices",
        static_broadcasted_argnums=(3,),  # replicate instead of sharding
    )

    update_target: Callable[
        [nnx.State, nnx.State],
        nnx.State,
    ] = jax.pmap(
        functools.partial(update_target, opt_graphdef, target_graphdef),
        axis_name="devices",
    )

    logger.info(f"Initializing replay buffer with size {buffer_size} and start size {buffer_start_size}...")  # fmt: skip

    dummy_transition = Transition(
        obs=jnp.zeros(single_obs_shape, dtype=obs_dtype),
        action=jnp.zeros((), dtype=jnp.int32),
        reward=jnp.zeros((), dtype=jnp.float32),
        next_obs=jnp.zeros(single_obs_shape, dtype=obs_dtype),
        done=jnp.zeros((), dtype=bool),
    )
    buffer = fbx.make_flat_buffer(
        max_length=buffer_size,
        min_length=buffer_start_size,
        sample_batch_size=global_batch_size,
        add_sequences=False,
        add_batch_size=n_envs,
    )
    buffer = buffer.replace(
        init=jax.jit(buffer.init),
        add=jax.jit(buffer.add, donate_argnums=0),
        sample=jax.jit(buffer.sample),
        can_sample=jax.jit(buffer.can_sample),
    )
    buffer_state = buffer.init(dummy_transition)

    logger.info("Starting training loop...")
    obss, infos = env.reset(seed=[seed + i for i in range(n_envs)])
    obss = jnp.array(obss)

    total_env_frames = 0
    loop_start_time = time.time()

    for step in range(1, n_total_steps // n_envs + 1):
        key, sample_key, train_key = jr.split(key, 3)
        action_keys: jax.Array = jr.split(sample_key, n_envs)

        actions: jax.Array = decide_action(
            opt.model,  # This is the q_network
            obss,
            action_keys,
            epsilon,
            n_actions,
            eval_mode,
        )
        epsilon = max(
            epsilon_end,
            epsilon_start - epsilon_decay_rate * total_env_frames,
        )

        next_obss: np.ndarray
        rewards: np.ndarray
        terminates: np.ndarray
        truncs: np.ndarray
        infos: dict
        next_obss, rewards, terminates, truncs, infos = env.step(
            jax.device_get(actions)
        )
        clip_rewards = np.clip(rewards, -1.0, 1.0)
        dones = np.logical_or(terminates, truncs)

        next_obss: jax.Array = jnp.array(next_obss)
        rewards: jax.Array = jnp.array(clip_rewards, dtype=jnp.float32)
        dones: jax.Array = jnp.array(dones, dtype=bool)

        buffer_state = buffer.add(
            buffer_state,
            Transition(
                obs=obss,
                action=actions,
                reward=rewards,
                next_obs=next_obss,
                done=dones,
            ),
        )
        obss = next_obss

        total_env_frames += n_envs

        for i in range(n_envs):
            if dones[i]:
                ep_ret = infos["episode"]["r"]
                ep_len = infos["episode"]["l"]
                mlflow.log_metric(
                    f"episode_return_env_{i}",
                    float(ep_ret[i].item()),
                    step=total_env_frames,
                )
                mlflow.log_metric(
                    f"episode_length_env_{i}",
                    float(ep_len[i].item()),
                    step=total_env_frames,
                )

        if buffer.can_sample(buffer_state):
            transitions = buffer.sample(
                buffer_state,
                train_key,
            ).experience.first
            transitions_sharded = shard_pytree(transitions, n_devices)

            loss_sharded, replicated_opt_state = train_step(
                replicated_opt_state,
                replicated_target_state,
                transitions_sharded,
                gamma,
            )
            loss: float = jax.device_get(loss_sharded[0])  # Get loss from one device

            opt_state: nnx.State = jax.tree_util.tree_map(
                lambda x: x[0], replicated_opt_state
            )
            nnx.update(opt, opt_state)

            if (
                total_env_frames // update_freq
                > (total_env_frames - n_envs) // update_freq
            ):
                replicated_target_state = update_target(
                    replicated_opt_state, replicated_target_state
                )
                logger.info(f"Target network updated at total_env_frames ~{total_env_frames}.")  # fmt: skip
                mlflow.log_metric("loss", float(loss), step=total_env_frames)
                mlflow.log_metric("epsilon", float(epsilon), step=total_env_frames)

        if step % (max(1, 1000 // n_envs)) == 0:
            steps_per_second = total_env_frames / (time.time() - loop_start_time + 1e-6)
            logger.info(f"Frames: {total_env_frames}, Steps: {step}, Steps/s: {steps_per_second:.2f}, Eps: {epsilon:.3f}")  # fmt: skip
            if buffer.can_sample(buffer_state) and "loss" in locals():
                logger.info(f"Latest Loss: {loss:.4f}")

    env.close()
    logger.info("Training finished.")
    final_time = time.time() - loop_start_time
    logger.info(f"Total training time: {final_time:.2f} seconds for {total_env_frames} environment frames.")  # fmt: skip
    mlflow.log_metric("total_training_time_seconds", final_time)
    mlflow.log_metric("total_environment_frames", total_env_frames)
