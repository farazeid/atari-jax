import collections
import logging
import random
import sys
import numpy as np # Add numpy import

import ale_py
import chex
import flashbax as fbx
import gymnasium
import jax
import jax.numpy as jnp
import jax.random as jr
import mlflow
import optax
from flax import nnx
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from gymnasium.vector import AsyncVectorEnv # Import AsyncVectorEnv

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@jax.jit
def preprocess_observations_jax(obs_batch: jnp.ndarray) -> jnp.ndarray:
    """Transpose (N,C,H,W) to (N,H,W,C), convert to float32, and normalize."""
    chex.assert_rank(obs_batch, 4) # Expect (N, C, H, W)
    # Transpose to (N, H, W, C)
    preprocessed_batch = obs_batch.transpose(0, 2, 3, 1)
    # Convert to float32 and normalize
    preprocessed_batch = preprocessed_batch.astype(jnp.float32) / 255.0
    return preprocessed_batch


@chex.dataclass
class Transition:
    obs: jnp.ndarray # Now stores preprocessed obs (N, H, W, C) or (H, W, C) for dummy
    action: int
    reward: float
    next_obs: jnp.ndarray # Now stores preprocessed next_obs
    done: bool


class DQN(nnx.Module):
    def __init__(self, n_actions: int, rngs: nnx.Rngs) -> None:
        # Input is now preprocessed: (batch_size, 84, 84, 4), float32
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

    @nnx.jit
    def __call__(self, preprocessed_obs: jnp.ndarray) -> jnp.ndarray: # Renamed obs to preprocessed_obs
        # preprocessed_obs shape: (batch_size, H, W, C), dtype: float32
        chex.assert_shape(preprocessed_obs, (None, 84, 84, 4))
        chex.assert_type(preprocessed_obs, jnp.float32) # Assert type

        # Normalization is already done
        x = preprocessed_obs

        x = nnx.relu(self.conv1(x))
        x = nnx.relu(self.conv2(x))
        x = nnx.relu(self.conv3(x))
        x = x.reshape((x.shape[0], -1))  # flatten, preserve batch dimension
        x = nnx.relu(self.dense1(x))
        x = self.dense2(x)
        return x


@nnx.jit
def loss_fn(q_network: DQN, target_q_network: DQN, batch: Transition, gamma: float):
    # batch.obs and batch.next_obs are already preprocessed (N, H, W, C)
    q_pred_all_actions = q_network(batch.obs)
    q_pred = jnp.take_along_axis(
        q_pred_all_actions, batch.action[:, None], axis=1
    ).squeeze(axis=1)

    q_next_target_all_actions = target_q_network(batch.next_obs)
    q_next_target_max = jnp.max(q_next_target_all_actions, axis=1)

    q_target = batch.reward + gamma * q_next_target_max * (1.0 - batch.done.astype(jnp.float32))
    q_target = jax.lax.stop_gradient(q_target)

    loss = jnp.mean(optax.huber_loss(q_pred, q_target))
    return loss


# Renamed from select_action to _select_action_single
# This function now processes a single preprocessed observation
def _select_action_single(
    q_network: DQN,
    single_preprocessed_obs: jnp.ndarray, # Renamed, expects (H, W, C), float32
    key: jr.PRNGKey,
    epsilon: float,
    n_actions: int,
    eval_mode: bool,
) -> int:
    """Selects an action using an epsilon-greedy policy for a single preprocessed observation."""
    explore_key, random_action_key = jr.split(key)

    def explore_fn():
        return jr.randint(random_action_key, (), 0, n_actions)

    def exploit_fn():
        # single_preprocessed_obs shape: (H, W, C)
        # Expand dims to (1, H, W, C) for the network
        obs_for_network = jnp.expand_dims(single_preprocessed_obs, 0)
        q_values = q_network(obs_for_network)
        return jnp.argmax(q_values[0]) # q_values shape is (1, n_actions)

    action = jax.lax.cond(
        jnp.logical_and(jnp.logical_not(eval_mode), jr.uniform(explore_key) < epsilon),
        explore_fn,
        exploit_fn,
    )
    return action

# New function for batch processing
@nnx.jit
def select_action_batch(
    q_network: DQN,
    obs_batch: jnp.ndarray, # Batch of observations (N, C, H, W)
    keys_batch: jr.PRNGKey, # Batch of PRNG keys (N, ...)
    epsilon: float,
    n_actions: int,
    eval_mode: bool,
) -> jnp.ndarray: # Returns a batch of actions (N,)
    """Selects actions for a batch of observations using vmap."""
    return jax.vmap(
        _select_action_single, in_axes=(None, 0, 0, None, None, None), out_axes=0
    )(q_network, obs_batch, keys_batch, epsilon, n_actions, eval_mode)

# Core logic for pmap
# The old train_step and update_target_network functions are removed as they are replaced by the _core and pmapped versions.
@nnx.jit
def _train_step_core(
    opt_device: nnx.Optimizer,
    target_q_network_device: DQN,
    sharded_batch: Transition, # This is the batch for a single device
    gamma: float
) -> tuple[jnp.ndarray, nnx.Optimizer]:
    q_network_device = opt_device.model
    loss, grads = nnx.value_and_grad(loss_fn)(
        q_network_device, target_q_network_device, sharded_batch, gamma
    )
    # Synchronize gradients and loss across devices
    grads = jax.lax.pmean(grads, axis_name='devices')
    loss = jax.lax.pmean(loss, axis_name='devices')
    opt_device.update(grads)
    return loss, opt_device

@nnx.jit
def _update_target_network_core(q_network_device: DQN, target_q_network_device: DQN) -> DQN:
    q_params = nnx.state(q_network_device, nnx.Param)
    nnx.update(target_q_network_device, q_params)
    return target_q_network_device

# Create pmapped functions
# Note: For Optimizer, Flax's recommendation for pmap is often to handle optimizer state explicitly
# or ensure the optimizer class itself is pmap-aware. NNX Optimizer might behave differently.
# We assume nnx.Optimizer can be replicated and its update method works correctly under pmap.
# If opt_device.update(grads) within pmap doesn't correctly update the replicated optimizer's state
# across devices (e.g. if opt_state isn't properly managed by pmap), this is a point of failure.
pmapped_train_step = jax.pmap(
    _train_step_core,
    axis_name='devices',
    in_axes=(0, 0, 0, None), # opt_device, target_q_network_device, sharded_batch are sharded, gamma is not
    out_axes=0 # loss and opt_device are sharded
)

pmapped_update_target_network = jax.pmap(
    _update_target_network_core,
    axis_name='devices',
    in_axes=(0, 0), # q_network_device, target_q_network_device are sharded
    out_axes=0 # target_q_network_device is sharded
)

# Old train_step and update_target_network functions are removed.

if __name__ == "__main__":
    # jax.disable_jit()
    mlflow.set_experiment("dqn")

    # Device Configuration
    n_devices = jax.local_device_count()
    logger.info(f"Number of available devices: {n_devices}")
    if n_devices == 0:
        raise ValueError("No JAX devices found. Ensure JAX is installed with appropriate backend (CPU/GPU/TPU).")


    validate: bool = True
    eval_mode: bool = False

    if validate:
        chex.enable_asserts()
    else:
        chex.disable_asserts()

    # start: hyperparameters
    seed: int = 42
    key = jr.PRNGKey(seed)

    n_steps: int = 50_000_000
    target_update_freq: int = 10_000
    batch_size: int = 32
    gamma: float = 0.99
    frame_skip: int = 4

    replay_size: int = 1_000_000
    replay_start_size: int = 5_000

    epsilon: float = 1.0
    epsilon_end: float = 0.1
    epsilon_lifetime: int = 1_000_000
    # epsilon_decay will be calculated based on global_step, so direct pre-calculation is removed.
    initial_epsilon: float = epsilon # Store initial epsilon

    lr: float = 0.00025
    momentum: float = 0.95
    opt_eps: float = 0.01
    grad_clip_norm: float = 1.0 # For gradient clipping

    n_envs: int = 4 # Set n_envs = 4
    # end: hyperparameters

    if batch_size % n_devices != 0:
        raise ValueError(f"Batch size ({batch_size}) must be divisible by the number of devices ({n_devices}).")

    env_id = "ALE/Breakout-v5"

    # Create env_fns for AsyncVectorEnv
    def make_env(env_id, seed, frame_skip, terminal_on_life_loss):
        def thunk():
            env = gymnasium.make(env_id, frameskip=1)
            env = AtariPreprocessing(
                env,
                frame_skip=frame_skip,
                screen_size=84,
                grayscale_obs=True,
                scale_obs=False,
                terminal_on_life_loss=terminal_on_life_loss,
            )
            env = FrameStackObservation(env, stack_size=4)
            env.action_space.seed(seed) # Important for reproducibility with vector envs
            return env
        return thunk

    env_fns = [make_env(env_id, seed + i, frame_skip, True) for i in range(n_envs)]
    env = AsyncVectorEnv(env_fns)


    # For a single environment, to get shape and dtype for raw observations
    _single_env_for_shape = gymnasium.make(env_id, frameskip=1)
    _single_env_for_shape = AtariPreprocessing(_single_env_for_shape, frame_skip=frame_skip, screen_size=84, grayscale_obs=True, scale_obs=False, terminal_on_life_loss=True)
    _single_env_for_shape = FrameStackObservation(_single_env_for_shape, stack_size=4)
    n_actions = _single_env_for_shape.action_space.n
    raw_single_obs_shape = _single_env_for_shape.observation_space.shape # (C, H, W) e.g. (4, 84, 84)
    # raw_obs_dtype = _single_env_for_shape.observation_space.dtype # uint8
    _single_env_for_shape.close()

    # Define shape for preprocessed observations (H, W, C)
    preprocessed_single_obs_shape = (raw_single_obs_shape[1], raw_single_obs_shape[2], raw_single_obs_shape[0])
    # obs_shape for the vectorized environment (raw observations)
    # raw_obs_batch_shape = (n_envs, *raw_single_obs_shape)
    # preprocessed_obs_batch_shape = (n_envs, *preprocessed_single_obs_shape)


    key, q_key, target_key = jr.split(key, 3)
    q_network = DQN(n_actions, rngs=nnx.Rngs(params=q_key))
    target_q_network = DQN(n_actions, rngs=nnx.Rngs(params=target_key))

    q_params_host = nnx.state(q_network, nnx.Param) # Get state from host q_network
    nnx.update(target_q_network, q_params_host) # Update host target_q_network

    optimizer_chain = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.rmsprop(learning_rate=lr, momentum=momentum, eps=opt_eps, centered=True)
    )
    opt_host = nnx.Optimizer(
        q_network, # This is q_network_host
        optimizer_chain,
    )

    # Replicate optimizer and target network to all devices
    devices = jax.local_devices()
    opt_replicated = jax.device_put_replicated(opt_host, devices)
    target_q_network_replicated = jax.device_put_replicated(target_q_network, devices)


    replay_buffer = fbx.make_flat_buffer(
        max_length=replay_size,
        min_length=replay_start_size,
        sample_batch_size=batch_size,
        add_sequences=False,  # Add individual transitions, not sequences
        add_batch_size=n_envs if n_envs > 1 else None,
    )
    replay_buffer = replay_buffer.replace(
        init=jax.jit(replay_buffer.init),
        add=jax.jit(replay_buffer.add, donate_argnums=0),
        sample=jax.jit(replay_buffer.sample),
        can_sample=jax.jit(replay_buffer.can_sample),
    )
    dummy_transition = Transition(
        obs=jnp.zeros(preprocessed_single_obs_shape, dtype=jnp.float32), # Use preprocessed shape and float32 type
        action=0,
        reward=0.0,
        next_obs=jnp.zeros(preprocessed_single_obs_shape, dtype=jnp.float32), # Use preprocessed shape and float32 type
        done=False,
    )
    replay_buffer_state = replay_buffer.init(dummy_transition)

    raw_obs, info = env.reset(seed=seed) # Get raw observations from env
    obs = preprocess_observations_jax(jnp.array(raw_obs)) # Preprocess them, obs is now (N,H,W,C), float32

    # Initialize episode returns for each environment
    episode_returns = jnp.zeros(n_envs, dtype=jnp.float32)


    # Main loop of (1) Sampling and (2) Training
    # Loop over agent steps, but consider n_envs for total interactions
    for step in range(1, (n_steps // n_envs) + 1):
        global_step = step * n_envs

        # Calculate current epsilon based on global_step
        epsilon = max(epsilon_end, initial_epsilon - (initial_epsilon - epsilon_end) * (global_step / epsilon_lifetime))

        # Sampling
        key, action_key_base = jr.split(key)
        action_keys = jr.split(action_key_base, n_envs)

        # obs is already preprocessed: (N, H, W, C), float32 from the previous iteration or initial reset
        # opt.model here refers to opt_host.model as this part is on the host
        actions = select_action_batch(
            opt_host.model, # Correctly using opt_host.model
            obs, # obs is preprocessed
            action_keys,
            epsilon,
            n_actions,
            eval_mode,
        )
        actions_np = np.array(actions)

        raw_next_obs, reward_vals, terminateds, truncateds, infos = env.step(actions_np) # Get raw next_obs

        # Preprocess next_obs and convert other environment outputs to JAX arrays
        next_obs = preprocess_observations_jax(jnp.array(raw_next_obs)) # next_obs is now (N,H,W,C), float32
        reward_vals = jnp.array(reward_vals)
        terminateds = jnp.array(terminateds)
        truncateds = jnp.array(truncateds)

        clipped_rewards = jnp.clip(reward_vals, -1.0, 1.0) # Batch
        dones = jnp.logical_or(terminateds, truncateds) # Batch

        # Add batch of preprocessed transitions to replay buffer
        # obs and next_obs are already preprocessed (N, H, W, C), float32
        replay_buffer_state = replay_buffer.add(
            replay_buffer_state,
            Transition( # Storing preprocessed obs and next_obs
                obs=obs, # current obs, already preprocessed
                action=actions,
                reward=clipped_rewards,
                next_obs=next_obs, # next_obs, now preprocessed
                done=dones,
            ),
        )

        obs = next_obs # Update obs for the next iteration (already preprocessed)
        episode_returns += clipped_rewards

        # Handle episode logging (simpler version for now)
        for i in range(n_envs):
            if dones[i]:
                # Log individual environment returns
                # The step for logging can be global_step or global_step - (n_envs - 1) + i for more precision
                mlflow.log_metric(f"episode_return_env_{i}", episode_returns[i].item(), step=global_step)
                episode_returns = episode_returns.at[i].set(0.0) # Reset return for that env

        # Training
        if replay_buffer.can_sample(replay_buffer_state):
            key, sample_key = jr.split(key)

            # Sample a global batch from the replay buffer
            experience_pair = replay_buffer.sample(replay_buffer_state, sample_key).experience
            current_transition_batch: Transition = experience_pair.first

            # Shard the batch for pmap
            # Each field in current_transition_batch has shape (global_batch_size, ...)
            # We need to reshape it to (n_devices, global_batch_size // n_devices, ...)
            def shard_array(x: jnp.ndarray) -> jnp.ndarray:
                return x.reshape((n_devices, -1) + x.shape[1:])
            sharded_transitions = jax.tree_util.tree_map(shard_array, current_transition_batch)

            # Perform the pmapped training step
            loss_replicated, opt_replicated = pmapped_train_step(
                opt_replicated, target_q_network_replicated, sharded_transitions, gamma
            )

            # Get loss from the first device for logging (they should be identical due to pmean)
            loss_value = jax.device_get(loss_replicated[0])
            mlflow.log_metric("loss", loss_value.item(), step=global_step)


            if global_step % target_update_freq == 0:
                # Extract the replicated model from the replicated optimizer
                # This assumes opt_replicated is a PyTree where each leaf is an Optimizer instance on a device,
                # or jax.tree_map can correctly access .model on the replicated structure.
                # If opt_replicated is a single Optimizer object that manages sharded states internally,
                # then opt_replicated.model might directly give the sharded model.
                # Given NNX structure, jax.device_put_replicated(opt_host, devices) likely creates
                # a list/tuple of Optimizer objects, one per device.
                # If opt_replicated is `List[nnx.Optimizer]`, then this is tricky.
                # Let's assume for NNX, `opt_replicated` itself is the replicated optimizer structure.
                # We need to get the replicated q_network model from it.
                # A common pattern is to have `opt_replicated.model` give the replicated model directly if the optimizer is designed for it.
                # Or, if opt_replicated is a PyTree of optimizers:
                q_network_model_replicated = jax.tree_map(lambda o: o.model, opt_replicated)

                target_q_network_replicated = pmapped_update_target_network(
                    q_network_model_replicated, target_q_network_replicated
                )
                mlflow.log_metric("epsilon", epsilon, step=global_step)

        if global_step % (1000 * n_envs) == 0: # Adjust logging frequency
            logger.info(f"Global Step {global_step}")

    env.close()
    logger.info("Training finished")
