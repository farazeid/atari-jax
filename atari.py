import time

import ale_py
import chex
import flashbax as fbx
import flax.nnx as nnx
import gymnasium
import jax
import jax.numpy as jnp
import mlflow
import optax
from gymnasium.wrappers import (
    AtariPreprocessing,
    FrameStackObservation,
    RecordEpisodeStatistics,
)
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


@chex.dataclass
class Transition:
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    next_obs: jax.Array
    terminated: jax.Array


class JaxWrapper(gymnasium.Env):
    def __init__(self, env: gymnasium.Env) -> None:
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        _ = env.single_observation_space
        self.single_observation_space = gymnasium.spaces.Box(
            low=0,
            high=255,
            shape=(84, 84, 4),
            dtype=_.dtype,
        )
        self.single_action_space = env.single_action_space

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        if obs.shape == (4, 84, 84):
            obs = obs.transpose(1, 2, 0)
        if len(obs.shape) == 4 and obs.shape[1:] == (4, 84, 84):
            obs = obs.transpose(0, 2, 3, 1)
        return jnp.array(obs), info

    def step(self, action: jax.Array):
        next_obs, reward, terminated, truncated, info = self.env.step(
            jax.device_get(action)
        )
        if next_obs.shape == (4, 84, 84):
            next_obs = next_obs.transpose(1, 2, 0)
        if len(next_obs.shape) == 4 and next_obs.shape[1:] == (4, 84, 84):
            next_obs = next_obs.transpose(0, 2, 3, 1)
        return (
            jnp.array(next_obs),
            jnp.array(reward),
            jnp.array(terminated),
            jnp.array(truncated),
            info,
        )


class DQN(nnx.Module):
    def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs) -> None:
        # Input to __call__ is expected to be NHWC: (batch_size, 84, 84, 4)
        self.conv1 = nnx.Conv(
            in_features=in_features,
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
        self.dense2 = nnx.Linear(in_features=512, out_features=out_features, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = nnx.relu(self.conv1(x))
        x = nnx.relu(self.conv2(x))
        x = nnx.relu(self.conv3(x))
        x = x.reshape((x.shape[0], -1))  # flatten, preserve batch dimension
        x = nnx.relu(self.dense1(x))
        x = self.dense2(x)
        return x


@nnx.jit
def decide_action(
    key: jax.random.PRNGKey,
    q_network: DQN,
    obs: jax.Array,
    epsilon: float,
    n_actions: int,
) -> jax.Array:
    epsilon_key, random_action_key = jax.random.split(key)

    def explore_fn() -> jax.Array:
        return jax.random.randint(random_action_key, (1,), 0, n_actions)

    def exploit_fn() -> jax.Array:
        q_values = q_network(obs)
        return q_values.argmax(axis=-1)

    action = jax.lax.cond(
        jax.random.uniform(epsilon_key) < epsilon,
        explore_fn,
        exploit_fn,
    )
    return action


@nnx.jit
def train_step(
    opt: nnx.Optimizer,
    target_network: nnx.Module,
    batch: Transition,
    gamma: float,
) -> jax.Array:
    def loss_fn(model: nnx.Module):
        q_value = model(batch.obs)  # (batch_size, n_actions)
        q_value = jax.vmap(lambda q, a: q[a])(q_value, batch.action)  # (batch_size,)

        target_next = target_network(batch.next_obs).max(axis=-1)  # (batch_size,)
        target_value = batch.reward + (1 - batch.terminated) * gamma * target_next

        return optax.l2_loss(q_value, target_value).mean()

    loss, grads = nnx.value_and_grad(loss_fn)(opt.model)
    opt.update(grads=grads)
    return loss


@nnx.jit
def update_target(
    opt: nnx.Optimizer,
    target_network: nnx.Module,
) -> nnx.Module:
    q_params = nnx.state(opt.model, nnx.Param)
    nnx.update(target_network, q_params)


if __name__ == "__main__":
    seed: int = 1
    key = jax.random.PRNGKey(seed)

    env_id: str = "BreakoutNoFrameskip-v4"
    total_timesteps: int = 10_000_000
    learning_rate: float = 1e-4
    n_envs: int = 1
    buffer_size: int = 1_000_000
    gamma: float = 0.99
    tau: float = 1.0
    target_network_frequency: int = 1_000
    batch_size: int = 32
    epsilon: float = 1
    epsilon_end: float = 0.01
    epsilon_lifetime: int = total_timesteps * 0.1
    epsilon_decay_rate: float = (epsilon - epsilon_end) / epsilon_lifetime
    learning_starts: int = 80_000
    train_frequency: int = 4

    """
    Make environment
    """

    def make_env(env_id: str, seed: int):
        def thunk():
            env = gymnasium.make(env_id)
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)
            env = EpisodicLifeEnv(env)
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)
            env = ClipRewardEnv(env)
            env = gymnasium.wrappers.ResizeObservation(env, (84, 84))
            env = gymnasium.wrappers.GrayscaleObservation(env)
            env = gymnasium.wrappers.FrameStackObservation(env, 4)
            env = gymnasium.wrappers.RecordEpisodeStatistics(env)
            env.action_space.seed(seed)
            return env

        return thunk

    envs = gymnasium.vector.SyncVectorEnv(
        [make_env(env_id, seed + i) for i in range(n_envs)],
        autoreset_mode=gymnasium.vector.AutoresetMode.SAME_STEP,
    )
    envs = JaxWrapper(envs)
    obs_shape = envs.single_observation_space.shape
    obs_dtype = envs.single_observation_space.dtype
    n_actions = envs.single_action_space.n

    """
    Make networks + optimiser
    """

    key, q_key = jax.random.split(key)
    q_network = DQN(
        in_features=obs_shape[-1],
        out_features=n_actions,
        rngs=nnx.Rngs(q_key),
    )
    target_network = DQN(
        in_features=obs_shape[-1],
        out_features=n_actions,
        rngs=nnx.Rngs(q_key),  # same key to initialise identical network parameters
    )

    opt = nnx.Optimizer(
        q_network,
        optax.adam(learning_rate=learning_rate),
    )

    """
    Make replay buffer
    """

    buffer = fbx.make_flat_buffer(
        max_length=buffer_size,
        min_length=buffer_size - 1,
        sample_batch_size=batch_size,
        add_sequences=False,
        add_batch_size=n_envs,
    )
    buffer = buffer.replace(
        init=jax.jit(buffer.init),
        add=jax.jit(buffer.add, donate_argnums=0),
        sample=jax.jit(buffer.sample),
        can_sample=jax.jit(buffer.can_sample),
    )

    dummy_transition = Transition(
        obs=jnp.zeros(obs_shape, dtype=obs_dtype),
        action=jnp.zeros((), dtype=jnp.int32),
        reward=jnp.zeros((), dtype=jnp.float32),
        next_obs=jnp.zeros(obs_shape, dtype=obs_dtype),
        terminated=jnp.zeros((), dtype=bool),
    )

    buffer_state = buffer.init(dummy_transition)

    """
    Training loop
    """

    start_time = time.time()

    obs, _ = envs.reset(seed=seed)
    for step in range(total_timesteps):
        key, sample_key = jax.random.split(key, 2)

        actions = decide_action(key, q_network, obs, epsilon, n_actions)
        epsilon = max(
            epsilon_end,
            epsilon - epsilon_decay_rate,
        )

        next_obs, reward, terminated, truncated, info = envs.step(actions)

        if "final_info" in info:
            # fmt: off
            mlflow.log_metric("charts/episodic_return", info["final_info"]["episode"]["r"], step)
            mlflow.log_metric("charts/episodic_length", info["final_info"]["episode"]["l"], step)
            # fmt: on

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncated):
            if trunc:
                real_next_obs.at[idx].set(info["final_obs"][idx])

        buffer_state = buffer.add(
            buffer_state,
            Transition(
                obs=obs,
                action=actions,
                reward=reward,
                next_obs=real_next_obs,
                terminated=terminated,
            ),
        )

        obs = next_obs

        if step < learning_starts:
            continue

        if step % train_frequency == 0:
            batch = buffer.sample(buffer_state, sample_key).experience.first
            loss = train_step(opt, target_network, batch, gamma)

            # fmt: off
            print(f"Step: {step}, Loss: {loss}, SPS: {int(step / (time.time() - start_time))}")
            mlflow.log_metric("losses/loss", loss, step)
            mlflow.log_metric("charts/SPS", int(step / (time.time() - start_time)), step)
            mlflow.log_metric("charts/epsilon", epsilon, step)
            # fmt: on

        if step % target_network_frequency == 0:
            update_target(opt, target_network)
            print(f"Updated target network at step {step}")

    envs.close()
