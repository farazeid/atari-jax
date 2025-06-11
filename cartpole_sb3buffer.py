import time

import chex
import flax.nnx as nnx
import gymnasium
import jax
import jax.numpy as jnp
import mlflow
import optax
from stable_baselines3.common.buffers import ReplayBuffer


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
        self.single_observation_space = env.single_observation_space
        self.single_action_space = env.single_action_space

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        return jnp.array(obs), info

    def step(self, action: jax.Array):
        next_obs, reward, terminated, truncated, info = self.env.step(
            jax.device_get(action)
        )
        return (
            jnp.array(next_obs),
            jnp.array(reward),
            jnp.array(terminated),
            jnp.array(truncated),
            info,
        )


class DQN(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.linear1 = nnx.Linear(in_features, 120, rngs=rngs)
        self.linear2 = nnx.Linear(120, 84, rngs=rngs)
        self.linear3 = nnx.Linear(84, out_features, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.linear1(x)
        x = jax.nn.relu(x)
        x = self.linear2(x)
        x = jax.nn.relu(x)
        x = self.linear3(x)
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
        q_value = jax.vmap(lambda q, a: q[a])(
            q_value, batch.action
        ).squeeze()  # (batch_size,)

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

    env_id: str = "CartPole-v1"
    n_envs: int = 1
    total_timesteps: int = 500_000
    learning_rate: float = 2.5e-4
    buffer_size: int = 10_000
    gamma: float = 0.99
    tau: float = 1.0
    target_network_frequency: int = 500
    batch_size: int = 128
    epsilon: float = 1.0
    epsilon_end: float = 0.05
    epsilon_lifetime: int = total_timesteps // 2
    epsilon_decay_rate: float = (epsilon - epsilon_end) / epsilon_lifetime
    learning_starts: int = 10_000
    train_frequency: int = 10

    """
    Make environment
    """

    def make_env(env_id: str, seed: int):
        def thunk():
            env = gymnasium.make(env_id)
            env = gymnasium.wrappers.RecordEpisodeStatistics(env)
            env.action_space.seed(seed)
            return env

        return thunk

    envs = gymnasium.vector.SyncVectorEnv(
        [make_env(env_id, seed + i) for i in range(n_envs)],
        autoreset_mode=gymnasium.vector.AutoresetMode.SAME_STEP,
    )
    obs_shape = envs.single_observation_space.shape
    obs_dtype = envs.single_observation_space.dtype
    n_actions = envs.single_action_space.n

    """
    Make networks + optimiser
    """

    key, q_key = jax.random.split(key)
    q_network = DQN(
        in_features=obs_shape[0],
        out_features=n_actions,
        rngs=nnx.Rngs(q_key),
    )
    target_network = DQN(
        in_features=obs_shape[0],
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

    buffer = ReplayBuffer(
        buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        "cpu",
        handle_timeout_termination=False,
    )

    """
    Training loop
    """

    start_time = time.time()

    obs, _ = envs.reset(seed=seed)
    for step in range(total_timesteps):
        key, decision_key, sample_key = jax.random.split(key, 2)

        actions = decide_action(
            decision_key, q_network, jnp.array(obs), epsilon, n_actions
        )
        epsilon = max(
            epsilon_end,
            epsilon - epsilon_decay_rate,
        )

        next_obs, reward, terminated, truncated, info = envs.step(
            jax.device_get(actions)
        )

        if "final_info" in info:
            # fmt: off
            mlflow.log_metric("charts/episodic_return", info["final_info"]["episode"]["r"], step)
            mlflow.log_metric("charts/episodic_length", info["final_info"]["episode"]["l"], step)
            # fmt: on

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncated):
            if trunc:
                real_next_obs[idx] = info["final_obs"][idx]

        buffer.add(
            obs=obs,
            next_obs=real_next_obs,
            action=actions,
            reward=reward,
            done=terminated,
            infos=info,
        )

        obs = next_obs

        if step < learning_starts:
            continue

        if step % train_frequency == 0:
            batch = buffer.sample(batch_size)
            batch = Transition(
                obs=batch.observations.numpy(),
                action=batch.actions.numpy(),
                reward=batch.rewards.flatten().numpy(),
                next_obs=batch.next_observations.numpy(),
                terminated=batch.dones.flatten().numpy(),
            )
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
