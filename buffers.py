import collections

import jax
import jax.numpy as jnp
import jax.random as jr

Transition = collections.namedtuple(
    "Transition",
    ("obs", "action", "reward", "next_obs", "done"),
)


class ReplayBuffer:
    """
    JAX-based replay buffer for Transition namedtuple.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple,
        obs_dtype: jnp.dtype,
    ) -> None:
        """
        Args:
            capacity (int): Maximum number of transitions to store.
            obs_shape (tuple): Shape of a single observation.
            obs_dtype: dtype of the observation (default: jnp.uint8).
        """
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype

        self._obs = jnp.zeros((capacity, *obs_shape), dtype=obs_dtype)
        self._actions = jnp.zeros((capacity,), dtype=jnp.int32)
        self._rewards = jnp.zeros((capacity,), dtype=jnp.float32)
        self._next_obs = jnp.zeros((capacity, *obs_shape), dtype=obs_dtype)
        self._dones = jnp.zeros((capacity,), dtype=jnp.bool_)

        self._size = 0
        self._pos = 0

    def add(
        self,
        obs: jnp.ndarray,
        action: int,
        reward: float,
        next_obs: jnp.ndarray,
        done: bool,
    ) -> None:
        """
        Add a transition to the buffer as Transition namedtuple.

        Args:
            obs: Observation, shape obs_shape.
            action: int.
            reward: float.
            next_obs: Observation, shape obs_shape.
            done: bool.
        """
        # Use JAX device_put to ensure data is on device
        self._obs = self._obs.at[self._pos].set(jnp.array(obs, dtype=self.obs_dtype))
        self._actions = self._actions.at[self._pos].set(
            jnp.array(action, dtype=jnp.int32)
        )
        self._rewards = self._rewards.at[self._pos].set(
            jnp.array(reward, dtype=jnp.float32)
        )
        self._next_obs = self._next_obs.at[self._pos].set(
            jnp.array(next_obs, dtype=self.obs_dtype)
        )
        self._dones = self._dones.at[self._pos].set(jnp.array(done, dtype=jnp.bool_))

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    @jax.jit
    def sample(
        self,
        batch_size: int,
        key: jnp.ndarray,
    ) -> Transition:
        """
        Sample a batch of Transition namedtuple.

        Args:
            batch_size (int): Number of transitions to sample.
            key: jax.random.PRNGKey

        Returns:
            dict of batched transitions.
        """
        idxs = jr.randint(key, (batch_size,), 0, self._size)
        batch = Transition(
            obs=self._obs[idxs],
            action=self._actions[idxs],
            reward=self._rewards[idxs],
            next_obs=self._next_obs[idxs],
            done=self._dones[idxs],
        )
        return batch

    @property
    def size(self):
        return self._size

    def __len__(self):
        return self._size
