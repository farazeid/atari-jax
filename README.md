# Project: JAX Deep Q-Network (DQN) Implementation

## Phase 1: JAX Fundamentals & Basic DQN (Single CPU/GPU)

This phase focuses on getting the core DQN algorithm working in JAX on a single device, understanding basic JAX transformations, and setting up the environment.

- [x] **Task 1.1: JAX Environment Setup**
    - [x] Install JAX with CPU support (or GPU if you have one).
    - [x] Install necessary libraries: `dm-haiku`, `optax`, `chex`, `rlax` (optional, but highly recommended for `DQN` related components like replay buffers, losses).
    - [x] Install `gym` (for Atari environments) and `atari_py` or `ale_py`.
    - [ ] Verify installations with simple JAX code: `print(jax.__version__)`.
- [ ] **Task 1.2: JAX Primitives & Immutability**
    - [ ] Understand JAX's functional programming paradigm and immutability.
    - [ ] Practice basic array manipulation with `jax.numpy`.
    - [ ] Learn about `jax.random` for pseudo-random number generation.
- [ ] **Task 1.3: Neural Network with Haiku**
    - [ ] Define the convolutional neural network architecture as specified in the DQN paper (3 conv layers, 2 dense layers).
    - [ ] Implement this using `haiku.transform`.
    - [ ] Initialize network parameters using ``.
    - [ ] Understand how to manage network states (parameters).
- [ ] **Task 1.4: Optimization with Optax**
    - [ ] Choose and initialize the Adam optimizer from `optax`.
    - [ ] Understand `optax.apply_updates` for parameter updates.
    - [ ] Implement a simple training loop for a dummy loss function to get familiar with `optax`.
- [ ] **Task 1.5: Core DQN Components (JAX-style)**
    - [ ] **Experience Replay Buffer:**
        - [ ] Implement a `ReplayBuffer` in pure JAX or `chex.ArrayTree` compatible structure.
        - [ ] Ensure efficient `add` and `sample` operations. Pay attention to how state (buffer contents, pointers) is managed in a functional paradigm. *Hint: This is often the trickiest part for pure JAX state management. Consider `chex.FIFOBuffer` or similar utility from `rlax` to simplify, or implement your own with careful indexing.*
    - [ ] **Epsilon-Greedy Policy:**
        - [ ] Implement an epsilon-greedy action selection function using `jax.random`.
        - [ ] Incorporate epsilon decay schedule.
    - [ ] **DQN Loss Function:**
        - [ ] Implement the Huber loss function (or squared error as an alternative if needed) as specified in the paper.
        - [ ] Calculate target Q-values using the target network.
- [ ] **Task 1.6: Single-Agent Training Loop**
    - [ ] Set up an OpenAI Gym Atari environment `ALE/Breakout-v5`.
    - [ ] Implement preprocessing: frame skipping, max-pooling, grayscale, resizing (as described in the paper). Use `gym.wrappers` or custom JAX-compatible functions.
    - [ ] Integrate all components: network, optimizer, replay buffer, policy.
    - [ ] Write the main training loop for a single agent, running on a single CPU or GPU.
    - [ ] Monitor training progress (e.g., loss, reward, epsilon value).

## Phase 2: JIT Compilation & Basic Parallelism (Single Machine, Multiple Devices)

This phase introduces `jax.jit` for performance, and then `jax.vmap` and `jax.pmap` for multi-device parallelization on a single machine.

- [ ] **Task 2.1: JIT Compilation for Performance**
    - [ ] Identify functions in your training loop that can be `jit`-compiled (e.g., network forward pass, loss calculation, optimizer update step).
    - [ ] Apply `@jax.jit` to these functions.
    - [ ] Understand how `jit` works (tracing, static arguments) and its limitations (side effects).
    - [ ] Debug common `jit` issues (e.g., "tracer has been lifted" errors).
- [ ] **Task 2.2: Vectorization with Vmap**
    - [ ] Understand the concept of `jax.vmap` for automatic batching.
    - [ ] Apply `vmap` to your network's forward pass to process batches of observations efficiently.
    - [ ] Potentially use `vmap` for processing a batch of transitions from the replay buffer.
- [ ] **Task 2.3: Data Parallelism with Pmap (Multi-GPU/TPU on Single Machine)**
    - [ ] Understand `jax.pmap` for replicating computations across multiple devices.
    - [ ] Modify your training loop to shard data (e.g., batches of experiences) across available devices.
    - [ ] Replicate network parameters across devices.
    - [ ] Implement `optax.DistributedUpdateState` or manually manage parameter synchronization (e.g., averaging gradients or parameters after each update) across devices using `jax.lax.pmean`.
    - [ ] Ensure proper `PRNGKey` management across parallel computations.
    - [ ] Run and verify on multiple GPUs (if available) or simulate with `JAX_PROFILE_DEVICES=2` (for CPU testing).
- [ ] **Task 2.4: Synchronous vs. Asynchronous Updates (Theoretical Understanding)**
    - [ ] Understand the implications of synchronous updates (used with `pmap`) vs. the asynchronous nature of the original DQN's distributed experience collection.
    - [ ] Note that for now, we are sticking to synchronous updates.

## Phase 3: Distributed Training (Multi-Machine) & Advanced Techniques

This phase tackles true multi-node distribution and further refinements for robust DQN training.

- [ ] **Task 3.1: JAX Distributed Setup**
    - [ ] Understand `jax.distributed.initialize()`.
    - [ ] Learn how to set up `JAX_PROCESS_INDEX`, `JAX_N_PROCESSES`, `JAX_SERVER_ADDR`, `JAX_SERVER_PORT` for multi-node communication.
    - [ ] Run a simple distributed JAX program (e.g., collective communication like `pmean` across nodes).
- [ ] **Task 3.2: Replicating Pmap Across Nodes**
    - [ ] Extend your `pmap`-based training loop to run across multiple machines.
    - [ ] Ensure parameters are synchronized correctly across *all* devices, regardless of which machine they are on. `jax.lax.pmean` handles this naturally when initialized correctly.
    - [ ] Manage global batching and data sharding across all available devices in the cluster.
- [ ] **Task 3.3: Distributed Experience Replay (Theoretical & Practical)**
    - [ ] **Theory:** Understand the original paper's asynchronous experience collection. This is a significant challenge for pure JAX.
    - [ ] **Practical (Simplified):** For exact reproduction within JAX's core functionality, the simplest approach is often to have a shared (potentially sharded) replay buffer, or to collect experiences locally on each worker and then aggregate/sample.
        - [ ] **Option A (Shared Buffer):** Implement a mechanism for workers to *send* experiences to a central (or distributed, sharded) replay buffer process, and for training processes to *request* samples. This will likely involve `jax.experimental.host_callback` or a separate process managing the buffer, which deviates from "core JAX" but is closer to the original paper's architecture.
        - [ ] **Option B (Synchronous Local Buffers + Pmean):** Each `pmap` replica (or node) maintains its own replay buffer, collects experiences, and then during the training step, gradients/parameters are averaged. This is simpler to implement in core JAX but doesn't perfectly reproduce the asynchronous aspect. *Given your "core JAX" constraint, this might be the most feasible initial approach for distributed replay.*
    - [ ] **Decision Point:** Choose which distributed replay strategy you will implement. For a learning exercise, Option B is much more straightforward with core JAX.
- [ ] **Task 3.4: Evaluation and Reproducibility**
    - [ ] Implement a separate evaluation loop where epsilon is set to a small value (or 0) and the agent's performance is measured over multiple episodes without exploration.
    - [ ] Save and load model checkpoints (parameters).
    - [ ] Ensure reproducibility by fixing random seeds at the start of the program across all processes.
    - [ ] Compare your results to the published paper's scores on selected Atari games.

## Phase 4: Refinement, Debugging & Understanding

This phase focuses on deep dives into JAX's internals and ensuring robust, reproducible results.

- [ ] **Task 4.1: JAX Debugging Techniques**
    - [ ] Learn to use `jax.debug.print` (or `jax.numpy.where` with custom values for conditional printing).
    - [ ] Understand `jax.disable_jit()` for easier debugging of raw Python code.
    - [ ] Use profiling tools (e.g., TensorBoard profiles) to identify bottlenecks.
- [ ] **Task 4.2: Understanding JAX's Trace & IR**
    - [ ] Explore how `jit` works by inspecting the JAX IR (intermediate representation).
    - [ ] Gain a deeper understanding of how `vmap` and `pmap` transform computations.
- [ ] **Task 4.3: Optimizations and Best Practices**
    - [ ] Review JAX's performance tips (e.g., avoiding Python loops inside `jit`ted functions, preferring array operations).
    - [ ] Consider using `checkify` for shape and type checking in debug mode.
    - [ ] Optimize memory usage (e.g., using `float16` if applicable and stable).
- [ ] **Task 4.4: Documentation and Code Structure**
    - [ ] Document your code thoroughly.
    - [ ] Structure your project with clear modules (e.g., `networks.py`, `replay_buffer.py`, `agent.py`, `train.py`).
    - [ ] Write unit tests for core JAX components if desired (e.g., network forward pass, loss calculation).
