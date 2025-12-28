# PyTorch Monarch: A Complete Guide

**A distributed programming framework for PyTorch based on scalable actor messaging**

---

## Table of Contents

1. [CEO Explanation](#1-ceo-explanation)
2. [Intern Onboarding Explanation](#2-intern-onboarding-explanation)
3. [Engineering Deep Dive](#3-engineering-deep-dive)
4. [Practical Playbook](#4-practical-playbook)
5. [Minimal Starter Snippets](#5-minimal-starter-snippets)
6. [What I Would Do Next Week](#6-what-i-would-do-next-week)

---

# 1) CEO Explanation

## What is PyTorch Monarch?

**In plain English:** Monarch is a framework that makes it easier to write programs that run across many computers (a cluster) for AI training. Think of it as a "traffic controller" for distributed computing.

### The "Single Controller" Model

Traditional distributed training (like `torchrun`) works like a **symphony without a conductor**—every musician (GPU process) has the same sheet music, starts at the same time, and hopes everyone stays in sync. If one musician makes a mistake, the whole performance stops.

Monarch works like a **symphony with a conductor**—there's one conductor (the controller) directing the orchestra (worker processes). The conductor:
- Knows what every musician should be doing
- Can give different instructions to different sections
- Can recover gracefully when something goes wrong
- Can add or remove musicians mid-performance

```
TRADITIONAL (SPMD/torchrun):         MONARCH (Single Controller):
┌─────────┐ ┌─────────┐              ┌─────────────────┐
│ GPU 0   │ │ GPU 1   │              │   CONTROLLER    │ ← Your Python code
│ (copy)  │ │ (copy)  │              │   (conductor)   │
└─────────┘ └─────────┘              └────────┬────────┘
     ↓           ↓                            │ directs
   Same code runs on all                      ▼
   No one is "in charge"          ┌─────────────────────────┐
                                  │    PROCESS MESH         │
                                  │ ┌─────┐ ┌─────┐ ┌─────┐ │
                                  │ │GPU 0│ │GPU 1│ │GPU 2│ │ ← Actors
                                  │ └─────┘ └─────┘ └─────┘ │
                                  └─────────────────────────┘
```

## Why is Distributed Training Hard?

| Challenge | What Happens Today | What Goes Wrong |
|-----------|-------------------|-----------------|
| **Coordination** | Every GPU runs identical code | Hard to do different things on different GPUs (like RL rollouts vs. learners) |
| **Failures** | One GPU crashes → entire job dies | 8-hour training run lost at hour 7 |
| **Iteration Speed** | Change code → restart everything → wait for cluster | 20+ minutes between experiments |
| **Cluster Friction** | Slurm scripts, environment issues, debugging mysteries | Engineers spend more time fighting infra than training models |

## The Monarch Pitch

### What Gets Simpler

| Problem | Monarch Solution |
|---------|-----------------|
| Different GPU roles | Define "Actor" classes—Trainer, Rollout Worker, Scorer. Send messages between them. |
| Handling failures | Supervision trees automatically detect and can restart failed actors. |
| Cluster abstraction | `this_host().spawn_procs(gpus=8)` works the same locally and on Slurm. |
| Dynamic workflows | Add actors, change behavior mid-run—no full restart needed. |

### What Stays Hard

- **Learning curve**: New mental model (actors, endpoints, meshes). Takes ~2 weeks for an engineer to be productive.
- **Debugging distributed bugs**: Easier than before, but still non-trivial.
- **Performance tuning**: RDMA/NCCL optimization still requires expertise.
- **Ecosystem maturity**: Early-stage framework; APIs may change.

## Where Monarch Fits in a Modern Stack

```
┌──────────────────────────────────────────────────────────────┐
│                    YOUR APPLICATION                          │
│  (Training loop, RL rollout, evaluation, data loading)       │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                      MONARCH                                 │
│  • Actors (encapsulate GPU workers)                          │
│  • Meshes (groups of actors)                                 │
│  • Endpoints (method calls across processes)                 │
│  • Channels (reliable messaging)                             │
│  • Supervision (fault tolerance)                             │
│  • RDMA (fast tensor transfer)                               │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                    INFRASTRUCTURE                            │
│  PyTorch | NCCL | CUDA | Slurm/TorchX | TCP/RDMA            │
└──────────────────────────────────────────────────────────────┘
```

**Use Cases:**
- ✅ **Large-scale training** (TorchTitan integration)
- ✅ **RLHF/RL workflows** (different actor roles: Generator, Scorer, Learner)
- ✅ **Fault-tolerant training** (with TorchFT concepts)
- ✅ **Multi-stage pipelines** (data prep → training → eval)
- ⚠️ **Inference serving** (possible but not primary focus)

## Decision Table: Should We Adopt Monarch?

| Factor | Adopt Monarch | Stick with torchrun/DDP |
|--------|--------------|-------------------------|
| **Team Size** | 3+ engineers who can learn new framework | Small team, no bandwidth |
| **Cluster Size** | 10+ nodes, frequent scaling | Single node or fixed small cluster |
| **Workload Type** | Complex (RL, multi-stage, different actor roles) | Simple DDP training |
| **Failure Rate** | High (preemption, unreliable nodes) | Stable cluster |
| **Iteration Speed** | Critical (ML research, rapid experiments) | Can tolerate slow cycles |
| **Risk Tolerance** | Can handle early-stage framework | Need battle-tested stability |

**Bottom Line:** If you're running complex multi-GPU workflows at scale and your team is frustrated with the status quo, Monarch is worth evaluating. For straightforward DDP training on stable clusters, the migration cost may not pay off yet.

---

# 2) Intern Onboarding Explanation

## The Mental Model: Four Key Concepts

Think of distributed computing like running a restaurant chain:

1. **Process Mesh** = The network of restaurant locations (one process per GPU)
2. **Actors** = The chefs at each location (your Python objects that do work)
3. **Endpoints** = The way head office communicates with chefs ("make this dish")
4. **Channels** = The reliable phone lines between locations

### Step 1: Process Mesh (One Process Per GPU)

A **Process Mesh** is a collection of OS processes, typically one per GPU. Think of it as "here are all the workers I have available."

```python
from monarch.actor import this_host

# Get the current machine
host = this_host()

# Spawn 8 processes, one for each GPU
proc_mesh = host.spawn_procs(per_host={"gpus": 8})
```

The mesh has a **shape** with named dimensions:

```
              gpus
         0  1  2  3  4  5  6  7
        ┌──┬──┬──┬──┬──┬──┬──┬──┐
hosts 0 │P0│P1│P2│P3│P4│P5│P6│P7│  ← 8 processes
        └──┴──┴──┴──┴──┴──┴──┴──┘
```

### Step 2: Actors (Python Objects That Live in Processes)

An **Actor** is a Python class that runs inside a process. Unlike a normal class:
- Each actor instance runs in its own process
- Actors communicate via **messages**, not direct method calls
- Actors have isolated state (no shared memory bugs!)

```python
from monarch.actor import Actor, endpoint

class Trainer(Actor):
    def __init__(self, lr: float):
        self.lr = lr
        self.step_count = 0
    
    @endpoint  # ← This marks a method as callable from outside
    async def train(self, batch_id: int) -> float:
        """Called remotely from the controller"""
        # Do training work on this GPU
        loss = self._do_training(batch_id)
        self.step_count += 1
        return loss
    
    def _do_training(self, batch_id):
        # Normal Python code here
        return 0.5  # Pretend loss
```

**Key insight:** The `@endpoint` decorator is what makes `train()` callable across processes.

### Step 3: Spawning Actors on the Mesh

```python
# Spawn one Trainer actor per GPU process
trainers = proc_mesh.spawn("trainers", Trainer, lr=0.001)
#                   ↑           ↑            ↑
#                   │           │            └─ Constructor args
#                   │           └─ Actor class
#                   └─ A name for this group of actors
```

Now you have 8 Trainer instances, one per GPU:

```
              gpus
         0  1  2  3  4  5  6  7
        ┌──┬──┬──┬──┬──┬──┬──┬──┐
        │T0│T1│T2│T3│T4│T5│T6│T7│  ← 8 Trainer actors
        └──┴──┴──┴──┴──┴──┴──┴──┘
```

### Step 4: Calling Endpoints

This is the magic—calling methods on actors across processes:

```python
# Call train() on ALL 8 trainers simultaneously
result = trainers.train.call(batch_id=42)

# Wait for all to complete and get results
losses = result.get()  # Returns a ValueMesh with 8 results
```

**Different calling patterns:**

```python
# Broadcast to all, collect all results
result = trainers.train.call(batch_id=42).get()

# Broadcast, wait for any ONE to respond (load balanced)
loss = trainers.train.choose(batch_id=42).get()

# Fire-and-forget (don't wait for response)
trainers.train.broadcast(batch_id=42)

# Stream results as they arrive
for future in trainers.train.stream(batch_id=42):
    loss = future.get()
    print(f"Got result: {loss}")
```

### Step 5: Channels for Custom Communication

For more complex patterns, actors can open **Channels** to send messages directly:

```python
from monarch.actor import Channel, Port, PortReceiver

class Producer(Actor):
    @endpoint
    async def produce(self, consumer_port: Port[int]) -> None:
        for i in range(10):
            consumer_port.send(i)  # Send to consumer

class Consumer(Actor):
    @endpoint
    async def consume(self) -> int:
        port, receiver = Channel[int].open()
        # Give producer our port...
        # Then receive from it:
        total = 0
        for _ in range(10):
            value = await receiver.recv()
            total += value
        return total
```

## ASCII Diagrams

### Controller + Process Mesh Layout

```
┌─────────────────────────────────────────────────────────────────┐
│                    YOUR LAPTOP / DRIVER NODE                     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                     CONTROLLER                              │ │
│  │    (Python interpreter running your training script)       │ │
│  │                                                             │ │
│  │    trainers = proc_mesh.spawn("trainers", Trainer)         │ │
│  │    trainers.train.call(step=0).get()                       │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │  TCP/Unix/MetaTLS channels
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         HOST 1                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                  PROCESS MESH (8 GPUs)                      ││
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ... ││
│  │  │Proc 0│ │Proc 1│ │Proc 2│ │Proc 3│ │Proc 4│ │Proc 5│     ││
│  │  │ GPU0 │ │ GPU1 │ │ GPU2 │ │ GPU3 │ │ GPU4 │ │ GPU5 │     ││
│  │  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘     ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### Actor Mesh on Top of Process Mesh

```
PROCESS MESH (infrastructure layer)
┌──────┬──────┬──────┬──────┐
│Proc 0│Proc 1│Proc 2│Proc 3│   ← OS processes, one per GPU
└──────┴──────┴──────┴──────┘

              ↓ spawn actors on

ACTOR MESH (application layer)
┌──────┬──────┬──────┬──────┐
│Train0│Train1│Train2│Train3│   ← Your Trainer actors
└──────┴──────┴──────┴──────┘

You can have MULTIPLE actor meshes on the same proc mesh:

┌──────┬──────┬──────┬──────┐
│Train0│Train1│Train2│Train3│   ← Trainers
├──────┼──────┼──────┼──────┤
│Eval 0│Eval 1│Eval 2│Eval 3│   ← Evaluators (different actor class)
└──────┴──────┴──────┴──────┘
```

### Endpoint Call Flow

```
CONTROLLER                           REMOTE ACTOR
    │                                      │
    │  trainers.train.call(step=0)         │
    │                                      │
    │  1. Serialize args (pickle)          │
    │  2. Create response port             │
    │  3. Send via channel                 │
    │─────────────────────────────────────▶│
    │                                      │ 4. Deserialize args
    │                                      │ 5. Execute train(step=0)
    │                                      │ 6. Serialize result
    │◀─────────────────────────────────────│ 7. Send response
    │                                      │
    │  .get() → ValueMesh[float]           │
    │     with results from all actors     │
    ▼                                      ▼
```

### Supervision Tree / Restart Behavior

```
                    ┌─────────────────┐
                    │ ROOT CLIENT     │ ← Your driver script
                    │ (controller)    │
                    └────────┬────────┘
                             │ owns
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
   ┌────────────┐     ┌────────────┐     ┌────────────┐
   │ HOST MESH  │     │ PROC MESH  │     │ ACTOR MESH │
   │  Agent     │     │  (8 procs) │     │ (Trainers) │
   └────────────┘     └─────┬──────┘     └────────────┘
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
     ┌─────────┐       ┌─────────┐       ┌─────────┐
     │ Proc 0  │       │ Proc 1  │       │ Proc 2  │
     │ Agent   │       │ Agent   │       │ Agent   │  ...
     └────┬────┘       └────┬────┘       └────┬────┘
          ▼                 ▼                 ▼
     ┌─────────┐       ┌─────────┐       ┌─────────┐
     │Trainer 0│       │Trainer 1│       │Trainer 2│
     └─────────┘       └─────────┘       └─────────┘

FAILURE PROPAGATION:
    • Trainer 1 crashes
    • Proc 1 Agent detects failure
    • Reports supervision event UP the tree
    • Controller receives MeshFailure
    • Can choose: restart, ignore, or fail

DEFAULT BEHAVIOR:
    • Unhandled failure → print error → exit program
    • Custom __supervise__() method can override
```

## What's an "Actor" vs. a Normal Python Class?

| Normal Python Class | Monarch Actor |
|---------------------|---------------|
| Lives in one process | One instance per process in the mesh |
| Call methods directly | Call endpoints via message passing |
| Shared state possible | Isolated state (no data races!) |
| Exceptions propagate locally | Exceptions become supervision events |
| No special lifecycle | Has spawn, init, cleanup lifecycle |

## What's an "Endpoint Function"?

An **endpoint** is a method decorated with `@endpoint` that can be called remotely:

```python
class MyActor(Actor):
    @endpoint
    async def my_endpoint(self, x: int) -> str:
        return f"Got {x}"
```

**Key differences from regular methods:**
1. Can be `async` or sync (but don't mix in same actor)
2. Arguments and return values are **serialized** (pickled)
3. Called via `.call()`, `.broadcast()`, `.choose()`, etc.
4. Exceptions become `ActorError` on the caller side

## What's a "Mesh" and Why Does it Matter?

A **Mesh** is a multi-dimensional array of things (processes or actors) with named dimensions:

```python
# 2 hosts × 8 GPUs = 16 processes
proc_mesh = host.spawn_procs(per_host={"hosts": 2, "gpus": 8})

# Shape: (hosts=2, gpus=8)
# You can slice it:
first_host = proc_mesh.slice(hosts=0)        # 8 GPUs on host 0
first_gpu_each_host = proc_mesh.slice(gpus=0) # GPU 0 on both hosts
```

**Why named dimensions matter:**
- Makes code readable: `mesh.slice(gpus=0)` vs `mesh[0, :]`
- Maps to physical topology: hosts, racks, GPUs
- Enables smart placement and collective operations

---

# 3) Engineering Deep Dive

## Architecture Overview

### Controller Responsibilities

The controller (your Python script running on the driver node):

1. **Orchestration**: Spawns hosts, processes, and actors
2. **Scheduling**: Decides what work each actor should do
3. **Coordination**: Synchronizes across actors when needed
4. **Monitoring**: Receives supervision events from workers
5. **Recovery**: Decides how to handle failures

**Key insight:** The controller is just another actor! It's the "root client actor" that owns the supervision tree.

### Worker Processes / Process Mesh

Each process in the mesh:
- Runs one `ProcMeshAgent` (internal actor managing that proc)
- Hosts zero or more user actors
- Has its own Python interpreter and memory space
- Communicates via channels (TCP/Unix/MetaTLS)

**Bootstrap sequence:**
```
1. Controller allocates hosts (Slurm, local process, etc.)
2. Each host starts a HostMeshAgent
3. HostMeshAgent spawns ProcMeshAgents (one per GPU)
4. ProcMeshAgents wait for actor spawn requests
5. Controller sends spawn messages → actors start
```

### Actor Lifecycle

```
                     CREATED
                        │
                        ▼
                   INITIALIZING ──┐
                        │         │ init fails
                        ▼         ▼
     ┌───────────── IDLE ◀────── FAILED
     │                  │
     │ recv message     │
     ▼                  │
  PROCESSING ───────────┘
     │                  │
     │ save/load        │
     ▼                  │
  SAVING/LOADING ───────┘
     │
     │ stop signal
     ▼
   STOPPING
     │
     ▼
   STOPPED
```

**Actor construction:**
```python
# In _Actor.handle() for Init message:
1. Receive ActorInitArgs (class, proc_mesh, name, creator, args)
2. Instantiate Class(*args, **kwargs)
3. Set up context (rank, proc_mesh, etc.)
4. Send response to caller
```

### Endpoint Invocation Semantics

**Sync vs Async:**
```python
# ASYNC endpoint (recommended for I/O-bound work)
@endpoint
async def process_data(self, data):
    result = await some_async_operation(data)
    return result

# SYNC endpoint (for CPU-bound work)
@endpoint
def compute(self, x):
    return expensive_computation(x)

# IMPORTANT: Don't mix sync and async endpoints in same actor!
# Sync endpoints block the event loop
```

**Calling patterns:**

| Pattern | Description | Use When |
|---------|-------------|----------|
| `.call()` | Broadcast to all, collect all results | Need results from every actor |
| `.call_one()` | Send to single actor (must be 1 actor in mesh) | Single-actor mesh |
| `.choose()` | Load-balanced to one actor | Request/response pattern |
| `.broadcast()` | Fire-and-forget to all | Don't need confirmation |
| `.stream()` | Broadcast, yield results as they arrive | Progressive processing |

**Backpressure:** Channels have bounded buffers (default 1024 messages). If a receiver is slow, senders may block when the buffer fills.

## Communication

### Channels / Message Transport

Channels provide one-way, typed message passing:

```
Tx<M> ──────────────────────▶ Rx<M>
(transmit end)               (receive end)
```

**Guarantees:**
- Per-sender FIFO ordering
- At-most-once delivery (no duplicates)
- Reconnect with retry on network failure
- Cancellation-safe

**Transport implementations:**

| Transport | Address | Use Case |
|-----------|---------|----------|
| Local | `local:uuid` | Same process, testing |
| TCP | `tcp:host:port` | Cross-machine |
| Unix | `unix:/path` | Same machine, fast |
| MetaTLS | `metatls:host:port` | Production with TLS |
| Sim | `sim:inner-addr` | Simulation/testing |

### Performance Implications

**Latency factors:**
1. Serialization (pickle) time
2. Network round-trip
3. Deserialization time
4. Queue wait time (backpressure)

**Throughput tips:**
- Batch messages when possible (one call with list vs. many calls)
- Use RDMA for large tensor transfers
- Avoid frequent small synchronizations

### RDMA / Tensor Engine

The tensor engine enables:
- **Point-to-point GPU memory transfers** without CPU involvement
- **Zero-copy** when possible
- **libibverbs-based** RDMA for low latency

```python
from monarch.rdma import RDMABuffer

# Register a tensor for RDMA
buffer = RDMABuffer(my_tensor)

# On receiver side, read directly into GPU memory
await buffer.read_into(target_tensor)
```

**When RDMA matters:**
- Large model weight synchronization (RLHF learner → generators)
- Gradient all-reduce across nodes
- Pipeline parallel activations transfer

**Requirements:**
- RDMA NICs (Mellanox, etc.)
- Build with `USE_TENSOR_ENGINE=1` (default)
- Proper driver/firmware setup

## Fault Tolerance

### Supervision Tree Concepts

Every actor has a parent (the actor that spawned it):

```
RootClientActor (your driver)
    └── ActorMesh "trainers"
            ├── Trainer[0]
            ├── Trainer[1]
            └── Trainer[2]
```

When an actor fails, the failure **propagates up** the tree until handled:

```python
class MyActor(Actor):
    def __supervise__(self, failure: MeshFailure) -> object:
        """Called when a child actor fails."""
        if should_restart(failure):
            self.restart_child(failure.actor_id)
            return True  # Handled, don't propagate
        return None  # Not handled, propagate up
```

### What Happens on Process/Host Failure

1. **Connection drops** → channels detect via TCP keepalive or explicit ping
2. **TxStatus flips to Closed** → pending sends fail
3. **Supervision event generated** → `MeshFailure` with error info
4. **Propagation** → event travels up supervision tree
5. **Handler decision** → restart, ignore, or crash

**Default behavior:** If no `__supervise__` handles it, `unhandled_fault_hook` is called, which logs and exits.

### Restart Semantics and State Recovery

**What Monarch provides:**
- Detection of failures
- Propagation to supervisor
- Infrastructure to restart actors

**What YOU must provide:**
- State checkpointing logic
- Recovery/reload logic
- Decision logic (when to restart vs. fail)

```python
class CheckpointedTrainer(Actor):
    def __init__(self):
        self.state = self.load_checkpoint_if_exists()
    
    @endpoint
    async def train_step(self):
        # ... training ...
        if step % 100 == 0:
            self.save_checkpoint()
    
    def __supervise__(self, failure):
        # On failure, we could restart from last checkpoint
        # But the framework doesn't auto-restore state
        pass
```

## Strengths & Weaknesses

### Strengths

| Strength | Details |
|----------|---------|
| **Cleaner abstraction** | Actor model fits distributed computing well |
| **Flexible topologies** | Different actors for different roles (RL!) |
| **Built-in fault detection** | Supervision events are automatic |
| **Python-native** | No separate config files, all in Python |
| **RDMA support** | High-performance tensor transfer |

### Weaknesses

| Weakness | Details |
|----------|---------|
| **Learning curve** | New paradigm for most PyTorch users |
| **Early stage** | APIs may change, docs are incomplete |
| **Debugging complexity** | Distributed tracing still evolving |
| **Overhead for simple cases** | DDP is simpler if that's all you need |
| **Serialization costs** | Everything crosses pickle boundary |

### Scaling Limits / Bottlenecks

1. **Controller bottleneck**: Single controller handles all coordination
   - Mitigate: Hierarchical controllers, reduce sync points
   
2. **Channel throughput**: TCP/Unix sockets have limits
   - Mitigate: Batch messages, use RDMA for tensors
   
3. **Serialization**: Pickle can be slow for complex objects
   - Mitigate: Keep endpoint args simple, pass references

### Operational Overhead

- **Dependencies**: Rust nightly, CUDA, NCCL, optionally RDMA libs
- **Build time**: First build can take 10+ minutes
- **Cluster setup**: Need proper network config for TCP/RDMA

## Common Failure Modes

### 1. Connection Errors / Flakiness

**Symptoms:**
```
ChannelError::Closed
Connection reset by peer
Timeout waiting for ack
```

**Causes:**
- Network partition
- Firewall blocking ports
- Process crashed

**Fixes:**
- Check network connectivity
- Ensure ports are open (default varies by transport)
- Check process is actually running (`ps aux | grep python`)

### 2. Placement / Scheduling Surprises

**Symptoms:**
- Actor ends up on wrong GPU
- `CUDA_VISIBLE_DEVICES` mismatch
- Process count doesn't match GPU count

**Causes:**
- Incorrect `per_host` specification
- Environment variable conflicts
- Slurm vs. local allocation mismatch

**Fixes:**
```python
# Explicitly check where you are
@endpoint
async def debug_placement(self):
    import os
    return {
        "rank": current_rank(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hostname": socket.gethostname()
    }
```

### 3. Deadlocks / Backpressure

**Symptoms:**
- Program hangs
- No progress, no errors
- Memory usage grows

**Causes:**
- Circular waiting (A waits for B, B waits for A)
- Sender outpaces receiver
- Forgot to await a Future

**Fixes:**
- Use `.broadcast()` for fire-and-forget
- Add timeouts to all waits
- Check for circular dependencies in message flow

### 4. GPU Resource Mismatch

**Symptoms:**
```
CUDA out of memory
RuntimeError: CUDA error: device-side assert triggered
```

**Causes:**
- Model too large for GPU
- Multiple actors on same GPU
- Memory leak across steps

**Fixes:**
- Verify GPU count: `proc_mesh.spawn_procs(per_host={"gpus": N})`
- Check `torch.cuda.memory_summary()`
- Ensure proper cleanup between runs

### 5. Environment / Version Skew

**Symptoms:**
- Import errors on workers
- Pickle errors (`Can't find class...`)
- Different behavior across nodes

**Causes:**
- Different Python/PyTorch versions
- Missing packages on workers
- Code not synced to workers

**Fixes:**
- Use consistent Docker images
- Use `sync_workspace()` for code sync
- Pin all dependency versions

---

# 4) Practical Playbook

## Adoption Path

```
PHASE 1: Local Development (Week 1)
├── Install Monarch locally
├── Run examples from repo
├── Write first Actor + endpoint
└── Test with fake_in_process_host()

PHASE 2: Multi-Process (Week 2)
├── Use this_host().spawn_procs()
├── Run with multiple local processes
├── Debug with logging
└── Handle first supervision events

PHASE 3: Multi-Host (Week 3-4)
├── Set up cluster access
├── Use Slurm or TorchX launch
├── Add proper error handling
└── Implement checkpointing

PHASE 4: Production (Week 5+)
├── Performance optimization
├── Comprehensive monitoring
├── CI/CD integration
└── On-call playbooks
```

## Engineering Checklist

### Repo Setup

```bash
# 1. Install Rust nightly
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup toolchain install nightly
rustup default nightly

# 2. Install system deps (Ubuntu)
sudo apt install -y ninja-build libunwind-dev clang

# 3. Install CUDA (if not present)
sudo apt install -y cuda-toolkit-12-8

# 4. Clone and install Monarch
git clone https://github.com/meta-pytorch/monarch.git
cd monarch
pip install uv
uv sync  # or USE_TENSOR_ENGINE=0 uv sync for CPU-only

# 5. Verify
uv run python -c "from monarch import actor; print('OK')"
```

### Recommended Project Layout

```
my_project/
├── pyproject.toml
├── README.md
├── src/
│   └── my_project/
│       ├── __init__.py
│       ├── actors/           # Actor definitions
│       │   ├── __init__.py
│       │   ├── trainer.py    # class Trainer(Actor)
│       │   ├── evaluator.py  # class Evaluator(Actor)
│       │   └── data_loader.py
│       ├── endpoints/        # Complex endpoint logic
│       │   └── training_step.py
│       ├── config/           # Configuration
│       │   └── defaults.py
│       └── main.py           # Entrypoint / controller logic
├── scripts/
│   ├── launch_local.sh
│   ├── launch_slurm.sh
│   └── launch_torchx.py
├── tests/
│   ├── test_actors.py        # Unit tests with mock
│   └── test_integration.py   # Multi-process tests
└── configs/
    ├── dev.yaml
    └── prod.yaml
```

### Logging / Metrics Strategy

**What to observe:**

| Metric | Where | Why |
|--------|-------|-----|
| Endpoint latency | Controller | Spot slow actors |
| Message queue depth | Channels | Detect backpressure |
| Actor spawn time | Controller | Bootstrap health |
| Supervision events | Controller | Failure detection |
| GPU memory | Each actor | OOM prevention |

**Built-in observability:**
```python
# Enable endpoint instrumentation
@endpoint(instrument=True)
async def train(self):
    ...

# Access metrics via OpenTelemetry
from monarch._src.actor.telemetry import METER
```

## Benchmark Plan

### Microbenchmark 1: Endpoint Ping-Pong

```python
"""Measure round-trip latency for endpoint calls."""
import time
from monarch.actor import Actor, endpoint, this_host

class PingActor(Actor):
    @endpoint
    async def ping(self) -> str:
        return "pong"

async def benchmark_ping():
    procs = this_host().spawn_procs(per_host={"gpus": 1})
    actor = procs.spawn("ping", PingActor)
    
    # Warmup
    for _ in range(100):
        await actor.ping.call_one()
    
    # Measure
    N = 1000
    start = time.perf_counter()
    for _ in range(N):
        await actor.ping.call_one()
    elapsed = time.perf_counter() - start
    
    print(f"Avg latency: {elapsed/N*1000:.2f} ms")
```

**Expected:** ~0.1-1 ms local, ~1-10 ms cross-node

### Microbenchmark 2: Tensor Transfer

```python
"""Measure RDMA tensor transfer throughput."""
import torch
from monarch.rdma import RDMABuffer

async def benchmark_tensor_transfer():
    # Create source and destination tensors
    src = torch.randn(1000, 1000, device="cuda")
    dst = torch.empty_like(src)
    
    buffer = RDMABuffer(src)
    
    # Warmup
    for _ in range(10):
        await buffer.read_into(dst)
    
    # Measure
    N = 100
    start = time.perf_counter()
    for _ in range(N):
        await buffer.read_into(dst)
    elapsed = time.perf_counter() - start
    
    bytes_transferred = src.numel() * src.element_size() * N
    throughput_gbps = bytes_transferred / elapsed / 1e9 * 8
    print(f"Throughput: {throughput_gbps:.1f} Gbps")
```

### Workload Benchmark 1: DDP-Style Training

```python
"""Benchmark distributed data parallel training pattern."""
class DDPBenchActor(Actor):
    def __init__(self):
        self.model = create_model().cuda()
        self.optimizer = optim.Adam(self.model.parameters())
    
    @endpoint
    async def train_step(self, batch_id: int) -> float:
        # Simulate training step
        data = torch.randn(32, 1024).cuda()
        output = self.model(data)
        loss = output.mean()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

async def benchmark_ddp():
    procs = this_host().spawn_procs(per_host={"gpus": 8})
    trainers = procs.spawn("ddp", DDPBenchActor)
    
    N = 100
    start = time.perf_counter()
    for i in range(N):
        await trainers.train_step.call(batch_id=i)
    elapsed = time.perf_counter() - start
    
    print(f"Throughput: {N/elapsed:.1f} steps/sec")
```

### Workload Benchmark 2: RL Rollout/Learner

```python
"""Benchmark async rollout generation pattern."""
class RolloutActor(Actor):
    @endpoint
    async def generate(self, prompt: str) -> List[float]:
        # Simulate generation
        await asyncio.sleep(0.1)  # Model inference time
        return [random.random() for _ in range(16)]

class LearnerActor(Actor):
    @endpoint
    async def train(self, trajectories: List[List[float]]) -> float:
        # Simulate training
        return sum(sum(t) for t in trajectories)

async def benchmark_rl():
    procs = this_host().spawn_procs(per_host={"gpus": 8})
    rollouts = procs.slice(gpus=slice(0, 7)).spawn("rollout", RolloutActor)
    learner = procs.slice(gpus=7).spawn("learner", LearnerActor)
    
    # Parallel rollout generation
    start = time.perf_counter()
    trajectories = await rollouts.generate.call(prompt="test")
    await learner.train.call_one(trajectories=[t for t in trajectories.values()])
    elapsed = time.perf_counter() - start
    
    print(f"Rollout->Train cycle: {elapsed*1000:.1f} ms")
```

## Operational Playbook

### Rolling Out Changes Safely

```bash
# 1. Test locally first
USE_TENSOR_ENGINE=0 pytest tests/ -v

# 2. Test on single node
python scripts/launch_local.sh

# 3. Test on small slice of cluster
python main.py --hosts=2 --gpus-per-host=2

# 4. Gradual rollout to full cluster
python main.py --hosts=8 --gpus-per-host=8
python main.py --hosts=64 --gpus-per-host=8
```

### Handling Failures Without Losing Long Runs

```python
class ResilientTrainer(Actor):
    def __init__(self):
        self.step = 0
        self.checkpoint_freq = 100
        self._load_latest_checkpoint()
    
    @endpoint
    async def train(self):
        while self.step < 10000:
            self._do_step()
            self.step += 1
            if self.step % self.checkpoint_freq == 0:
                self._save_checkpoint()
    
    def _save_checkpoint(self):
        path = f"/checkpoints/step_{self.step}.pt"
        torch.save({
            "step": self.step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }, path)
    
    def _load_latest_checkpoint(self):
        checkpoints = sorted(glob("/checkpoints/*.pt"))
        if checkpoints:
            ckpt = torch.load(checkpoints[-1])
            self.step = ckpt["step"]
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
```

### Troubleshooting Flowchart

```
ISSUE: Job not starting
│
├─ Check: Is Slurm allocation valid?
│   └─ squeue -u $USER
│
├─ Check: Are workers reachable?
│   └─ ssh each host
│
├─ Check: Are ports open?
│   └─ nc -zv host port
│
└─ Check: Python environment correct?
    └─ which python; python --version

ISSUE: Actors not responding
│
├─ Check: Are processes running?
│   └─ ps aux | grep python on each host
│
├─ Check: Logs for errors?
│   └─ Look for ActorError, SupervisionError
│
├─ Check: Network connectivity?
│   └─ ping between hosts
│
└─ Check: GPU health?
    └─ nvidia-smi on each host

ISSUE: Performance worse than expected
│
├─ Check: Endpoint latency
│   └─ Add timing around .call()
│
├─ Check: Serialization overhead
│   └─ Profile pickle.dumps size
│
├─ Check: GPU utilization
│   └─ nvidia-smi dmon
│
└─ Check: Network saturation
    └─ iftop / nload
```

---

# 5) Minimal Starter Snippets

## Hello Cluster Example

```python
#!/usr/bin/env python3
"""
Minimal Monarch example: Actor + Endpoint + Mesh spawn.

Run with: python hello_cluster.py
"""
from monarch.actor import Actor, endpoint, this_host, current_rank


class HelloActor(Actor):
    """A simple actor that says hello."""
    
    def __init__(self, greeting: str):
        self.greeting = greeting
        self.rank = current_rank()
        print(f"[Rank {self.rank}] Initialized with greeting: {greeting}")
    
    @endpoint
    async def say_hello(self, name: str) -> str:
        """Return a greeting. Called remotely from controller."""
        message = f"[Rank {self.rank}] {self.greeting}, {name}!"
        print(message)
        return message


async def main():
    # 1. Get the current host
    host = this_host()
    
    # 2. Spawn processes (one per GPU, or just 2 for testing)
    proc_mesh = host.spawn_procs(per_host={"gpus": 2})
    
    # 3. Spawn actors on the process mesh
    greeters = proc_mesh.spawn("greeters", HelloActor, greeting="Hello")
    
    # 4. Call the endpoint on all actors
    results = await greeters.say_hello.call(name="World")
    
    # 5. Print results from all actors
    print("\n--- Results from all actors ---")
    for point, message in results.items():
        print(f"  Actor at {point}: {message}")
    
    # 6. Clean shutdown
    from monarch.actor import shutdown_context
    await shutdown_context()
    print("\nDone!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

**Output:**
```
[Rank {'gpus': 0}] Initialized with greeting: Hello
[Rank {'gpus': 1}] Initialized with greeting: Hello
[Rank {'gpus': 0}] Hello, World!
[Rank {'gpus': 1}] Hello, World!

--- Results from all actors ---
  Actor at Point(rank=0, gpus=0): [Rank {'gpus': 0}] Hello, World!
  Actor at Point(rank=1, gpus=1): [Rank {'gpus': 1}] Hello, World!

Done!
```

## Multi-Host Launch Pattern

### Using Slurm

```python
#!/usr/bin/env python3
"""
Multi-host Monarch example using Slurm.
"""
from monarch.actor import Actor, endpoint, current_rank
from monarch.job import SlurmJob


class Worker(Actor):
    @endpoint
    async def get_info(self) -> dict:
        import socket
        import os
        return {
            "rank": current_rank(),
            "hostname": socket.gethostname(),
            "cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "N/A")
        }


async def main():
    NUM_NODES = 2
    GPUS_PER_NODE = 8
    
    # Create Slurm job
    job = SlurmJob(
        meshes={"workers": NUM_NODES},
        job_name="monarch_multihost",
        gpus_per_node=GPUS_PER_NODE,
        time_limit="01:00:00",
    )
    
    try:
        # Get allocated hosts
        state = job.state()
        
        # Create process mesh
        proc_mesh = state.workers.spawn_procs({"gpus": GPUS_PER_NODE})
        
        # Spawn actors
        workers = proc_mesh.spawn("workers", Worker)
        
        # Query all workers
        infos = await workers.get_info.call()
        
        print(f"Got info from {len(list(infos.values()))} workers:")
        for point, info in infos.items():
            print(f"  {point}: {info}")
    
    finally:
        job.kill()  # Always release resources


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### Using Local Processes (for testing)

```python
#!/usr/bin/env python3
"""
Multi-process local testing (simulates multi-host).
"""
from monarch.actor import Actor, endpoint, this_host


class Worker(Actor):
    @endpoint
    async def work(self, x: int) -> int:
        return x * x


async def main():
    # Spawn 4 local processes (simulates 4 GPUs)
    host = this_host()
    procs = host.spawn_procs(per_host={"gpus": 4})
    
    workers = procs.spawn("workers", Worker)
    
    # Test broadcast call
    results = await workers.work.call(x=5)
    print(f"Results: {list(results.values())}")  # [25, 25, 25, 25]
    
    from monarch.actor import shutdown_context
    await shutdown_context()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## Supervision / Restart Example

```python
#!/usr/bin/env python3
"""
Supervision example showing failure handling.
"""
from monarch.actor import Actor, endpoint, this_host, MeshFailure
import random


class UnreliableWorker(Actor):
    """A worker that sometimes fails."""
    
    def __init__(self):
        self.call_count = 0
    
    @endpoint
    async def do_work(self) -> str:
        self.call_count += 1
        # Simulate random failure
        if random.random() < 0.3:
            raise RuntimeError(f"Random failure on call {self.call_count}!")
        return f"Success on call {self.call_count}"


class Supervisor(Actor):
    """Demonstrates custom supervision handling."""
    
    def __init__(self):
        self.failures_handled = 0
    
    def __supervise__(self, failure: MeshFailure) -> object:
        """
        Called when a child actor fails.
        
        Returns:
            True: Failure was handled, don't propagate
            None/False: Not handled, propagate up
        """
        self.failures_handled += 1
        print(f"[Supervisor] Handling failure #{self.failures_handled}")
        print(f"[Supervisor] Failure report: {failure.report()}")
        
        # For this example, we just log and continue
        # In production, you might restart the actor
        return True  # Mark as handled


async def main():
    host = this_host()
    procs = host.spawn_procs(per_host={"gpus": 2})
    
    # Spawn workers
    workers = procs.spawn("workers", UnreliableWorker)
    
    # Try to call workers multiple times
    for i in range(5):
        print(f"\n--- Attempt {i+1} ---")
        try:
            results = await workers.do_work.call()
            for point, result in results.items():
                print(f"  {point}: {result}")
        except Exception as e:
            print(f"  Error: {e}")
    
    from monarch.actor import shutdown_context
    await shutdown_context()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

---

# 6) What I Would Do Next Week

## Onboarding Plan for a New Engineer

### Day 1: Environment Setup + First Example

**Morning (2 hours):**
1. Clone the Monarch repo
2. Install dependencies (follow README)
3. Run `uv run python -c "from monarch import actor; print('OK')"`
4. If issues, troubleshoot with team

**Afternoon (3 hours):**
1. Run the `examples/` from the repo
2. Read `README.md` thoroughly
3. Modify `hello_cluster.py` to add your own endpoint
4. Experiment with different `per_host` configurations

**Evening (1 hour):**
1. Read this guide's "CEO Explanation" section
2. Write down 3 questions you have

### Day 2: Core Concepts Deep Dive

**Morning (3 hours):**
1. Read the Hyperactor book (docs/source/books/hyperactor-book/)
   - Focus on: Actors, Handlers, Channels
2. Implement a simple Producer/Consumer with Channels

**Afternoon (3 hours):**
1. Read the Process Mesh docs
2. Implement an actor that uses `current_rank()` to do different work
3. Try slicing a mesh and spawning on a subset

### Day 3: Fault Tolerance

**Morning (2 hours):**
1. Read supervision documentation
2. Implement `__supervise__` in your actor
3. Intentionally cause failures and observe behavior

**Afternoon (3 hours):**
1. Implement a checkpointing pattern
2. Test recovery from simulated failure
3. Understand supervision event flow

### Day 4: Real Workload Pattern

**Morning (3 hours):**
1. Choose a pattern: DDP training OR RL rollout/learner
2. Implement a skeleton version
3. Get it running locally

**Afternoon (3 hours):**
1. Add proper error handling
2. Add logging and observability
3. Profile and identify bottlenecks

### Day 5: Multi-Host + Production Prep

**Morning (2 hours):**
1. Set up Slurm/cluster access
2. Run your workload on 2+ nodes
3. Debug any multi-host issues

**Afternoon (3 hours):**
1. Document what you learned
2. Create a PR with your example
3. Present findings to team

## Quick Reference Card

```
# Basic pattern
from monarch.actor import Actor, endpoint, this_host

class MyActor(Actor):
    @endpoint
    async def my_method(self, x: int) -> int:
        return x * 2

host = this_host()
procs = host.spawn_procs(per_host={"gpus": 8})
actors = procs.spawn("my_actors", MyActor)
result = await actors.my_method.call(x=5)

# Calling patterns
.call()      → all, wait for all
.call_one()  → one (must be singleton mesh)
.choose()    → load balanced to one
.broadcast() → all, fire-and-forget
.stream()    → all, yield as ready

# Common imports
from monarch.actor import (
    Actor,
    endpoint,
    this_host,
    this_proc,
    current_rank,
    current_size,
    context,
    Channel,
    Port,
    PortReceiver,
    MeshFailure,
    shutdown_context
)
```

---

## References

- **Monarch GitHub**: https://github.com/meta-pytorch/monarch
- **Monarch Docs**: https://meta-pytorch.org/monarch/
- **PyTorch Blog**: https://pytorch.org/blog/introducing-pytorch-monarch/
- **Hyperactor Book**: `docs/source/books/hyperactor-book/`
- **Examples**: `examples/` directory in repo

---

*Document generated from Monarch source code analysis. Version: As of source code in repository.*
