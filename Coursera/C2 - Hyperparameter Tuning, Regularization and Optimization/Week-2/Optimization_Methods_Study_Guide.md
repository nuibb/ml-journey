# Optimization Methods for Neural Networks - Complete Study Guide

> Based on Coursera Deep Learning Specialization - Course 2, Week 2

---

## Table of Contents

1. [Gradient Descent (Batch)](#1-gradient-descent-batch)
2. [Stochastic Gradient Descent (SGD)](#2-stochastic-gradient-descent-sgd)
3. [Mini-Batch Gradient Descent](#3-mini-batch-gradient-descent)
4. [Momentum](#4-momentum)
5. [RMSProp](#5-rmsprop)
6. [Adam Optimizer](#6-adam-optimizer)
7. [Learning Rate Decay and Scheduling](#7-learning-rate-decay-and-scheduling)
8. [Comparison of All Methods](#8-comparison-of-all-methods)
9. [Quick Reference Cheat Sheet](#9-quick-reference-cheat-sheet)

---

## 1. Gradient Descent (Batch)

### What Is It?

Batch Gradient Descent computes the gradient of the cost function with respect to parameters using the **entire training set** in each iteration. It then takes one step in the direction that reduces the cost.

### Intuition

Imagine standing on a hilly landscape and wanting to reach the lowest valley. At each step, you look at the **entire terrain** (all training examples), compute the steepest downhill direction, and take one step. This gives you a very accurate direction but it's slow because you must survey the entire landscape before each step.

### Update Rule

For each layer `l = 1, ..., L`:

```
W[l] = W[l] - alpha * dW[l]
b[l] = b[l] - alpha * db[l]
```

Where:
- `W[l]`, `b[l]` are the weight matrix and bias vector for layer `l`
- `dW[l]`, `db[l]` are the gradients (partial derivatives of cost J with respect to W and b)
- `alpha` is the learning rate (a scalar that controls step size)

### Python Implementation

```python
def update_parameters_with_gd(parameters, grads, learning_rate):
    L = len(parameters) // 2  # number of layers

    for l in range(1, L + 1):
        parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]

    return parameters
```

### Full Training Loop

```python
X = data_input
Y = labels
m = X.shape[1]  # Number of training examples
parameters = initialize_parameters(layers_dims)

for i in range(num_iterations):
    # Forward propagation on ALL examples
    a, caches = forward_propagation(X, parameters)
    # Compute cost over ALL examples
    cost_total = compute_cost(a, Y)
    # Backward propagation
    grads = backward_propagation(a, caches, parameters)
    # Update parameters (one step per full pass)
    parameters = update_parameters(parameters, grads)
    cost_avg = cost_total / m
```

### Characteristics

| Property | Value |
|----------|-------|
| Convergence path | Smooth, directly toward minimum |
| Speed per iteration | Slow (processes all m examples) |
| Memory | High (must hold all data in memory) |
| Best for | Small datasets (< ~2000 examples) |

### When to Use

- Small datasets where the entire dataset fits in memory easily
- When you need very smooth convergence curves
- When computational cost per iteration is not a concern

---

## 2. Stochastic Gradient Descent (SGD)

### What Is It?

SGD is the opposite extreme of Batch GD: it updates parameters after computing the gradient on **just 1 training example** at a time.

### Intuition

Instead of surveying the entire terrain before each step, you look at just one random point and take a step. This is very fast per step but the direction is noisy -- you'll zigzag a lot on the way to the minimum.

### Training Loop

```python
X = data_input
Y = labels
m = X.shape[1]
parameters = initialize_parameters(layers_dims)

for i in range(num_iterations):
    cost_total = 0
    for j in range(m):
        # Forward propagation on ONE example
        a, caches = forward_propagation(X[:, j], parameters)
        cost_total += compute_cost(a, Y[:, j])
        # Backward propagation on ONE example
        grads = backward_propagation(a, caches, parameters)
        # Update parameters after EACH example
        parameters = update_parameters(parameters, grads)
    cost_avg = cost_total / m
```

### Characteristics

| Property | Value |
|----------|-------|
| Convergence path | Very noisy, oscillates heavily |
| Speed per iteration | Very fast (just 1 example) |
| Memory | Very low |
| Best for | Very large datasets, online learning |

### Key Difference from Batch GD

SGD requires **3 nested loops**:
1. Over the number of iterations (epochs)
2. Over the `m` training examples
3. Over the layers (to update W and b)

Batch GD only has loops 1 and 3.

### Downsides

- Loses the speedup from vectorization (processing 1 example at a time instead of matrix operations)
- High variance in parameter updates leads to oscillation
- May never truly "converge" -- keeps bouncing around the minimum

---

## 3. Mini-Batch Gradient Descent

### What Is It?

Mini-batch GD is the sweet spot between Batch GD and SGD. Instead of using all `m` examples or just 1, you split the dataset into **mini-batches** of size `B` (commonly 32, 64, 128, or 256) and update parameters after each mini-batch.

### Intuition

Instead of looking at the whole terrain or just one point, you look at a **small group** of points. This gives you a reasonably accurate direction while being fast enough to make many updates per epoch.

### How Mini-Batches Are Created

**Step 1: Shuffle** the training set randomly (both X and Y synchronously).

```python
permutation = list(np.random.permutation(m))
shuffled_X = X[:, permutation]
shuffled_Y = Y[:, permutation].reshape((1, m))
```

**Step 2: Partition** into mini-batches of the chosen size.

```python
# If m = 148 and mini_batch_size = 64:
# Mini-batch 1: examples 0-63   (64 examples)
# Mini-batch 2: examples 64-127 (64 examples)
# Mini-batch 3: examples 128-147 (20 examples - the remainder)
```

### Python Implementation

```python
import math
import numpy as np

def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []

    # Step 1: Shuffle
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))

    # Step 2: Partition
    num_complete_minibatches = math.floor(m / mini_batch_size)

    for k in range(num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : (k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : (k + 1) * mini_batch_size]
        mini_batches.append((mini_batch_X, mini_batch_Y))

    # Handle the last (possibly smaller) mini-batch
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size:]
        mini_batches.append((mini_batch_X, mini_batch_Y))

    return mini_batches
```

### Choosing Mini-Batch Size

| Size | Name | Behavior |
|------|------|----------|
| `B = m` | Batch Gradient Descent | Smooth but slow. No mini-batch processing. |
| `B = 1` | Stochastic Gradient Descent | Fast but noisy. Loses vectorization advantage. |
| `1 < B < m` | Mini-Batch GD | Fast AND takes advantage of vectorization. |

**Typical values**: 64, 128, 256, 512 (powers of 2 are preferred because they align well with memory/CPU architectures).

### Key Takeaways

- Shuffling before each epoch ensures different mini-batch compositions, preventing the model from memorizing the order
- The last mini-batch may be smaller than the rest -- that's fine
- Mini-batch GD converges with some oscillation (less than SGD, more than Batch GD)

---

## 4. Momentum

### What Is It?

Momentum adds a "memory" of past gradients to the update step. Instead of updating parameters using only the current gradient, it uses an **exponentially weighted average** of past gradients. This smooths out oscillations and accelerates convergence.

### Intuition

Think of a ball rolling downhill. It builds up speed (momentum) as it rolls. Even if the hill has bumps, the ball's accumulated velocity keeps it moving in the general downhill direction. The "velocity" is the exponentially weighted average of past gradients.

### The Math

For each layer `l = 1, ..., L`:

**Compute velocity** (exponentially weighted average of gradients):
```
v_dW[l] = beta * v_dW[l] + (1 - beta) * dW[l]
v_db[l] = beta * v_db[l] + (1 - beta) * db[l]
```

**Update parameters** using velocity:
```
W[l] = W[l] - alpha * v_dW[l]
b[l] = b[l] - alpha * v_db[l]
```

Where:
- `beta` is the momentum hyperparameter (typically 0.9)
- `v_dW[l]`, `v_db[l]` are the velocity variables (initialized to zeros)
- `alpha` is the learning rate

### Exponentially Weighted Average Explained

When `beta = 0.9`, the velocity roughly averages over the last ~10 gradients:

```
v_t = 0.9 * v_{t-1} + 0.1 * gradient_t
    = 0.1 * gradient_t + 0.09 * gradient_{t-1} + 0.081 * gradient_{t-2} + ...
```

Each past gradient contributes less weight, exponentially decaying by a factor of `beta`.

### Python Implementation

```python
def initialize_velocity(parameters):
    L = len(parameters) // 2
    v = {}
    for l in range(1, L + 1):
        v["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        v["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])
    return v


def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    L = len(parameters) // 2

    for l in range(1, L + 1):
        # Compute velocities
        v["dW" + str(l)] = beta * v["dW" + str(l)] + (1 - beta) * grads["dW" + str(l)]
        v["db" + str(l)] = beta * v["db" + str(l)] + (1 - beta) * grads["db" + str(l)]

        # Update parameters
        parameters["W" + str(l)] -= learning_rate * v["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * v["db" + str(l)]

    return parameters, v
```

### Choosing Beta

| Beta Value | Effect |
|------------|--------|
| `beta = 0` | Equivalent to standard gradient descent (no momentum) |
| `beta = 0.9` | Good default; averages roughly last 10 gradients |
| `beta = 0.99` | Stronger smoothing; averages roughly last 100 gradients |
| `beta > 0.99` | May over-smooth and slow convergence |

### Key Takeaways

- Momentum **dampens oscillations** in directions where gradients keep changing sign
- Momentum **accelerates convergence** in directions where gradients consistently point the same way
- Velocity starts at zero, so it takes a few iterations to "build up"
- Common beta range: 0.8 to 0.999

---

## 5. RMSProp

### What Is It?

RMSProp (Root Mean Square Propagation) adapts the learning rate for each parameter individually. It divides the gradient by the square root of an exponentially weighted average of squared past gradients. This makes updates smaller for parameters with large gradients and larger for parameters with small gradients.

### Intuition

In many optimization landscapes, some directions (parameters) have steep slopes while others are relatively flat. Standard gradient descent takes the same-sized step in all directions, which means:
- It oscillates on steep slopes (too large a step)
- It moves slowly on flat slopes (too small a step)

RMSProp fixes this by **normalizing** each parameter's gradient by its recent magnitude.

### The Math

For each layer `l = 1, ..., L`:

**Compute squared gradient moving average:**
```
s_dW[l] = beta2 * s_dW[l] + (1 - beta2) * (dW[l])^2
s_db[l] = beta2 * s_db[l] + (1 - beta2) * (db[l])^2
```

**Update parameters:**
```
W[l] = W[l] - alpha * dW[l] / (sqrt(s_dW[l]) + epsilon)
b[l] = b[l] - alpha * db[l] / (sqrt(s_db[l]) + epsilon)
```

Where:
- `(dW[l])^2` is element-wise squaring
- `beta2` is typically 0.999
- `epsilon` is a tiny number (e.g., 1e-8) to avoid division by zero

### Why It Works

- If a parameter's gradient is consistently **large**, `s` will be large, so `gradient / sqrt(s)` will be **small** -- damping the update
- If a parameter's gradient is consistently **small**, `s` will be small, so `gradient / sqrt(s)` will be **large** -- boosting the update

### Python Example

```python
def update_with_rmsprop(parameters, grads, s, learning_rate, beta2=0.999, epsilon=1e-8):
    L = len(parameters) // 2

    for l in range(1, L + 1):
        # Update squared gradient average
        s["dW" + str(l)] = beta2 * s["dW" + str(l)] + (1 - beta2) * np.square(grads["dW" + str(l)])
        s["db" + str(l)] = beta2 * s["db" + str(l)] + (1 - beta2) * np.square(grads["db" + str(l)])

        # Update parameters
        parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)] / (np.sqrt(s["dW" + str(l)]) + epsilon)
        parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)] / (np.sqrt(s["db" + str(l)]) + epsilon)

    return parameters, s
```

---

## 6. Adam Optimizer

### What Is It?

Adam (**Ada**ptive **M**oment Estimation) combines the best ideas from **Momentum** and **RMSProp**:
1. **From Momentum**: It tracks an exponentially weighted average of past gradients (first moment) to smooth out the update direction
2. **From RMSProp**: It tracks an exponentially weighted average of past squared gradients (second moment) to adapt the learning rate per parameter
3. **Bias correction**: It corrects for the fact that both averages are initialized to zero and are therefore biased toward zero in early steps

### The Math

For each layer `l = 1, ..., L`:

**Step 1 - First moment estimate (Momentum-like):**
```
v_dW[l] = beta1 * v_dW[l] + (1 - beta1) * dW[l]
v_db[l] = beta1 * v_db[l] + (1 - beta1) * db[l]
```

**Step 2 - Bias correction for first moment:**
```
v_corrected_dW[l] = v_dW[l] / (1 - beta1^t)
v_corrected_db[l] = v_db[l] / (1 - beta1^t)
```

**Step 3 - Second moment estimate (RMSProp-like):**
```
s_dW[l] = beta2 * s_dW[l] + (1 - beta2) * (dW[l])^2
s_db[l] = beta2 * s_db[l] + (1 - beta2) * (db[l])^2
```

**Step 4 - Bias correction for second moment:**
```
s_corrected_dW[l] = s_dW[l] / (1 - beta2^t)
s_corrected_db[l] = s_db[l] / (1 - beta2^t)
```

**Step 5 - Update parameters:**
```
W[l] = W[l] - alpha * v_corrected_dW[l] / (sqrt(s_corrected_dW[l]) + epsilon)
b[l] = b[l] - alpha * v_corrected_db[l] / (sqrt(s_corrected_db[l]) + epsilon)
```

Where:
- `t` = current time step (incremented each update)
- `beta1` = 0.9 (first moment decay rate)
- `beta2` = 0.999 (second moment decay rate)
- `alpha` = learning rate
- `epsilon` = 1e-8 (numerical stability)

### Why Bias Correction?

Since `v` and `s` are initialized to zero, the early estimates are biased toward zero. For example, at t=1:
```
v_1 = 0.9 * 0 + 0.1 * g_1 = 0.1 * g_1  (way too small!)
v_corrected_1 = 0.1 * g_1 / (1 - 0.9^1) = 0.1 * g_1 / 0.1 = g_1  (correct!)
```

As `t` grows, `beta^t` approaches 0, so the correction factor `1/(1 - beta^t)` approaches 1 and has negligible effect.

### Python Implementation

```python
def initialize_adam(parameters):
    L = len(parameters) // 2
    v = {}
    s = {}
    for l in range(1, L + 1):
        v["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        v["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])
        s["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        s["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])
    return v, s


def update_parameters_with_adam(parameters, grads, v, s, t,
                                 learning_rate=0.01, beta1=0.9,
                                 beta2=0.999, epsilon=1e-8):
    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}

    for l in range(1, L + 1):
        # First moment (mean of gradients)
        v["dW" + str(l)] = beta1 * v["dW" + str(l)] + (1 - beta1) * grads["dW" + str(l)]
        v["db" + str(l)] = beta1 * v["db" + str(l)] + (1 - beta1) * grads["db" + str(l)]

        # Bias-corrected first moment
        v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - beta1 ** t)
        v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - beta1 ** t)

        # Second moment (mean of squared gradients)
        s["dW" + str(l)] = beta2 * s["dW" + str(l)] + (1 - beta2) * np.square(grads["dW" + str(l)])
        s["db" + str(l)] = beta2 * s["db" + str(l)] + (1 - beta2) * np.square(grads["db" + str(l)])

        # Bias-corrected second moment
        s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - beta2 ** t)
        s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - beta2 ** t)

        # Update parameters
        parameters["W" + str(l)] -= learning_rate * v_corrected["dW" + str(l)] / (np.sqrt(s_corrected["dW" + str(l)]) + epsilon)
        parameters["b" + str(l)] -= learning_rate * v_corrected["db" + str(l)] / (np.sqrt(s_corrected["db" + str(l)]) + epsilon)

    return parameters, v, s, v_corrected, s_corrected
```

### Adam Hyperparameters

| Hyperparameter | Typical Value | Notes |
|----------------|---------------|-------|
| `alpha` | Needs tuning | The only one you usually tune |
| `beta1` | 0.9 | First moment decay (momentum) |
| `beta2` | 0.999 | Second moment decay (RMSProp) |
| `epsilon` | 1e-8 | Numerical stability; rarely tuned |

### Advantages of Adam

- Relatively low memory requirements (though higher than plain GD or Momentum)
- Usually works well even with minimal hyperparameter tuning (only alpha typically needs tuning)
- Converges faster than other methods in most practical scenarios
- Handles sparse gradients well

---

## 7. Learning Rate Decay and Scheduling

### Why Decay the Learning Rate?

During early training, a large learning rate helps the model move quickly toward the minimum. But as training progresses, a fixed learning rate can cause the model to oscillate around the minimum without converging precisely.

**The fix**: gradually reduce the learning rate so that early steps are large (for fast progress) and later steps are small (for precise convergence).

### 7.1 Exponential Decay (Every Iteration)

**Formula:**
```
alpha = alpha_0 / (1 + decay_rate * epoch_num)
```

**Python Implementation:**
```python
def update_lr(learning_rate0, epoch_num, decay_rate):
    learning_rate = learning_rate0 / (1 + decay_rate * epoch_num)
    return learning_rate
```

**Example:**
```python
alpha_0 = 0.1
decay_rate = 1

# Epoch 0: alpha = 0.1 / (1 + 1*0) = 0.1
# Epoch 1: alpha = 0.1 / (1 + 1*1) = 0.05
# Epoch 2: alpha = 0.1 / (1 + 1*2) = 0.033
# Epoch 3: alpha = 0.1 / (1 + 1*3) = 0.025
```

**Problem**: When applied every iteration, the learning rate drops to near-zero too quickly, especially over thousands of epochs. At epoch 1000: `alpha = 0.1 / 1001 = 0.0001`.

### 7.2 Fixed Interval Scheduling

**Solution**: Only decay the learning rate every `T` epochs (e.g., every 1000 epochs). The learning rate stays constant within each interval.

**Formula:**
```
alpha = alpha_0 / (1 + decay_rate * floor(epoch_num / time_interval))
```

**Python Implementation:**
```python
def schedule_lr_decay(learning_rate0, epoch_num, decay_rate, time_interval=1000):
    learning_rate = learning_rate0 / (1 + decay_rate * np.floor(epoch_num / time_interval))
    return learning_rate
```

**Example:**
```python
alpha_0 = 0.5
decay_rate = 0.3
time_interval = 100

# Epoch 0-99:   floor(epoch/100) = 0 -> alpha = 0.5 / (1 + 0.3*0) = 0.5
# Epoch 100-199: floor(epoch/100) = 1 -> alpha = 0.5 / (1 + 0.3*1) = 0.385
# Epoch 200-299: floor(epoch/100) = 2 -> alpha = 0.5 / (1 + 0.3*2) = 0.3125
# Epoch 300-399: floor(epoch/100) = 3 -> alpha = 0.5 / (1 + 0.3*3) = 0.263
```

This creates a **staircase** pattern where the learning rate is constant within each interval and drops at interval boundaries.

### 7.3 Other Common Decay Schedules (Bonus)

**Exponential decay:**
```
alpha = alpha_0 * decay_rate^epoch_num
```

**Square root decay:**
```
alpha = alpha_0 / sqrt(epoch_num)
```

**Step decay (halving):**
```
alpha = alpha_0 * 0.5^(floor(epoch_num / step_size))
```

### Impact on Different Optimizers

| Optimizer + LR Decay | Accuracy (from assignment) |
|----------------------|---------------------------|
| Mini-Batch GD + Decay | > 94.6% |
| Momentum + Decay | > 95.6% |
| Adam + Decay | ~94% |

Without learning rate decay, GD and Momentum only achieved ~71%, while Adam alone achieved >94%. **Learning rate decay brings simpler optimizers up to Adam-level performance**, though Adam still converges faster.

---

## 8. Comparison of All Methods

### Side-by-Side Summary

| Method | Update Rule | Memory | Convergence Speed | Oscillation |
|--------|------------|--------|-------------------|-------------|
| **Batch GD** | `W -= alpha * dW` | Low | Slow (1 update/epoch) | None |
| **SGD** | Same, but on 1 example | Low | Fast updates, slow convergence | Very high |
| **Mini-Batch GD** | Same, but on B examples | Low | Good balance | Moderate |
| **Momentum** | Uses velocity `v` | +v | Faster than GD | Reduced |
| **RMSProp** | Adapts per-param LR via `s` | +s | Faster, less oscillation | Reduced |
| **Adam** | Combines v + s + bias correction | +v, +s | Fastest in practice | Minimal |

### When to Use What

| Scenario | Recommended |
|----------|-------------|
| Small dataset, simple model | Batch GD or Mini-batch GD |
| Large dataset | Mini-batch GD with Momentum or Adam |
| Default choice / don't know what to pick | **Adam** |
| Sparse data (NLP, embeddings) | Adam |
| You want to fine-tune every last bit | SGD + Momentum + LR scheduling |
| Research / state-of-the-art training | SGD + Momentum (often generalizes better) |

---

## 9. Quick Reference Cheat Sheet

### Gradient Descent Variants

```
Batch GD:     W -= alpha * dW                              (all m examples)
SGD:          W -= alpha * dW                              (1 example at a time)
Mini-Batch:   W -= alpha * dW                              (B examples at a time)
```

### Momentum

```
v = beta * v + (1 - beta) * dW       # Update velocity
W = W - alpha * v                     # Update parameters
                                      # beta = 0.9 typical
```

### RMSProp

```
s = beta2 * s + (1 - beta2) * dW^2   # Update squared gradient avg
W = W - alpha * dW / (sqrt(s) + eps)  # Update parameters
                                      # beta2 = 0.999 typical
```

### Adam

```
v = beta1 * v + (1 - beta1) * dW           # Momentum (first moment)
v_corrected = v / (1 - beta1^t)            # Bias correction
s = beta2 * s + (1 - beta2) * dW^2         # RMSProp (second moment)
s_corrected = s / (1 - beta2^t)            # Bias correction
W = W - alpha * v_corrected / (sqrt(s_corrected) + eps)
```

### Learning Rate Decay

```
# Every-iteration decay
alpha = alpha_0 / (1 + decay_rate * epoch)

# Fixed-interval scheduling
alpha = alpha_0 / (1 + decay_rate * floor(epoch / interval))
```

### Default Hyperparameters

```
learning_rate (alpha):  needs tuning (start with 0.001)
beta  (momentum):       0.9
beta1 (Adam 1st moment): 0.9
beta2 (Adam 2nd moment): 0.999
epsilon:                1e-8
mini_batch_size:        64 (or 128, 256)
```

---

## Worked Example: Putting It All Together

Here is a complete training loop using Adam with mini-batches and learning rate scheduling:

```python
import numpy as np
import math

def train_model(X, Y, layers_dims, learning_rate=0.01, mini_batch_size=64,
                beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=5000,
                decay_rate=0.3, time_interval=1000):

    parameters = initialize_parameters(layers_dims)
    v, s = initialize_adam(parameters)
    t = 0
    learning_rate0 = learning_rate
    m = X.shape[1]

    for epoch in range(num_epochs):
        # Create mini-batches (re-shuffle each epoch)
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed=epoch)
        cost_total = 0

        for (mini_X, mini_Y) in minibatches:
            # Forward propagation
            a, caches = forward_propagation(mini_X, parameters)
            cost_total += compute_cost(a, mini_Y)

            # Backward propagation
            grads = backward_propagation(mini_X, mini_Y, caches)

            # Adam update
            t += 1
            parameters, v, s, _, _ = update_parameters_with_adam(
                parameters, grads, v, s, t,
                learning_rate, beta1, beta2, epsilon
            )

        # Learning rate scheduling (fixed interval)
        learning_rate = schedule_lr_decay(learning_rate0, epoch, decay_rate, time_interval)

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: cost = {cost_total / m:.6f}, lr = {learning_rate:.6f}")

    return parameters
```

---

*Document created for study and reference. Based on Coursera Deep Learning Specialization, Course 2 (Improving Deep Neural Networks), Week 2 Assignment.*
