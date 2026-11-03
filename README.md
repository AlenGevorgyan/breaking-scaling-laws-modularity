# Modular Neural Networks on Compositional CIFAR-10

A PyTorch implementation of modular neural networks on **Compositional CIFAR-10**, inspired by
*Breaking Neural Network Scaling Laws with Modularity* (Boopathy et al., ICLR 2025,
[arXiv:2409.05780](https://arxiv.org/abs/2409.05780)). It uses a data-driven module initialization in
which each module learns, from data alone, which part of the input to attend to.

## The task

Each input is **k CIFAR-10 images** flattened and joined end to end (input dimension 3072·k). The target
is a k-hot vector of size 10·k: one class label for each image. The task is modular by
construction, because each block of outputs depends on only one part of the input.

We compare three models:

| Model | Description |
|---|---|
| **Monolithic** | Fully connected MLP (512-512-256-128) with BatchNorm |
| **Modular (random init)** | 16 modules. Each one projects the input with its own matrix Û_j (input_dim × 256), then passes the result through a **shared** small MLP (256-128-64). A final linear layer combines the module outputs |
| **Ours (module init)** | Same architecture, but every Û_j is initialized from data so that it focuses on one image slot |

## Method

### Module initialization (data-driven; image positions are never hard-coded)

For each output block j (the 10 logits of image j):

1. **Relevance screening.** Compute the input–output cross-covariance C_j = Cov(x, y_j). The relevance
   of input coordinate c is r_{j,c} = Σ_classes C_{j,c}². From the data alone, this puts **>99%** of
   its mass on the correct image.
2. **Mask.** m_j ∝ r_j. It stays fixed during training so each module stays focused.
3. **Projection.** Û_j = the top 256 principal directions of the masked input m_j ⊙ x. Every image slot
   has the same statistics, so the PCA bases found for different slots are nearly identical up to the
   shift. That lets the *shared* module network be shared between positions.
4. **Whitening.** Each projected feature is rescaled to unit variance, so no single direction dominates
   training.
5. **Round-robin assignment.** Module i serves block i mod k, so every image slot gets modules. Extra
   replicas of a slot get a random rotation of the same basis.

The only thing the method uses is the output block structure y = [y₁, …, y_k]. It finds where each
image sits inside x from the data.

### Implementation details

- All modules go through the shared net as **one stacked batch** (`einsum`), so BatchNorm statistics
  are consistent between training and evaluation.
- All three methods use Adam with **lr = 1e-3**, cosine schedule, BCE loss and gradient clipping at 1.0.
- CIFAR images are transformed once and cached for fast data loading.

## Results (k = 3)

| Setting | Monolithic | Modular (random) | Ours |
|---|---|---|---|
| Clean training data | 32.0% | 33.3% | **39.4%** |
| Training noise σ = 3 | — | 32.2% | **34.4%** |

Best test accuracy per image, averaged over the k images. Training set: 5,000 samples (3,000 for the
noise run). Test set: 1,000 samples. 10 epochs.

Two other observations:

- The random modular net reaches **~99% train** accuracy but only **33% test**. Its 37M free
  projection weights let it memorize instead of learning per-image structure.
- With the new init, every module puts **>99%** of its weight on its own image, compared with 33%
  (chance) for random init.

### Caveats

- These results were produced with a NumPy re-implementation of exactly the same models and training
  loop. They still need to be confirmed by running the PyTorch notebook.
- Each number is a **single run**. Differences under ~2–3 points (such as the noise row) should be
  confirmed over several seeds.
- k = 2, k = 4 and the out-of-distribution (unseen class combinations) experiment have not been run yet.
- The initialization differs from the kernel-based procedure in the paper, so this is a variant
  inspired by the paper, not a reproduction of it.

## How to run

```bash
pip install torch torchvision numpy matplotlib tqdm
jupyter notebook modular-nn-compositional-cifar10-fixed.ipynb
```

CIFAR-10 downloads automatically to `./data`. A GPU is recommended: the modular model has about
38M parameters for k = 3.

### Notebook structure

1. Setup and imports
2. Compositional CIFAR-10 dataset
3. Model architectures (monolithic, modular)
4. Module initialization (relevance-masked, whitened PCA)
5. Training and evaluation utilities
6. Main experiment: accuracy vs. number of images k
7. Noise robustness
8. Out-of-distribution: unseen class combinations
9. Visualization of module projections
10. Summary dashboard

## Next steps

- Run 3+ seeds and report mean ± std
- Larger k (5, 6, 8) and smaller training sets, where the gap should grow
- Convolutional modules
- Freeze Û for the first epochs and train only the shared net

## Reference

```bibtex
@inproceedings{boopathy2025breaking,
  title     = {Breaking Neural Network Scaling Laws with Modularity},
  author    = {Akhilan Boopathy and Sunshine Jiang and William Yue and Jaedong Hwang and Abhiram Iyer and Ila Fiete},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025}
}
```
