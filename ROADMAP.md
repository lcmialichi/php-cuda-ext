# php-cuda-ext Roadmap

This document describes the project's direction. It's a living document —
suggestions are welcome via
[GitHub Discussions](https://github.com/lcmialichi/php-cuda-ext/discussions).

## Vision

Make PHP a first-class language for GPU-accelerated computing, enabling web
applications, data pipelines, and scientific research to run heavy numerical
workloads without leaving the PHP ecosystem.

**Guiding principles:**

1. **No Python dependency** or bindings to external frameworks
2. **Native PHP syntax and semantics** — no strange DSLs
3. **Explicit control** over GPU execution (no implicit transfers)
4. **Transparency and performance** over "magical" abstractions
5. **Build fundamental primitives**, not a black box

## Short Term (next 3 months)

Focus: **stability and foundation**

### API & Stability
- [ ] Freeze the public `CudaArray` API (v0.1.0)
- [ ] Define Semantic Versioning (SemVer) policy
- [ ] Clearly mark experimental APIs with `@internal` or `_experimental` suffix

### Quality & Testing
- [ ] Reach ≥ 70% test coverage for core operations
- [ ] Add regression tests for known bugs
- [ ] Set up CI that builds the extension (even without a GPU)
- [ ] Add memory leak tests (valgrind/ASan)

### Documentation
- [ ] Translate README and CONTRIBUTING to other languages (if contributors step up)
- [ ] Create `docs/getting-started.md` with a full tutorial
- [ ] Document architectural decisions (ADRs) under `docs/adr/`
- [ ] Add examples for every supported operation

### Community
- [ ] Create 10+ issues labeled `good first issue`
- [ ] Enable GitHub Discussions
- [ ] Create a Discord/Matrix server
- [ ] Publish the first technical blog post

## Mid Term (3 to 12 months)

Focus: **expanding capabilities**

### Operations & Types
- [ ] Optimized matmul (using cuBLAS)
- [ ] More activation functions: `tanh`, `sigmoid`, `relu`, `softmax`
- [ ] Linear algebra ops: `dot`, `norm`, basic `einsum`
- [ ] Support `complex64` / `complex128`
- [ ] Slicing and advanced indexing

### Interoperability
- [ ] Multi-GPU support
- [ ] Explicit CUDA streams for parallelism
- [ ] Integration with PHP's `ffi` for advanced use cases
- [ ] Export/import in common formats (NumPy `.npy`, safetensors)

### Tooling
- [ ] Built-in profiler (kernel time, memory usage)
- [ ] `php-cuda` CLI for common tasks
- [ ] Debug mode with shape/bounds checking

### Performance
- [ ] Reduce CPU↔GPU transfer overhead (pinned memory, streams)
- [ ] Automatic kernel fusion for chained expressions
- [ ] Public, comparable benchmark suite

## Long Term (1+ year)

Focus: **ecosystem**

### Standard Library
- [ ] `php-cuda/nn` — neural network layers (linear, conv, pooling)
- [ ] `php-cuda/optim` — optimizers (SGD, Adam, RMSprop)
- [ ] `php-cuda/data` — data pipeline, augmentation
- [ ] `php-cuda/io` — common dataset loaders

### Integrations
- [ ] Windows support (at least via WSL2) and macOS (via eGPU if viable)
- [ ] Bindings to advanced CUDA libraries: cuDNN, cuBLAS, cuSPARSE
- [ ] Plugins for PHP frameworks (Laravel, Symfony) for ML tasks

### Education
- [ ] Extensive book/docs: "GPU Computing in PHP"
- [ ] Courses and workshops
- [ ] Academic papers using the project

### Sustainability
- [ ] Seek funding (GitHub Sponsors, NLnet, Sovereign Tech Fund)
- [ ] Establish governance with multiple maintainers
- [ ] Annual community events

## How to Participate

Any item here can become an issue. If you want to work on something:

1. Open an issue referencing the roadmap item
2. Use the `roadmap` label to track it
3. Read [CONTRIBUTING.md](CONTRIBUTING.md)

## Out of Scope (for now)

To stay focused, we do **not** plan to:

- Reimplement PyTorch or TensorFlow in PHP
- Support non-NVIDIA GPUs (AMD, Intel) — the focus is CUDA
- Compete with Python on raw ML performance in production
- Provide "magical" abstractions that hide the GPU

The goal is to offer **solid foundations** for the community to build on.