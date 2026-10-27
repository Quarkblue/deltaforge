# deltaforge

**A hand-written CUDA inference engine that runs a 2B-parameter vision-language model end to end on one 8 GB laptop GPU, with no PyTorch, ONNX Runtime or llama.cpp underneath it.**

| | |
|---|---|
| **Owner** | Divyansh Chauhan |
| **Language** | C++17 / CUDA |
| **Target** | NVIDIA RTX 4060 Laptop (sm_89) |
| **Status** | Phase 3 of 11 |
| **Revised** | 2026-10-10 |

---

## Summary

An inference engine is the program that actually runs a trained neural network: it loads the weights, performs the arithmetic on the GPU, and emits tokens. Production engines hide that arithmetic behind large frameworks. deltaforge implements it directly, one operation at a time, as a way of learning modern model architecture at the level where the numbers are.

The target model is Qwen3.5-2B, chosen because it is deliberately awkward. It mixes two different attention mechanisms, carries a vision tower, and has a speculative-decoding head. Nothing in it is cut from scope. A simpler model would have taught less.

Every component is checked against HuggingFace Transformers running the same model, stage by stage, before it is trusted. That comparison harness is as much the product as the kernels are.

---

## Goals

- **Run the whole model.** Text generation first, then image input, then the multi-token prediction head. Text is the first milestone, not the boundary.
- **Own every line of the inference path.** The only third-party code on the hot path is cuBLAS for matrix multiplication. Weight loading, memory management, all element-wise kernels, attention, the recurrent state updates, sampling and tokenization are written here.
- **Fit the hardware honestly.** 8 GB of VRAM, 4.24 GiB of which is weights. No offloading to host memory, no quantization in version one.
- **Prove correctness numerically.** Every stage matches a reference activation from HuggingFace within a recorded tolerance, and every check carries a deliberately wrong variant that must fail.
- **Be fast enough to use.** Interactive single-stream decoding on the target laptop.

### Non-goals

- Training, fine-tuning or backpropagation.
- Multi-GPU, multi-node, or batched serving of concurrent users. Batch size is fixed at one.
- A general framework. This engine runs one architecture well rather than any model badly.
- Beating vLLM or TensorRT-LLM on throughput.
- Portability in version one. Windows and Ada-generation consumer GPUs are the only supported target.

---

## Capabilities

| Capability | What it means | Milestone |
|---|---|---|
| Weight loading | Memory-map a safetensors file, parse its header, upload one contiguous slab to the GPU | Done |
| Text generation | Prompt in, tokens out, greedy and sampled, with a key-value cache | Phase 6 |
| Tokenization | Byte-level BPE in C++, reading the model's own vocabulary and merge rules | Phase 6.5 |
| Image input | Vision transformer tower plus a projection into the text embedding space | Phase 8 |
| Mixed image and text positions | Multimodal rotary embeddings, so pictures and words share a position scheme | Phase 9 |
| Speculative decoding | The model's multi-token prediction head, for throughput | Phase 10 |

---

## Architecture

Four layers, each independently testable. The loader and the test harness are shared by every phase; kernels are added one component at a time.

| Layer | Responsibility | Key files |
|---|---|---|
| **Model loading** | Map the weight file without reading it into the heap, resolve 632 tensors by name and data type, upload once, expose typed views per layer | `safetensors.hpp`, `model.hpp`, `mapped_file.hpp` |
| **Runtime** | One device allocation for activations with bump allocation and per-token reset, persistent caches, a single cuBLAS handle and stream | `arena.hpp`, `device_buffer.hpp`, `context.hpp`, `tensor.hpp` |
| **Kernels** | Roughly ten hand-written kernels plus cuBLAS for every matrix multiply | `kernel.cu`, `kernels.hpp`, `bf16.hpp` |
| **Verification** | Reference activations captured from HuggingFace, a comparison routine, and a test binary that re-runs every earlier check | `dump_oracle.py`, `compare.hpp`, `tests/` |

### The kernel set

Deliberately small. Matrix multiplication goes to cuBLAS because hand-written GEMM would be slower and teaches less than the rest of the list. Everything else is written here:

- embedding lookup
- RMS normalization, plain and gated
- rotary position embedding
- fused SiLU and multiply
- sigmoid output gate
- masked softmax
- residual addition
- causal depthwise convolution step
- gated delta-rule state update
- argmax sampling

---

## Verification method

> **A wrong kernel usually produces the right shape.**
> Reordering attention heads, scaling a norm by the wrong tensor, or rotating too many dimensions all return a correctly shaped tensor full of plausible numbers. Shape assertions and eyeball checks on generated text both miss them. So every stage is compared against a recorded reference, and every comparison ships with a wrong variant that must fail.

1. **Capture.** Run HuggingFace once on a fixed prompt with hooks on every module, saving each intermediate tensor to a safetensors file. 196 tensors, 49 MiB.
2. **Match precision.** The reference runs in the same number format as the engine, and upcasts to 32-bit in exactly the places the reference does, so a difference means a bug rather than rounding.
3. **Compare.** Each kernel's output is copied back and compared against its reference stage in double precision, reporting relative error, the worst element, and any disagreement about infinities or not-a-number values.
4. **Falsify.** Each check has a sibling that must differ: the naive tensor split, the wrong normalization weight, the wrong key-value head expansion. If a wrong variant starts passing, the test itself has stopped working.
5. **Regress.** One command re-runs every stage from every earlier phase and exits with the number of failures, so a kernel added late cannot quietly break an early one.

Components are built in single-token stepping form first, including the prefill pass, which runs as a loop over that same step. Correctness comes before the batched and chunked forms that make it fast, and those are then checked against the step version.

---

## Technical specifications

### Platform

| | |
|---|---|
| **GPU** | NVIDIA RTX 4060 Laptop, compute capability 8.9, 24 streaming multiprocessors, 8 GB GDDR6, about 6.4 GB free for the engine |
| **Host** | Windows 11, MSVC 19.44 |
| **Toolchain** | CUDA 13.3, CMake 3.18 or newer, C++17, Visual Studio 2022 generator |
| **Dependencies** | cuBLAS and a single-header JSON parser. Python with PyTorch and Transformers is used only to produce reference activations, never at runtime. |

### Engine parameters

| | |
|---|---|
| **Weights** | One safetensors shard, 4,548,144,832 bytes (4.236 GiB), 632 tensors: 596 bfloat16, 36 float32. Data type is read per tensor. |
| **Numerics** | bfloat16 activations and weights; normalization, softmax and the recurrent state in float32, matching the reference implementation |
| **Batch size** | 1 |
| **Max context** | 32,768 tokens, limited by cache memory rather than the model, which allows 262,144 |

### Model under test

The engine is written for one architecture. It has 24 decoder layers of 2048 dimensions over a 248,320-token vocabulary, and the layers are not all alike: six use grouped-query attention with 8 query heads against 2 key-value heads, and the other eighteen are gated delta networks, a recurrent linear-attention design that keeps a fixed-size state instead of a growing cache. On top of that sit a 24-block vision transformer and a multi-token prediction head. The output projection shares its weights with the input embedding table.

### Memory budget

Batch size 1, bfloat16 activations, 32,768-token context.

| Allocation | Size | Basis |
|---|---:|---|
| Weights, single slab | 4338 MiB | Measured file size |
| Key-value cache | 384 MiB | 12 KiB per token across 6 attention layers, at full context |
| Recurrent states | 18 MiB | 18 gated delta layers, fixed size, float32 |
| Convolution states | 0.84 MiB | 18 layers of depthwise convolution history |
| Activation arena | 16 MiB | Reserved; 1.26 MiB measured in use |
| cuBLAS workspace | 32 MiB | Reserved |
| **Total** | **4789 MiB** | About 1.7 GiB of headroom |

Only the key-value cache and the attention score buffer grow with context length.

### Performance targets

Single-stream decoding is limited by memory bandwidth, not arithmetic: each token reads roughly 3.7 GiB of weights, against about 270 GB/s of bandwidth on this card. That puts the ceiling near 70 tokens per second and makes the target 30 or better, with the shared output projection over the 248,320-token vocabulary as the largest single read.

*These are estimates from the hardware specification, not measurements. Real numbers are measured in phase 7 and recorded here.*

---

## Roadmap

| Phase | Deliverable | State |
|---:|---|---|
| 0 | Build toolchain, CUDA compiling and running | Done |
| 1 | Weight loading, typed registry, single-slab upload with self-checks | Done |
| 2 | Reference implementations of the bracketing operations and the attention block, checked against HuggingFace | Done |
| 3 | Reference activation dumps, runtime skeleton, test harness, first three kernels | **Active** |
| 4 | Full-attention decoder layer in stepping form, with its key-value cache | Next |
| 5 | Gated delta network layer: projections, convolution state, delta-rule update | Planned |
| 6 | All 24 layers plus greedy decoding, matching the reference token for token | Planned |
| 6.5 | Byte-level BPE tokenizer in C++ | Planned |
| 7 | Speed: batched prefill, chunked recurrence, measured tokens per second | Planned |
| 8 | Vision tower and projection into the text stream | Planned |
| 9 | Multimodal position embeddings and image-text fusion | Planned |
| 10 | Multi-token prediction head and speculative decoding | Planned |

Phase 6 is the first milestone a person can watch: text coming out of the engine.

---

## Measured so far

| Check | Result | Meaning |
|---|---:|---|
| All 632 tensors claimed exactly once, upload verified byte for byte | Exact | Nothing is misnamed, overlapped or left behind |
| Reference dumps, both precisions | 196 tensors | 49 MiB and 80 MiB, regenerated by one command, reloaded bit for bit |
| Number-format conversion against PyTorch | 65,536 / 65,536 | Every bfloat16 bit pattern, including rounding ties and not-a-number cases |
| Activation arena, 10,000 reset cycles | 0 allocations | No device allocation happens inside the decode loop |
| Peak activation memory per step | 1.26 MiB | Against a 16 MiB reservation |
| Attention block against the reference (phase 2) | 4.9e-07 | Relative error three orders inside its budget |
| 16-bit run against the 32-bit anchor, end of stack | 4.1e-03 | The precision floor: anything near it is a real bug, not rounding |

---

## Risks

- **The gated delta network is the hard part.** It is a recurrent state update with gating and a delta rule, and the reference uses two different formulations for prefill and decode. *Mitigation:* build the single-token form first and treat the chunked version as an optimization validated against it.
- **Memory headroom is thin.** 4.24 GiB of weights on an 8 GB card that is also driving a display. *Mitigation:* a single slab allocation, a measured activation budget, and no duplicate buffers.
- **Precision drift accumulates over 24 layers.** *Mitigation:* the reference runs in the engine's own number format, tolerances are relative rather than absolute, and a per-layer drift table shows where divergence starts.
- **Correct arithmetic, wrong plumbing.** Reversed head order or a mis-offset cache mask passes any shape check. *Mitigation:* wrong-variant tests alongside every comparison.
- **Tokenizer mismatch.** A byte-level BPE implementation that differs on one edge case produces fluent nonsense. *Mitigation:* compare token ids against the reference tokenizer over a corpus, not a handful of sentences.

---

## Definition of done

1. The engine loads the model and generates text from a prompt with no Python in the process.
2. Greedy output matches the reference implementation token for token on a fixed prompt.
3. It accepts an image and answers questions about it.
4. Decode speed and peak memory are measured and recorded on the target laptop.
5. One command re-runs every stage comparison and reports zero failures.

---

## Repository

```
include/   C++ headers: loader, runtime, kernels interface, comparison
src/       entry point and CUDA kernels
tests/     stage-by-stage comparisons (deltaforge_test)
python/    HuggingFace reference dumper
models/    weights (not committed)
dumps/     reference activations (not committed, regenerated)
docs/      PROJECT.md decision log and per-session briefs
```

Build with CMake using the Visual Studio 2022 generator. `infer` runs the engine; `deltaforge_test` runs every stage comparison and is registered with CTest.

Development runs as one session per component, each seeded by a brief under `docs/sessions/` that records what to research, what counts as done, and what went wrong. `docs/PROJECT.md` holds the decision log.
