# Sequence Parallelism Bug Fixes

Fixes applied to `ms-swift` sequence parallelism implementation (Ulysses + Zigzag Ring Attention).

---

## 1. [CRITICAL] Loss normalization inflated by `world_size`

**File:** `swift/trainers/seq2seq_trainer.py` (line ~198)

**Bug:** `per_token_loss_func_sp` gathers per-token losses across ALL SP ranks via `GatherLoss.apply`, so `outputs.loss` contains losses for the full (un-split) sequence. However, `num_items_in_batch` was computed from `labels` which is the local shard (1/world_size of the full sequence). This means:

```
effective_loss = full_sequence_loss_sum / local_shard_token_count
```

The loss is inflated by approximately `world_size`, which is equivalent to multiplying the learning rate by the SP degree. At SP=8, training is 8x more aggressive than intended.

**Fix:** Scale `num_items_in_batch` by `sequence_parallel_size` when SP is active, so the denominator matches the gathered numerator. This applies whether `num_items_in_batch` is computed locally or passed in from the trainer.

---

## 2. [HIGH] Missing float upcast in SP cross-entropy

**File:** `swift/trainers/utils.py` (line ~76)

**Bug:** The non-SP path (`per_token_loss_func`) explicitly upcasts logits to float32 before computing cross-entropy:
```python
logits = logits.float()
```

The SP path (`per_token_loss_func_sp`) skipped this upcast, passing bf16/fp16 logits directly into `CrossEntropyLoss`. With large vocabularies (130k+ for Qwen, LLaMA-3, etc.), the softmax computation in bf16 overflows, producing NaN or inf losses. This bug only manifests in the SP training path, making it hard to diagnose.

**Fix:** Added `logits = logits.float()` before flattening, matching the non-SP path.

---

## 3. [HIGH] Bare `raise` with no active exception

**File:** `swift/sequence_parallel/zigzag_ring_attn.py` (lines 475, 489, 499, 535)

**Bug:** NaN detection checks in the backward pass use bare `raise`:
```python
if block_out.isnan().any() or block_lse.isnan().any():
    raise
```

`raise` with no active exception context produces `RuntimeError: No active exception to re-raise`, which is confusing and hides the actual problem (NaN in ring attention backward). Debugging distributed NaN issues is already hard enough without misleading error messages.

**Fix:** Replaced all four bare `raise` statements with descriptive `RuntimeError` messages that include which tensor is NaN and at which ring communication step, e.g.:
```python
raise RuntimeError(f'NaN detected in block_out or block_lse at backward step {step}')
```

---

## 4. [MEDIUM] Hardcoded `seed=42` in `SequenceParallelSampler`

**File:** `swift/trainers/mixin.py` (line 1179)

**Bug:** The SP data sampler was always constructed with `seed=42`, ignoring the user-configured `args.data_seed`. All SP experiments produce the same data shuffle order regardless of seed configuration, breaking reproducibility and making ablation studies unreliable.

**Fix:** Changed to `seed=self.args.data_seed or 42`, so user-specified seeds are respected while preserving the default behavior when no seed is set.

---

## 5. [LOW] `shape[-0]` typo in forward assertions

**File:** `swift/sequence_parallel/zigzag_ring_attn.py` (lines 198-199)

**Bug:** Assertions used `shape[-0]` instead of `shape[0]`:
```python
assert k.shape[-0] == cu_seqlens_kv[-1]
```

In Python, `-0 == 0`, so `shape[-0]` evaluates to `shape[0]` — accidentally correct. The backward pass (line 248-249) correctly uses `shape[0]` without the minus sign, confirming this was unintentional.

**Fix:** Changed to `shape[0]` for clarity and consistency with the backward pass.

---

## Known issues NOT fixed (require deeper refactoring)

These were identified but left unfixed to avoid breaking changes:

- **`gather()` ring index math for mixed-length packed batches** (`ulysses.py:548`): `accumulated_length * rp_world_size` assumes equal padded sub-sequence lengths. Incorrect for variable-length packed batches with ring attention. Fixing this requires reworking the gather/split index logic.

- **`_global_inited` prevents correct multi-model SP** (`ulysses.py:445`): In RLHF (DPO/GRPO), `prepare()` is called for both policy and reference models. The second call skips device mesh init but overwrites `num_heads`, potentially causing `num_heads % sp_world_size != 0`. Fixing requires per-model SP state tracking.

- **`ChunkedCrossEntropyLoss.backward` in-place mutation** (`utils.py:103`): Overwrites the saved `logits` tensor, breaking `retain_graph=True` (used in some gradient accumulation or second-order gradient scenarios).

- **`GatherLoss.backward` scaling for ring+Ulysses** (`utils.py:57`): Scales gradient by `world_size` then does two-level split. Mathematically should cancel, but warrants verification with specific shapes for the combined ring+Ulysses case.
