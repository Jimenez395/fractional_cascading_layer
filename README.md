# fractional_cascading_layer

A list-pair layer for fractional cascading: each layer stores a sorted list and a reference to the next layer, so locating `x` in the first layer yields `x`'s position in the next layer in O(1) additional comparisons.

## Usage

```python
from fractional_cascading_layer import FCLayer

second = FCLayer([10, 20, 30])
first = FCLayer([1, 2, 3], next_layer=second)

# Search the current layer.
r = first.search(2)
assert r.found and r.position == 1

# Locate x in the next layer in O(1) additional comparisons.
rn = first.locate_in_next(20)
assert rn.found and rn.position == 1

# Walk the whole chain, one result per layer.
results = first.chain_search(3)
assert len(results) == 2
```

## Why this exists

The problem is repetitive binary search across multiple sorted lists: searching for `x` in each of `k` lists independently costs `O(k log n)`. Fractional cascading cuts this to `O(log n + k)` by augmenting each layer with a sample of the next layer's sorted list, so the position found in layer `i` bounds the position in layer `i + 1` to a two-element window that a single comparison closes.

The trade-off here is simplicity over textbook fidelity: instead of materialising the augmented lists (every second element of the next layer merged in), each layer holds a direct reference to the next layer's underlying sorted list. The behavioural promise — O(1) additional comparisons to resolve the next-layer position once the current layer has been searched — is preserved, and the lookup math stays easy to audit. The augmentation is effectively delegated to the cross-layer reference rather than precomputed into each layer's own list.

## Edge cases

- **Duplicates are rejected at construction.** Every position must have a unique owner, so duplicate keys raise `ValueError`. This is the one constraint a caller will trip over if they feed in raw, un-deduplicated data.
- **Empty layers are supported**, including as the next layer: `locate_in_next` returns `found=False, position=0`.
- The input sequence is **snapshotted and sorted** at construction; mutating the original afterwards has no effect.
