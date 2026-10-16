"""Tiny optional GLM arithmetic ABI smoke, using a separately rebuilt _ext.
No upstream source is imported into this repository; see decision note commands.
"""
import importlib.metadata
import json

import mlx.core as mx
import numpy as np
import _ext

mx.set_memory_limit(128 * 2**20)
mx.set_cache_limit(16 * 2**20)
rng = np.random.default_rng(53)


def tensor(shape):
    return mx.array(rng.normal(0, 0.1, shape).astype(np.float32)).astype(mx.bfloat16)


def host(a):
    return np.asarray(a.astype(mx.float32))


# Upstream's decode guard requires at least 4096 keys (1 MiB in BF16).
q, k = tensor((1, 32, 1, 128)), tensor((1, 1, 4096, 128))
w = mx.array(rng.uniform(0, 1, (1, 1, 32)).astype(np.float32))
scale = 128**-0.5
scores = _ext.dsa_indexer_scores_decode_fp32(q, k, w, scale)
mx.eval(scores)
mx.synchronize()
reference = (np.maximum(host(q)[0, :, 0] @ host(k)[0, 0].T, 0) *
             host(w)[0, 0, :, None]).sum(axis=0) * scale
score_error = float(np.max(np.abs(host(scores).reshape(-1) - reference)))
np.testing.assert_allclose(host(scores).reshape(-1), reference, atol=2e-5, rtol=2e-4)

# Supported native prefill ABI: 64 heads, L=2, latent512/RoPE64, K=16.
ql, qp = tensor((1, 64, 2, 512)), tensor((1, 64, 2, 64))
kv, kp = tensor((1, 1, 16, 512)), tensor((1, 1, 16, 64))
indices = mx.broadcast_to(mx.arange(16, dtype=mx.uint32), (1, 1, 2, 16))
scale = 576**-0.5
out = _ext.glm_dsa_sparse_mla_attention(ql, qp, kv, kp, indices, scale, causal=False)
mx.eval(out)
mx.synchronize()
logits = (host(ql)[0] @ host(kv)[0, 0].T + host(qp)[0] @ host(kp)[0, 0].T) * scale
probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
probs /= probs.sum(axis=-1, keepdims=True)
reference = probs @ host(kv)[0, 0]
mla_error = float(np.max(np.abs(host(out)[0] - reference)))
np.testing.assert_allclose(host(out)[0], reference, atol=2e-3, rtol=1e-2)
print(json.dumps(dict(mlx=importlib.metadata.version('mlx'),
                      nanobind=importlib.metadata.version('nanobind'),
                      score_max_abs_error=score_error,
                      sparse_mla_max_abs_error=mla_error)))
