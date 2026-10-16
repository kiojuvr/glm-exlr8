"""CPU-only M2 checkpoint preflight; NEVER grants model-load admission.

Exit 2: incomplete/invalid inputs. Exit 3: structurally ready, but full manifest,
record and publisher-hash verification is still required. No weights are loaded.
Original implementation; no reference repository source imported.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import struct

EXPECTED_SET_ID = '2011c0cedde962d19fc43e74c1d409279e398d40d27a6fdb4bcb7b49ea2ea6fb'
MODEL_ROOT = '/Volumes/KIOXIA-PRO-1/models/kindlingai/glm-5.3-exlr8-k2-k3-k4'
CONFIG = dict(model_type='glm_moe_dsa', architectures=['GlmMoeDsaForCausalLM'],
              num_hidden_layers=78, hidden_size=6144, vocab_size=154880,
              first_k_dense_replace=3, n_routed_experts=256, n_shared_experts=1,
              num_experts_per_tok=8, kv_lora_rank=512, qk_rope_head_dim=64,
              index_head_dim=128, index_n_heads=32)


def required_paths():
    # Main backbone only. Layer 78, alternate widths and TP slices are optional.
    return (['config.json', 'generation_config.json', 'tokenizer.json',
             'tokenizer_config.json', 'chat_template.jinja', 'LICENSE',
             'manifests/exlr8-manifest-up1.json', 'dense/exlr8-vocab.safetensors'] +
            [f'dense/exlr8-dense-{i:03}.safetensors' for i in range(78)] +
            [f'experts/exlr8-layer-{i:03}.{kind}.safetensors'
             for i in range(3, 78) for kind in ('k4', 'index')])


def safetensors_extent(path):
    """Check header bounds and declared file extent, NOT payload authenticity."""
    size = path.stat().st_size
    with path.open('rb') as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError('truncated length prefix')
        length = struct.unpack('<Q', prefix)[0]
        if not 2 <= length <= min(16 * 2**20, size - 8):
            raise ValueError('invalid/oversized header length')
        header = json.loads(f.read(length))
    if not isinstance(header, dict):
        raise ValueError('header is not an object')
    offsets = []
    for name, info in header.items():
        if name == '__metadata__':
            continue
        begin, end = info['data_offsets']
        if not (type(begin) is int and type(end) is int and 0 <= begin <= end):
            raise ValueError('invalid tensor offsets')
        offsets.append(end)
    if not offsets or 8 + length + max(offsets) != size:
        raise ValueError('declared tensor extent differs from file size')
    return size


def inspect(root):
    required = required_paths()
    missing, errors, present_bytes = [], [], 0
    for name in required:
        path = root / name
        if not path.is_file():
            missing.append(name)
            continue
        present_bytes += path.stat().st_size
        if name.endswith('.safetensors'):
            try:
                safetensors_extent(path)
            except (ValueError, KeyError, TypeError, OSError) as e:
                errors.append(f'{name}: {e}')
    if (root / 'config.json').is_file():
        try:
            config = json.loads((root / 'config.json').read_text())
            for key, expected in CONFIG.items():
                if config.get(key) != expected:
                    errors.append(f'config.{key}: expected {expected!r}, got {config.get(key)!r}')
        except (ValueError, AttributeError, OSError) as e:
            errors.append(f'config.json: {e}')
    manifest = root / 'manifests/exlr8-manifest-up1.json'
    claimed_set_id = None
    if manifest.is_file():
        try:
            claimed_set_id = json.loads(manifest.read_text()).get('set_id')
            if claimed_set_id != EXPECTED_SET_ID:
                errors.append('manifest set_id differs from planning target')
        except (ValueError, AttributeError, OSError) as e:
            errors.append(f'manifest: {e}')
    if not (root / 'hashes').is_dir() or not any((root / 'hashes').rglob('*')):
        errors.append('hashes/: missing/empty publisher hash metadata')
    partials = list((root / '.cache').rglob('*.incomplete'))
    metadata_hashes = {}
    for name in ('config.json', 'chat_template.jinja'):
        if (root / name).is_file():
            metadata_hashes[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    environment = {}
    for name in ('mlx', 'mlx-metal', 'nanobind'):
        try:
            environment[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            environment[name] = None
    return dict(status='BLOCKED_INCOMPLETE_OR_INVALID' if missing or errors else 'REQUIRES_VERIFICATION',
                admitted=False, root=str(root), environment=environment,
                required_file_count=len(required), present_required_file_count=len(required) - len(missing),
                present_required_bytes=present_bytes, missing=missing, errors=errors,
                expected_set_id=EXPECTED_SET_ID, manifest_claimed_set_id=claimed_set_id,
                verified_weight_set_id=None, verified_payload_hashes=False,
                local_metadata_sha256=metadata_hashes,
                incomplete_download_count=len(partials),
                incomplete_download_bytes=sum(p.stat().st_size for p in partials if p.is_file()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', default=MODEL_ROOT, type=Path)
    args = parser.parse_args()
    result = inspect(args.root)
    print(json.dumps(result, indent=2))
    return 2 if result['missing'] or result['errors'] else 3


if __name__ == '__main__':
    raise SystemExit(main())
