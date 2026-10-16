"""Small regression checks for the non-admitting M2 preflight."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

from tools.m2_preflight import CONFIG, EXPECTED_SET_ID, inspect, required_paths, safetensors_extent


def tensor_file(path):
    header = json.dumps({'x': {'dtype': 'U8', 'shape': [1], 'data_offsets': [0, 1]}}).encode()
    path.write_bytes(struct.pack('<Q', len(header)) + header + b'\0')


class PreflightTests(unittest.TestCase):
    def test_only_main_up1_paths(self):
        paths = required_paths()
        self.assertEqual(len(paths), len(set(paths)))
        self.assertIn('experts/exlr8-layer-077.k4.safetensors', paths)
        self.assertIn('experts/exlr8-layer-003.index.safetensors', paths)
        self.assertFalse(any('078' in p or '.k2.' in p or '.k3.' in p for p in paths))

    def test_missing_checkpoint_is_not_admitted(self):
        with tempfile.TemporaryDirectory() as d:
            result = inspect(Path(d))
            self.assertFalse(result['admitted'])
            self.assertIn('manifests/exlr8-manifest-up1.json', result['missing'])
            self.assertIn('tokenizer.json', result['missing'])
            self.assertIsNone(result['verified_weight_set_id'])

    def test_extent_and_truncation(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'x.safetensors'
            tensor_file(p)
            self.assertEqual(safetensors_extent(p), p.stat().st_size)
            p.write_bytes(p.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, 'extent'):
                safetensors_extent(p)
            p.write_bytes(struct.pack('<Q', 2**40))
            with self.assertRaisesRegex(ValueError, 'header length'):
                safetensors_extent(p)

    def test_even_all_present_cannot_grant_admission(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in required_paths():
                p = root / name
                p.parent.mkdir(parents=True, exist_ok=True)
                if name.endswith('.safetensors'):
                    tensor_file(p)
                else:
                    p.write_text('{}')
            (root / 'config.json').write_text(json.dumps(CONFIG))
            manifest = root / 'manifests/exlr8-manifest-up1.json'
            manifest.write_text(json.dumps({'set_id': EXPECTED_SET_ID}))
            (root / 'hashes').mkdir()
            (root / 'hashes/unverified').touch()
            result = inspect(root)
            self.assertEqual(result['status'], 'REQUIRES_VERIFICATION')
            self.assertFalse(result['admitted'])
            self.assertFalse(result['verified_payload_hashes'])
            manifest.write_text(json.dumps({'set_id': 'wrong'}))
            result = inspect(root)
            self.assertTrue(any('set_id' in e for e in result['errors']))
            (root / 'config.json').write_text('{')
            self.assertTrue(any('config.json' in e for e in inspect(root)['errors']))


if __name__ == '__main__':
    unittest.main()
