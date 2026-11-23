import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from human_alignment.clean_recognition import CATEGORIES, analyze, classify, digest, paired_summary, validate_manifest


class CleanRecognitionTests(unittest.TestCase):
    def test_prototypes_average_raw_vectors_before_cosine(self):
        rows, vectors = [], []
        for i, category in enumerate(CATEGORIES):
            for split in ('reference', 'reference', 'test'):
                rows.append(dict(category=category, split=split))
                vectors.append(np.eye(16)[i])
        vectors[0] = 10 * np.eye(16)[0]
        vectors[1] = np.eye(16)[1]
        vectors[2] = np.eye(16)[0] + 2 * np.eye(16)[1]
        _, predicted, correct = classify(np.asarray(vectors), rows)
        self.assertEqual(predicted[0], CATEGORIES[1])
        self.assertTrue(correct[1:].all())

    def test_paired_bootstrap_identical_models_has_zero_difference(self):
        rows = [dict(category=c, source_id=f'{c}{i}') for c in CATEGORIES for i in range(4)]
        correct = np.tile(np.asarray([True, False, True, True]), 16)
        report = paired_summary(np.stack([correct] * 3, axis=1), rows, draws=100)
        self.assertEqual(report['balanced_accuracy']['base'], .75)
        for result in report['paired_differences'].values():
            self.assertEqual(result, dict(estimate=0., ci95=[0., 0.]))

    def test_manifest_rejects_source_leakage_and_changed_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for c in CATEGORIES:
                for split in ('reference', 'test'):
                    name = f'{c}-{split}'
                    path = root / name
                    path.write_text(name)
                    rows.append(dict(image_id=name, path=name, sha256=digest(path), category=c,
                                     split=split, source_id=name, provenance='unit-test fixture'))
            data = dict(images=rows, benchmark_source_ids=[f'{c}-test' for c in CATEGORIES],
                        source_audit='fixture', dataset_provenance='fixture')
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps(data))
            self.assertEqual(len(validate_manifest(manifest)), 32)
            rows[0]['source_id'] = data['benchmark_source_ids'][0]
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'Benchmark source'):
                validate_manifest(manifest)
            rows[0]['source_id'] = rows[0]['image_id']
            manifest.write_text(json.dumps(data))
            (root / rows[0]['path']).write_text('modified')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                validate_manifest(manifest)

    def test_analysis_checks_identity_and_writes_all_comparisons(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / 'manifest.json'
            manifest.write_text('{}')
            rows = [dict(image_id=f'{c}-{split}', category=c, split=split,
                         source_id=f'{c}-{split}')
                    for c in CATEGORIES for split in ('reference', 'test')]
            vectors = np.repeat(np.eye(16), 2, axis=0)
            for family in ('dino', 'siglip'):
                for variant in ('base', 'unalignet', 'alignet'):
                    np.savez(root / f'{family}_{variant}.npz',
                             image_ids=[r['image_id'] for r in rows],
                             manifest_sha256=digest(manifest),
                             pre_logits=vectors, triplet_logits=vectors)
            args = SimpleNamespace(out=root, manifest=manifest, bootstrap=20)
            analyze(args, rows)
            report = json.loads((root / 'report.json').read_text())
            self.assertEqual(len(report['results']), 4)
            self.assertEqual(len((root / 'predictions.csv').read_text().splitlines()), 193)
            manifest.write_text('{"changed": true}')
            with self.assertRaisesRegex(ValueError, 'different manifest'):
                analyze(args, rows)

    def test_zero_vectors_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'zero representation'):
            classify(np.zeros((1, 16)), [dict(split='test', category=CATEGORIES[0])])


if __name__ == '__main__':
    unittest.main()
