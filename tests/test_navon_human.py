import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from human_alignment.navon_human import (
    design, evaluate, fit, log_softmax, loss_gradient, read_manifest, sha256, train,
)


def fixture(root):
    stimuli, responses = [], []
    for split in ('train', 'validation', 'test'):
        for g in ('H', 'S'):
            for l in ('H', 'S'):
                name = f'{split}-{g}{l}'
                path = root / name
                path.write_text(name)  # Parser fixture only; never used as image data.
                stimuli.append(dict(image_id=name, source_id=name, study='fixture',
                    condition=split, split=split, path=name, sha256=sha256(path),
                    global_letter=g, local_letter=l))
                for level, letter in (('global', g), ('local', l)):
                    responses.append(dict(image_id=name, level=level,
                        counts=[8, 2] if letter == 'H' else [2, 8]))
    data = dict(labels=['H', 'S'], stimuli=stimuli, responses=responses,
                provenance='synthetic software test', source_audit='fixture', split_description='fixture')
    manifest = root / 'manifest.json'
    manifest.write_text(json.dumps(data))
    features = np.asarray([[1 if r['global_letter'] == 'H' else -1,
                            1 if r['local_letter'] == 'H' else -1] for r in stimuli])
    return manifest, data, features


class NavonHumanTests(unittest.TestCase):
    def test_soft_target_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(10)
        x = rng.normal(size=(7, 4))
        targets = rng.dirichlet([2, 2, 2], size=7)
        weights = rng.normal(size=(4, 3))
        _, gradient = loss_gradient(x, targets, weights, .03)
        numerical = np.zeros_like(weights)
        for i, j in np.ndindex(weights.shape):
            delta = np.zeros_like(weights)
            delta[i, j] = 1e-5
            numerical[i, j] = (loss_gradient(x, targets, weights + delta, .03)[0]
                               - loss_gradient(x, targets, weights - delta, .03)[0]) / 2e-5
        np.testing.assert_allclose(gradient, numerical, atol=1e-9)

    def test_manifest_rejects_source_leakage_and_invalid_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest, data, _ = fixture(Path(temp))
            self.assertEqual(len(read_manifest(manifest)['stimuli']), 12)
            data['stimuli'][4]['source_id'] = data['stimuli'][0]['source_id']
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'multiple splits'):
                read_manifest(manifest)
            data['stimuli'][4]['source_id'] = 'validation-HH'
            data['responses'][0]['counts'] = [0, 0]
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'counts'):
                read_manifest(manifest)

    def test_instruction_heads_and_test_targets_do_not_affect_fit(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest, _, features = fixture(Path(temp))
            data = read_manifest(manifest)
            x, human, truth, rows = design(data, features)
            self.assertTrue(np.all(x[0, 3:] == 0))
            self.assertTrue(np.all(x[1, :3] == 0))
            train_mask = np.array([r['split'] == 'train' for r in rows])
            valid_mask = np.array([r['split'] == 'validation' for r in rows])
            test_mask = ~(train_mask | valid_mask)
            weights, _, _ = fit(x, human, train_mask, valid_mask, epochs=200)
            altered = human.copy()
            altered[test_mask] = altered[test_mask, ::-1]
            other, _, _ = fit(x, altered, train_mask, valid_mask, epochs=200)
            np.testing.assert_array_equal(weights, other)
            probabilities = np.exp(log_softmax(x[test_mask] @ weights))
            report = evaluate(probabilities, human[test_mask], truth[test_mask],
                              [r for r in rows if r['split'] == 'test'])
            self.assertEqual(report['all']['letter_accuracy'], 1.)
            self.assertLess(report['all']['human_kl'], .002)

    def test_end_to_end_cached_embedding_training_and_metadata_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, _, features = fixture(root)
            embeddings = root / 'embeddings.npz'
            data = read_manifest(manifest)
            np.savez(embeddings, features=features,
                image_ids=[r['image_id'] for r in data['stimuli']],
                manifest_sha256=sha256(manifest), family='dino', variant='base', representation='pre_logits')
            out = root / 'results'
            out.mkdir()
            args = SimpleNamespace(manifest=manifest, embeddings=embeddings, family='dino',
                representation='pre_logits', out=out, seeds=[0, 1], epochs=100, learning_rate=.03, l2=.001)
            train(args, data)
            report = json.loads((out / 'report.json').read_text())
            self.assertEqual(len(report['results']), 4)
            for name in report['results']:
                saved = json.loads((out / f'{name}.json').read_text())
                self.assertEqual(len(saved['predictions']), 8)
                self.assertTrue(all(r['image_id'].startswith('test-') for r in saved['predictions']))
            args.representation = 'triplet_logits'
            with self.assertRaisesRegex(ValueError, 'metadata mismatch'):
                train(args, data)

    def test_matching_distribution_has_zero_kl(self):
        human = np.array([[.8, .2], [.1, .9]])
        rows = [dict(level='global', global_letter='H', local_letter='S'),
                dict(level='local', global_letter='S', local_letter='S')]
        metrics = evaluate(human, human, np.array([0, 1]), rows)
        self.assertAlmostEqual(metrics['all']['human_kl'], 0.)
        self.assertIsNone(metrics['global/congruent'])
        self.assertAlmostEqual(metrics['all']['human_letter_accuracy'], .85)


if __name__ == '__main__':
    unittest.main()
