"""Separate-reference cosine prototypes for proposal research question 3."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

CATEGORIES = tuple(sorted('airplane bear bicycle bird boat bottle car cat chair clock dog elephant keyboard knife oven truck'.split()))
FAMILIES = ('dino', 'siglip')
VARIANTS = ('base', 'unalignet', 'alignet')
REPRESENTATIONS = ('pre_logits', 'triplet_logits')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_manifest(path):
    """Require explicit source identities; hashes alone cannot detect derivatives."""
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if not data.get('source_audit') or not data.get('dataset_provenance'):
        raise ValueError('Document source_audit and dataset_provenance before inference')
    excluded = set(data['benchmark_source_ids'])
    if not excluded:
        raise ValueError('Supply all known clean/conflict benchmark source IDs')
    rows = data['images']
    ids, hashes, sources = set(), {}, {}
    for row in rows:
        if row['image_id'] in ids:
            raise ValueError('Duplicate image_id')
        ids.add(row['image_id'])
        if row['split'] not in ('reference', 'test') or row['category'] not in CATEGORIES:
            raise ValueError('Invalid split or category')
        if not row['source_id'] or not row.get('provenance'):
            raise ValueError('Every image requires source_id and provenance')
        image = (path.parent / row['path']).resolve()
        if digest(image) != row['sha256']:
            raise ValueError(f'Image hash mismatch: {image}')
        if row['sha256'] in hashes:
            raise ValueError('Duplicate image bytes; deduplicate before evaluation')
        hashes[row['sha256']] = row['image_id']
        previous = sources.setdefault(row['source_id'], (row['split'], row['category']))
        if previous != (row['split'], row['category']):
            raise ValueError('Source reused across splits or categories')
        if row['split'] == 'reference' and row['source_id'] in excluded:
            raise ValueError('Benchmark source or derivative in reference set')
        row['_path'] = str(image)
    for split in ('reference', 'test'):
        if {r['category'] for r in rows if r['split'] == split} != set(CATEGORIES):
            raise ValueError(f'All 16 categories required in {split}')
    return rows


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    lengths = np.linalg.norm(x, axis=-1, keepdims=True)
    if not np.isfinite(x).all() or np.any(lengths == 0):
        raise ValueError('Nonfinite or zero representation/prototype')
    return x / lengths


def classify(x, rows):
    if x.ndim != 2 or len(x) != len(rows):
        raise ValueError('Embedding shape does not match manifest')
    unit(x)  # Validate all inputs, including individual reference vectors.
    # Average raw embeddings as specified in proposal, then cosine-normalize.
    prototypes = unit(np.stack([
        x[[i for i, r in enumerate(rows) if r['split'] == 'reference' and r['category'] == c]].mean(axis=0)
        for c in CATEGORIES
    ]))
    indices = [i for i, r in enumerate(rows) if r['split'] == 'test']
    scores = unit(x[indices]) @ prototypes.T
    predictions = np.asarray(CATEGORIES)[scores.argmax(axis=1)]
    truth = np.asarray([rows[i]['category'] for i in indices])
    return indices, predictions, predictions == truth


def paired_summary(correct, rows, draws=5000, seed=2026):
    """Category-stratified source bootstrap, conditional on fixed references."""
    rng = np.random.default_rng(seed)
    category = np.asarray([r['category'] for r in rows])
    source = np.asarray([r['source_id'] for r in rows])
    estimates, boot = [], np.zeros((draws, correct.shape[1]))
    for c in CATEGORIES:
        groups = [np.flatnonzero((category == c) & (source == s)) for s in sorted(set(source[category == c]))]
        totals = np.stack([correct[g].sum(axis=0) for g in groups])
        sizes = np.asarray([len(g) for g in groups])
        estimates.append(totals.sum(axis=0) / sizes.sum())
        chosen = rng.integers(len(groups), size=(draws, len(groups)))
        boot += totals[chosen].sum(axis=1) / sizes[chosen].sum(axis=1)[:, None] / len(CATEGORIES)
    estimate = np.mean(estimates, axis=0)
    result = {'balanced_accuracy': dict(zip(VARIANTS, estimate.tolist())), 'paired_differences': {}}
    for a, b in ((1, 0), (2, 1)):
        result['paired_differences'][f'{VARIANTS[a]}_minus_{VARIANTS[b]}'] = {
            'estimate': float(estimate[a] - estimate[b]),
            'ci95': np.quantile(boot[:, a] - boot[:, b], [.025, .975]).tolist(),
        }
    return result


def embed(args, rows):
    import tensorflow as tf
    from human_alignment.models import VisionModel, load_images
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    model = VisionModel(args.family, args.variant, args.models, download=args.download)
    shape = model.input_spec.shape.as_list()
    height, width = shape[1:3]
    if height is None or width is None:
        raise ValueError('Export has no fixed spatial dimensions; confirm preprocessing first')
    values = {key: [] for key in REPRESENTATIONS}
    for start in range(0, len(rows), args.batch_size):
        batch = load_images([r['_path'] for r in rows[start:start + args.batch_size]], size=(width, height))
        outputs = model.predict(batch)
        for key in values:
            values[key].append(outputs[key])
    target = args.out / f'{args.family}_{args.variant}.npz'
    np.savez_compressed(target, image_ids=[r['image_id'] for r in rows],
                        manifest_sha256=digest(args.manifest),
                        **{key: np.concatenate(value) for key, value in values.items()})
    target.with_suffix('.json').write_text(json.dumps({
        'checkpoint': model.name, 'saved_model_sha256': digest(model.path / 'saved_model.pb'),
        'manifest_sha256': digest(args.manifest), 'embeddings_sha256': digest(target),
        'tensorflow': tf.__version__, 'input_shape': shape,
        'preprocessing': 'models.load_images: EXIF transpose, RGB, OpenCV INTER_CUBIC direct resize, float32 /255; DINO training provenance unresolved',
    }, indent=2) + '\n')


def analyze(args, rows):
    report, predictions, checksums = {}, [], {}
    for family in FAMILIES:
        vectors = {}
        for variant in VARIANTS:
            path = args.out / f'{family}_{variant}.npz'
            checksums[path.name] = digest(path)
            with np.load(path, allow_pickle=False) as saved:
                if saved['manifest_sha256'].item() != digest(args.manifest):
                    raise ValueError('Embeddings were produced from a different manifest')
                if saved['image_ids'].tolist() != [r['image_id'] for r in rows]:
                    raise ValueError('Embedding image IDs/order do not match manifest')
                vectors[variant] = {rep: saved[rep] for rep in REPRESENTATIONS}
        for rep in REPRESENTATIONS:
            correctness = []
            for variant in VARIANTS:
                indices, predicted, correct = classify(vectors[variant][rep], rows)
                correctness.append(correct)
                for i, label, ok in zip(indices, predicted, correct):
                    predictions.append(dict(family=family, variant=variant, representation=rep,
                        image_id=rows[i]['image_id'], source_id=rows[i]['source_id'],
                        category=rows[i]['category'], prediction=label, correct=int(ok)))
            report[f'{family}/{rep}'] = paired_summary(np.stack(correctness, axis=1), [rows[i] for i in indices], args.bootstrap)
    with (args.out / 'predictions.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    (args.out / 'report.json').write_text(json.dumps({
        'manifest_sha256': digest(args.manifest), 'embedding_sha256': checksums,
        'bootstrap_draws': args.bootstrap, 'bootstrap_seed': 2026,
        'uncertainty': 'Paired category-stratified source bootstrap; fixed reference set; exploratory 95% intervals, no multiplicity adjustment',
        'results': report}, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('validate', 'embed', 'analyze'))
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('outputs/clean_recognition'))
    parser.add_argument('--family', choices=FAMILIES)
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--models', default='models')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--bootstrap', type=int, default=5000)
    args = parser.parse_args()
    if args.batch_size < 1 or args.bootstrap < 1:
        parser.error('Batch size and bootstrap draws must be positive')
    if args.command == 'embed' and (args.family is None or args.variant is None):
        parser.error('embed requires --family and --variant')
    rows = validate_manifest(args.manifest)
    if args.command == 'validate':
        print(f'Validated {len(rows)} images; source provenance still requires human review.')
        return
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == 'embed':
        embed(args, rows)
    else:
        analyze(args, rows)


if __name__ == '__main__':
    main()
