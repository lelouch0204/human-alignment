"""Frozen-backbone Navon behavioral alignment pilot and letter-label control."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

LEVELS = ('global', 'local')
SPLITS = ('train', 'validation', 'test')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_manifest(path):
    """Validate curated, aggregated human responses; never infer source groups."""
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    for field in ('provenance', 'split_description', 'source_audit'):
        if not isinstance(data.get(field), str) or not data[field].strip():
            raise ValueError(f'Missing {field}')
    labels = data['labels']
    if len(labels) < 2 or len(set(labels)) != len(labels) or not all(isinstance(x, str) for x in labels):
        raise ValueError('labels must be unique response strings')
    stimuli = data['stimuli']
    ids, groups, hashes = {}, {}, {}
    for i, row in enumerate(stimuli):
        if row['image_id'] in ids:
            raise ValueError('Duplicate image_id')
        ids[row['image_id']] = i
        if row['split'] not in SPLITS:
            raise ValueError('Invalid split')
        for field in ('source_id', 'study', 'condition'):
            if not row.get(field):
                raise ValueError(f'Missing stimulus {field}')
        for field in ('global_letter', 'local_letter'):
            if row[field] not in labels:
                raise ValueError(f'{field} not in response labels')
        # All renderings of a source must carry the same canonical source_id.
        previous = groups.setdefault(row['source_id'], row['split'])
        if previous != row['split']:
            raise ValueError('Source appears in multiple splits')
        image = (path.parent / row['path']).resolve()
        if sha256(image) != row['sha256']:
            raise ValueError(f'Image hash mismatch: {image}')
        previous = hashes.setdefault(row['sha256'], row['split'])
        if previous != row['split']:
            raise ValueError('Identical image bytes appear in multiple splits')
        row['_path'] = str(image)
    seen, coverage = set(), set()
    for row in data['responses']:
        if row['image_id'] not in ids or row['level'] not in LEVELS:
            raise ValueError('Invalid response image_id or level')
        key = (row['image_id'], row['level'])
        if key in seen:
            raise ValueError('Aggregate repeated trials before supplying responses')
        seen.add(key)
        counts = row['counts']
        if len(counts) != len(labels) or any(type(n) is not int or n < 0 for n in counts) or sum(counts) == 0:
            raise ValueError('counts must be nonnegative integers with positive total, in labels order')
        coverage.add((stimuli[ids[row['image_id']]]['split'], row['level']))
    if coverage != {(split, level) for split in SPLITS for level in LEVELS}:
        raise ValueError('Each split requires responses at both target levels')
    if {r['image_id'] for r in data['responses']} != set(ids):
        raise ValueError('Every stimulus must have responses')
    for level in LEVELS:
        observed = {stimuli[ids[r['image_id']]][f'{level}_letter'] for r in data['responses']
                    if r['level'] == level and stimuli[ids[r['image_id']]]['split'] == 'train'}
        if observed != set(labels):
            raise ValueError(f'Training targets must cover all labels for {level}')
    return data


def normalized(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError('Expected a two-dimensional embedding matrix')
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(x).all() or np.any(norm == 0):
        raise ValueError('Expected finite, nonzero embedding rows')
    return x / norm


def design(data, embeddings):
    """Two instruction-specific linear heads on the same frozen features."""
    features = normalized(embeddings)
    indices = {r['image_id']: i for i, r in enumerate(data['stimuli'])}
    x, human, truth, rows = [], [], [], []
    for response in data['responses']:
        stimulus = data['stimuli'][indices[response['image_id']]]
        feature = np.r_[features[indices[response['image_id']]], 1.]
        empty = np.zeros_like(feature)
        x.append(np.r_[feature, empty] if response['level'] == 'global' else np.r_[empty, feature])
        counts = np.asarray(response['counts'], dtype=float)
        human.append(counts / counts.sum())
        truth.append(data['labels'].index(stimulus[f"{response['level']}_letter"]))
        rows.append({k: v for k, v in (stimulus | response).items() if not k.startswith('_')})
    return np.asarray(x), np.asarray(human), np.asarray(truth), rows


def log_softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def loss_gradient(x, targets, weights, l2):
    logp = log_softmax(x @ weights)
    loss = -(targets * logp).sum(axis=1).mean() + .5 * l2 * np.square(weights).sum()
    gradient = x.T @ (np.exp(logp) - targets) / len(x) + l2 * weights
    return float(loss), gradient


def fit(x, targets, train, validation, *, seed=0, epochs=500, learning_rate=.03, l2=.001):
    """Fixed-budget full-batch Adam; select epoch using validation targets only."""
    weights = np.random.default_rng(seed).normal(scale=.01, size=(x.shape[1], targets.shape[1]))
    first, second = np.zeros_like(weights), np.zeros_like(weights)
    best, best_loss, best_epoch, history = weights.copy(), float('inf'), 0, []
    for epoch in range(1, epochs + 1):
        train_loss, gradient = loss_gradient(x[train], targets[train], weights, l2)
        first = .9 * first + .1 * gradient
        second = .999 * second + .001 * gradient ** 2
        weights -= learning_rate * (first / (1 - .9 ** epoch)) / (np.sqrt(second / (1 - .999 ** epoch)) + 1e-8)
        validation_loss, _ = loss_gradient(x[validation], targets[validation], weights, 0.)
        if not np.isfinite(train_loss + validation_loss):
            raise ValueError('Training diverged')
        history.append(dict(epoch=epoch, training_loss=train_loss, validation_cross_entropy=validation_loss))
        if validation_loss < best_loss:
            best, best_loss, best_epoch = weights.copy(), validation_loss, epoch
    return best, best_epoch, history


def evaluate(probabilities, human, truth, rows):
    logp = np.log(np.clip(probabilities, 1e-300, 1.))
    entropy = -(human * np.log(np.clip(human, 1e-300, 1.))).sum(axis=1)
    cross_entropy = -(human * logp).sum(axis=1)
    predicted = probabilities.argmax(axis=1)
    result = {}
    masks = {'all': np.ones(len(rows), dtype=bool)}
    for level in LEVELS:
        mask = np.asarray([r['level'] == level for r in rows])
        masks[level] = mask
        congruent = np.asarray([r['global_letter'] == r['local_letter'] for r in rows])
        masks[f'{level}/congruent'] = mask & congruent
        masks[f'{level}/incongruent'] = mask & ~congruent
    for name, mask in masks.items():
        if not mask.any():
            result[name] = None
            continue
        result[name] = dict(n_stimulus_tasks=int(mask.sum()),
            human_cross_entropy=float(cross_entropy[mask].mean()),
            human_kl=float((cross_entropy - entropy)[mask].mean()),
            letter_accuracy=float((predicted[mask] == truth[mask]).mean()),
            human_letter_accuracy=float(human[np.arange(len(rows)), truth][mask].mean()),
            expected_human_agreement=float(human[np.arange(len(rows)), predicted][mask].mean()))
    return result


def extract(args, data):
    import tensorflow as tf
    from human_alignment.models import VisionModel, load_images
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    model = VisionModel(args.family, 'base', args.models, download=args.download)
    shape = model.input_spec.shape.as_list()
    height, width = shape[1:3]
    if height is None or width is None:
        raise ValueError('Confirm preprocessing for export with dynamic spatial dimensions')
    chunks = []
    for start in range(0, len(data['stimuli']), args.batch_size):
        paths = [row['_path'] for row in data['stimuli'][start:start + args.batch_size]]
        chunks.append(model.embed(load_images(paths, size=(width, height)), args.representation))
    np.savez_compressed(args.embeddings, features=np.concatenate(chunks),
        image_ids=[r['image_id'] for r in data['stimuli']], manifest_sha256=sha256(args.manifest),
        family=args.family, variant='base', representation=args.representation)
    args.embeddings.with_suffix('.metadata.json').write_text(json.dumps(dict(
        checkpoint=model.name, saved_model_sha256=sha256(model.path / 'saved_model.pb'),
        tensorflow=tf.__version__, input_shape=shape, embeddings_sha256=sha256(args.embeddings),
        preprocessing='models.load_images: EXIF transpose, RGB, OpenCV INTER_CUBIC direct resize, float32 /255; DINO training preprocessing provenance unresolved'), indent=2) + '\n')


def train(args, data):
    with np.load(args.embeddings, allow_pickle=False) as saved:
        expected = dict(manifest_sha256=sha256(args.manifest), family=args.family,
                        variant='base', representation=args.representation)
        for key, value in expected.items():
            if saved[key].item() != value:
                raise ValueError(f'Embedding metadata mismatch: {key}')
        if saved['image_ids'].tolist() != [r['image_id'] for r in data['stimuli']]:
            raise ValueError('Embedding image order differs from manifest')
        features = saved['features']
    if features.ndim != 2 or len(features) != len(data['stimuli']):
        raise ValueError('Embedding shape differs from manifest')
    x, human, truth, rows = design(data, features)
    masks = {split: np.asarray([r['split'] == split for r in rows]) for split in SPLITS}
    one_hot = np.eye(len(data['labels']))[truth]
    results = {}
    for seed in args.seeds:
        for objective, targets in (('human', human), ('letter', one_hot)):
            weights, epoch, history = fit(x, targets, masks['train'], masks['validation'],
                seed=seed, epochs=args.epochs, learning_rate=args.learning_rate, l2=args.l2)
            key = f'{objective}-seed{seed}'
            test_rows = [r for r in rows if r['split'] == 'test']
            probabilities = np.exp(log_softmax(x[masks['test']] @ weights))
            metrics = evaluate(probabilities, human[masks['test']], truth[masks['test']], test_rows)
            results[key] = dict(selected_epoch=epoch, test=metrics)
            np.savez_compressed(args.out / f'{key}.npz', weights=weights, labels=data['labels'],
                                family=args.family, representation=args.representation,
                                feature_normalization='unit L2; append intercept; global block then local block')
            predictions = [dict(image_id=r['image_id'], source_id=r['source_id'], level=r['level'],
                                study=r['study'], condition=r['condition'],
                                global_letter=r['global_letter'], local_letter=r['local_letter'],
                                counts=r['counts'], probabilities=p.tolist()) for r, p in zip(test_rows, probabilities)]
            (args.out / f'{key}.json').write_text(json.dumps(dict(history=history, predictions=predictions), indent=2) + '\n')
    (args.out / 'report.json').write_text(json.dumps(dict(
        method='Frozen official Base backbone; two instruction-specific linear softmax heads; equal stimulus-task weighting',
        manifest_sha256=sha256(args.manifest), embeddings_sha256=sha256(args.embeddings),
        configuration={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        labels=data['labels'], split_counts={k: int(v.sum()) for k, v in masks.items()},
        results=results), indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('validate', 'embed', 'train'))
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--family', choices=('dino', 'siglip'), default='dino')
    parser.add_argument('--representation', choices=('pre_logits', 'triplet_logits'), default='pre_logits')
    parser.add_argument('--models', default='models')
    parser.add_argument('--embeddings', type=Path, default=Path('outputs/navon_human/dino_base_pre_logits.npz'))
    parser.add_argument('--out', type=Path, default=Path('outputs/navon_human/pilot'))
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--learning-rate', type=float, default=.03)
    parser.add_argument('--l2', type=float, default=.001)
    args = parser.parse_args()
    if args.batch_size < 1 or args.epochs < 1 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0 or not np.isfinite(args.l2) or args.l2 < 0:
        parser.error('Require positive batch size, epochs, learning rate and nonnegative L2')
    if any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error('Seeds must be distinct nonnegative integers')
    data = read_manifest(args.manifest)
    if args.command == 'validate':
        print(f"Validated {len(data['stimuli'])} stimuli and {len(data['responses'])} stimulus-task response rows")
    elif args.command == 'embed':
        args.embeddings.parent.mkdir(parents=True, exist_ok=True)
        extract(args, data)
    else:
        if args.out.exists() and any(args.out.iterdir()):
            parser.error('Output directory is not empty; choose a new --out to preserve prior runs')
        args.out.mkdir(parents=True, exist_ok=True)
        train(args, data)


if __name__ == '__main__':
    main()
