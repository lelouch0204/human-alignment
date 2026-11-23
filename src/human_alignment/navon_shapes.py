"""Human-distance evaluation and diagonal metric adaptation on 49 shapes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr

REVISION = 'eabd49541b10745fd2469e4137acefbc859a5a3d'
SOURCE = f'https://raw.githubusercontent.com/georginjacob/Multiscale/{REVISION}/FigureCodes/L2_VSmain.mat'
SOURCE_SHA = '2c4b72f6fe9f37c1848bb1023c67d2b27e0a275171784cd884cd91f7349f86c7'
REPRESENTATIONS = ('pre_logits', 'triplet_logits')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def prepare(root, download=False):
    root.mkdir(parents=True, exist_ok=True)
    source = root / 'L2_VSmain.mat'
    if not source.exists():
        if not download:
            raise FileNotFoundError(f'{source}: use prepare --download-data')
        with urlopen(SOURCE, timeout=120) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != SOURCE_SHA:
            raise ValueError('Downloaded source hash mismatch')
        source.write_bytes(content)
    if sha(source) != SOURCE_SHA:
        raise ValueError('Source hash mismatch')
    if (root / 'manifest.json').exists():
        validated_data(root)
        print(f'Using validated prepared data in {root}')
        return
    data = loadmat(source, simplify_cells=True)['L2_str']
    pairs = np.asarray(data['img_pairs'], dtype=int) - 1
    rt = np.asarray(data['RT']['trial_wise'], dtype=float)
    if pairs.shape != (1176, 2) or rt.shape != (1176, 8, 2):
        raise ValueError('Unexpected source schema')
    if not np.isfinite(rt).all() or np.any(rt <= 0):
        raise ValueError('Unexpected invalid RT; audit exclusions before continuing')
    if len(set(map(tuple, pairs))) != 1176 or set(pairs.flat) != set(range(49)):
        raise ValueError('Invalid image pair coverage')
    from PIL import Image
    image_dir = root / 'images'
    image_dir.mkdir(exist_ok=True)
    images = []
    for i, pixels in enumerate(data['images']):
        if pixels.shape != (400, 400) or pixels.dtype != np.uint8:
            raise ValueError('Unexpected image schema')
        path = image_dir / f'{i:02d}.png'
        Image.fromarray(pixels).save(path)
        images.append(dict(image_id=i, path=str(path.relative_to(root)), sha256=sha(path)))
    # Published trial axis: A-among-B and B-among-A. Average directions first.
    subject_rt = rt.mean(axis=2)
    np.savez_compressed(root / 'human.npz', pairs=pairs, subject_rt=subject_rt,
        human_distance=1 / subject_rt.mean(axis=1), pair_details=data['Image_Pair_Details'])
    write_json(root / 'manifest.json', dict(source=SOURCE, source_sha256=SOURCE_SHA,
        revision=REVISION, human_sha256=sha(root / 'human.npz'), images=images,
        target='1 / mean RT across two directions and eight participants; released consolidated trials; no additional exclusion'))
    print(f'Prepared 49 images and 1176 human-distance targets in {root}')


def validated_data(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['source_sha256'] != SOURCE_SHA or sha(root / 'L2_VSmain.mat') != SOURCE_SHA:
        raise ValueError('Unexpected source fingerprint')
    if sha(root / 'human.npz') != manifest['human_sha256']:
        raise ValueError('Human target file changed')
    for row in manifest['images']:
        if sha(root / row['path']) != row['sha256']:
            raise ValueError('Stimulus image changed')
    return manifest


def embed(args):
    manifest = validated_data(args.data)
    dest = args.cache / f'{args.family}_{args.variant}.npz'
    if dest.exists():
        with np.load(dest, allow_pickle=False) as saved:
            if saved['manifest_sha256'].item() != sha(args.data / 'manifest.json'):
                raise ValueError('Existing embedding cache has different inputs')
            if saved['family'].item() != args.family or saved['variant'].item() != args.variant:
                raise ValueError('Existing cache model mismatch')
        print(f'Using cached {dest}')
        return
    import tensorflow as tf
    from human_alignment.models import VisionModel, load_images
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    model = VisionModel(args.family, args.variant, args.models, download=args.download_models)
    height, width = model.input_spec.shape.as_list()[1:3]
    if height is None or width is None:
        raise ValueError('Export has dynamic dimensions; preprocessing needs review')
    outputs = {rep: [] for rep in REPRESENTATIONS}
    paths = [args.data / row['path'] for row in manifest['images']]
    for start in range(0, len(paths), args.batch_size):
        result = model.predict(load_images(paths[start:start + args.batch_size], size=(width, height)))
        for rep in outputs:
            outputs[rep].append(result[rep])
    args.cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **{rep: np.concatenate(values) for rep, values in outputs.items()},
        image_ids=[row['image_id'] for row in manifest['images']],
        manifest_sha256=sha(args.data / 'manifest.json'), family=args.family, variant=args.variant)
    write_json(dest.with_suffix('.json'), dict(checkpoint=model.name,
        saved_model_sha256=sha(model.path / 'saved_model.pb'), tensorflow=tf.__version__,
        embeddings_sha256=sha(dest), input_dimensions=[height, width],
        preprocessing='models.load_images: RGB, direct OpenCV bicubic resize, float32 /255. Original DINO training preprocessing unresolved.'))


def pair_features(features, pairs):
    features = np.asarray(features, dtype=float)
    lengths = np.linalg.norm(features, axis=1, keepdims=True)
    if not np.isfinite(features).all() or np.any(lengths == 0):
        raise ValueError('Invalid embedding vectors')
    features = features / lengths
    return (features[pairs[:, 0]] - features[pairs[:, 1]]) ** 2


def fit_metric(x, y, regularization):
    """Nonnegative diagonal metric, shrunk toward the unchanged representation."""
    xscale, yscale = x.sum(axis=1).mean(), y.mean()
    if xscale <= 0 or yscale <= 0:
        raise ValueError('Degenerate training distances')
    a, b = x / xscale, y / yscale
    def objective(w):
        residual = a @ w - b
        loss = np.mean(residual ** 2) + regularization * np.mean((w - 1) ** 2)
        grad = 2 * a.T @ residual / len(b) + 2 * regularization * (w - 1) / len(w)
        return loss, grad
    result = minimize(objective, np.ones(x.shape[1]), jac=True, method='L-BFGS-B',
        bounds=[(0, None)] * x.shape[1], options=dict(maxiter=2000, ftol=1e-10))
    if not result.success:
        raise RuntimeError(f'Metric fitting failed: {result.message}')
    return result.x, xscale, yscale


def image_folds(pairs, n_images, seed=2026):
    groups = np.array_split(np.random.default_rng(seed).permutation(n_images), 7)
    for fold in range(7):
        test, validation = groups[fold], groups[(fold + 1) % 7]
        train = np.setdiff1d(np.arange(n_images), np.r_[test, validation])
        masks = [np.isin(pairs, ids).all(axis=1) for ids in (train, validation, test)]
        yield train, validation, test, masks


def correlation(y, prediction):
    if len(y) < 3 or np.ptp(y) == 0 or np.ptp(prediction) == 0:
        return dict(pearson=None, spearman=None)
    return dict(pearson=float(pearsonr(y, prediction).statistic),
                spearman=float(spearmanr(y, prediction).statistic))


def adapt(x, y, pairs, n_images, penalties=(.01, .1, 1., 10.)):
    folds, weights, predictions = [], [], []
    for fold, (train_ids, val_ids, test_ids, masks) in enumerate(image_folds(pairs, n_images)):
        train_mask, val_mask, test_mask = masks
        scores = []
        for penalty in penalties:
            w, xs, ys = fit_metric(x[train_mask], y[train_mask], penalty)
            scores.append(float(np.mean((x[val_mask] @ w / xs - y[val_mask] / ys) ** 2)))
        penalty = penalties[int(np.argmin(scores))]
        # Refit on train+validation images, including their cross-pairs; no test images.
        refit = np.isin(pairs, np.r_[train_ids, val_ids]).all(axis=1)
        w, xs, ys = fit_metric(x[refit], y[refit], penalty)
        indices = np.flatnonzero(test_mask)
        raw = x[test_mask].sum(axis=1) / xs * ys
        fitted = x[test_mask] @ w / xs * ys
        folds.append(dict(fold=fold, train_ids=train_ids.tolist(), validation_ids=val_ids.tolist(),
            test_ids=test_ids.tolist(), penalty=penalty, validation_mse=scores,
            refit_distance_scale=float(xs), refit_target_scale=float(ys),
            frozen=correlation(y[test_mask], raw), adapted=correlation(y[test_mask], fitted)))
        weights.append(w)
        for index, base, pred in zip(indices, raw, fitted):
            predictions.append(dict(fold=fold, pair_index=int(index), image_i=int(pairs[index, 0]),
                image_j=int(pairs[index, 1]), human=float(y[index]), frozen=float(base), adapted=float(pred)))
    return folds, np.stack(weights), predictions


def analyze(args):
    manifest = validated_data(args.data)
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('Choose an empty --out directory to preserve previous results')
    args.out.mkdir(parents=True, exist_ok=True)
    with np.load(args.data / 'human.npz') as data:
        pairs, y = data['pairs'], data['human_distance']
    report = dict(source=SOURCE, manifest_sha256=sha(args.data / 'manifest.json'),
                  primary_representation='pre_logits', frozen={}, adaptation={}, cache_sha256={})
    for family in ('dino', 'siglip'):
        for variant in ('base', 'unalignet', 'alignet'):
            path = args.cache / f'{family}_{variant}.npz'
            report['cache_sha256'][path.name] = sha(path)
            with np.load(path, allow_pickle=False) as saved:
                if saved['manifest_sha256'].item() != sha(args.data / 'manifest.json'):
                    raise ValueError('Embedding manifest mismatch')
                if saved['family'].item() != family or saved['variant'].item() != variant:
                    raise ValueError('Embedding checkpoint mismatch')
                if saved['image_ids'].tolist() != [r['image_id'] for r in manifest['images']]:
                    raise ValueError('Embedding image order mismatch')
                for rep in REPRESENTATIONS:
                    features = saved[rep]
                    if features.ndim != 2 or len(features) != len(manifest['images']):
                        raise ValueError('Embedding shape mismatch')
                    x = pair_features(features, pairs)
                    key = f'{family}_{variant}_{rep}'
                    # Unit-normalized Euclidean distance for the frozen evaluation.
                    report['frozen'][key] = correlation(y, np.sqrt(x.sum(axis=1)))
                    np.savez_compressed(args.out / f'{key}-distances.npz', pairs=pairs,
                                        human=y, frozen=np.sqrt(x.sum(axis=1)))
                    if variant == 'base' or args.adapt_all:
                        folds, weights, predictions = adapt(x, y, pairs, len(features))
                        report['adaptation'][key] = dict(folds=folds, test_pair_count=len(predictions),
                            mean_fold_spearman={name: float(np.mean([r[name]['spearman'] for r in folds]))
                                if all(r[name]['spearman'] is not None for r in folds) else None
                                for name in ('frozen', 'adapted')})
                        np.savez_compressed(args.out / f'{key}-weights.npz', weights=weights)
                        write_json(args.out / f'{key}-predictions.json', predictions)
    report['method'] = ('Frozen: normalized Euclidean distance vs inverse mean human RT. '
        'Adaptation: nonnegative diagonal squared-distance readout fit on '
        + ('all checkpoints; ' if args.adapt_all else 'Base only; ') +
        '7 image-disjoint folds, seed 2026, validation penalties [0.01,0.1,1,10], '
        '147 held-out pairs in total. No inferential confidence intervals; pairs share images.')
    write_json(args.out / 'report.json', report)
    print(f'Report saved to {args.out / "report.json"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'embed', 'analyze'))
    parser.add_argument('--data', type=Path, default=Path('outputs/navon_shapes/data'))
    parser.add_argument('--cache', type=Path, default=Path('outputs/navon_shapes/embeddings'))
    parser.add_argument('--out', type=Path, default=Path('outputs/navon_shapes/results'))
    parser.add_argument('--family', choices=('dino', 'siglip'), default='dino')
    parser.add_argument('--variant', choices=('base', 'unalignet', 'alignet'), default='base')
    parser.add_argument('--models', default='models')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--download-data', action='store_true')
    parser.add_argument('--download-models', action='store_true')
    parser.add_argument('--adapt-all', action='store_true', help='Fit the readout separately for each checkpoint')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('Batch size must be positive')
    if args.command == 'prepare':
        prepare(args.data, args.download_data)
    elif args.command == 'embed':
        embed(args)
    else:
        analyze(args)


if __name__ == '__main__':
    main()
