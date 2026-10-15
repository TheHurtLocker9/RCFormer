"""Content identities for isolated runs; never reuse results by filename alone."""
import hashlib
import json
from pathlib import Path


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def source_hash():
    root = Path(__file__).resolve().parent
    return json_hash({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted(root.rglob('*.py')) if '__pycache__' not in p.parts})


def run_identity(cfg, bundle, split):
    return {
        'protocol_version': cfg['protocol_version'], 'run_kind': cfg['run_kind'],
        'dataset': bundle.name, 'split_id': cfg['split_id'], 'seed': cfg['seed'],
        'config_sha256': json_hash(cfg), 'source_sha256': source_hash(),
        'split_sha256': json_hash(split), 'features_sha256': tensor_hash(bundle.data.x),
        'labels_sha256': tensor_hash(bundle.data.y),
        'message_edges_sha256': tensor_hash(bundle.data.edge_index),
        'evidence_edges_sha256': tensor_hash(bundle.data.evidence_edge_index),
        'message_edges': int(bundle.data.edge_index.shape[1]),
        'evidence_pairs': int(bundle.data.evidence_edge_index.shape[1]),
        'graph_protocol': bundle.data.graph_protocol,
        'evidence_protocol': bundle.data.evidence_protocol,
    }


def claim_output(output_dir, identity):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    with (path / 'identity.json').open('x', encoding='utf-8') as handle:
        json.dump(identity, handle, indent=2, sort_keys=True)
    return path
