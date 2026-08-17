from pathlib import Path
import json
import pytest
torch = pytest.importorskip('torch')
from leakagebench_midi.models import build_tcn, build_transformer, load_checkpoint

def transformer_config():
    return {'vocab_size': 17, 'context_length': 16, 'layers': 1, 'd_model': 16, 'heads': 2, 'ffn_dim': 32, 'dropout': 0.0, 'gradient_checkpointing': False}

def tcn_config():
    return {'vocab_size': 17, 'context_length': 16, 'channels': 256, 'ffn_dim': 1024, 'blocks': 9, 'kernel_size': 5, 'dilations': [1, 2, 4, 8, 16, 32, 64, 128, 256], 'dropout': 0.0}

def artifact(model, architecture, config):
    return {'format_version': '1.0', 'artifact_type': 'LeakageBench-MIDI model checkpoint', 'model_id': 'test-model', 'dataset': 'lmd', 'architecture': architecture, 'model_size': 'test', 'condition': 'clean', 'seed': 7, 'paper_role': 'test', 'parameter_count': sum((p.numel() for p in model.parameters())), 'state_dict': model.state_dict(), 'model_config': config, 'tokenizer_config': {'vocab_size': 17}, 'source_checkpoint_sha256': '0' * 64, 'software_version': 'v1.0.0', 'model_artifact_version': 'v1.0.0', 'weight_license': 'CC-BY-4.0'}

@pytest.mark.parametrize('architecture,config,builder', [('transformer', transformer_config(), build_transformer), ('tcn', tcn_config(), build_tcn)])
def test_checkpoint_load(tmp_path, architecture, config, builder):
    path = tmp_path / f'{architecture}.pt'
    torch.save(artifact(builder(config), architecture, config), path)
    (loaded, metadata) = load_checkpoint(path)
    assert not loaded.training and metadata['software_version'] == 'v1.0.0'

def test_public_loader_unknown_architecture(tmp_path):
    cfg = transformer_config()
    value = artifact(build_transformer(cfg), 'unknown', cfg)
    path = tmp_path / 'x.pt'
    torch.save(value, path)
    with pytest.raises(ValueError, match='unknown architecture'):
        load_checkpoint(path)

def test_public_loader_missing_state_dict(tmp_path):
    cfg = transformer_config()
    value = artifact(build_transformer(cfg), 'transformer', cfg)
    value.pop('state_dict')
    path = tmp_path / 'x.pt'
    torch.save(value, path)
    with pytest.raises(ValueError, match='missing required fields: state_dict'):
        load_checkpoint(path)

@pytest.mark.parametrize('mutation', ['bad_shape', 'strict_mismatch'])
def test_public_loader_rejects_state_mismatch(tmp_path, mutation):
    cfg = transformer_config()
    value = artifact(build_transformer(cfg), 'transformer', cfg)
    if mutation == 'bad_shape':
        value['state_dict']['embed.weight'] = value['state_dict']['embed.weight'][:-1]
    else:
        value['state_dict']['unexpected.weight'] = torch.zeros(1)
    path = tmp_path / 'x.pt'
    torch.save(value, path)
    with pytest.raises(RuntimeError):
        load_checkpoint(path)

def test_model_metadata_version(tmp_path):
    cfg = transformer_config()
    value = artifact(build_transformer(cfg), 'transformer', cfg)
    value['software_version'] = 'v1.0.0-rc2'
    path = tmp_path / 'x.pt'
    torch.save(value, path)
    with pytest.raises(ValueError, match='unsupported model metadata version'):
        load_checkpoint(path)

def test_model_manifest_loader_reference():
    root = Path(__file__).resolve().parents[2]
    manifest = root.parent.parent / 'models/LeakageBench-MIDI-Model-Checkpoints-v1.0.0-candidate/MODEL_MANIFEST.json'
    if manifest.exists():
        rows = json.loads(manifest.read_text())['models']
        assert len(rows) == 30 and all((x['public_loader'] == 'leakagebench_midi.models.load_checkpoint' for x in rows))
