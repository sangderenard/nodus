"""Source-only tests never run the real compiler or a service."""
import dataclasses
import hashlib
import json
from pathlib import Path
import pytest
from tools import turing_row_tool_compiler as adapter


@pytest.fixture
def source_request(tmp_path):
    contract = tmp_path / 'contract.yaml'
    contract.write_text('caller-selected-contract')
    output = tmp_path / 'product'
    return dict(unstable_source_only=True, source_names=['Owner.method', 'byte_fifo'],
                source_text='# untouched\r\nclass Owner:\r\n    def method(self, x: object): return x\r\n\r\ndef byte_fifo(x: bytes): return x\r\n',
                backend='c', output_dir=str(output), extraction_contract=str(contract),
                turing_root=str(tmp_path / 'turing'),
                site_artifact_context=dict(owner_id=3, workspace_id='a'*32, job_id='42', job_directory=str(output)))


class StubDivider:
    def __init__(self, refused=False):
        self.refused = refused
        self.spec = None
    def attempt(self, spec_path):
        spec = self.spec = json.loads(spec_path.read_text())
        self.source = Path(spec['source_file']).read_bytes()
        self.contract = Path(spec['extraction_contract']).read_text()
        out = Path(spec['out_dir'])
        entries = []
        for i, name in enumerate(spec['names']):
            if self.refused:
                entries.append(dict(name=name, state='refused', stage='emission', error_type='EmissionShortfall', error='actual compiler diagnostic: object not supported'))
            else:
                product = out / f'{i}.c'
                product.write_bytes(b'/* emitted only */')
                entries.append(dict(name=name, state='complete', stage='emission', product=product.name,
                                    sha256=hashlib.sha256(product.read_bytes()).hexdigest(), note='emitted-not-built'))
        (out / 'attempt.json').write_text(json.dumps(dict(backend=spec['backend'], entries=entries)))
        return 0


@pytest.mark.parametrize('backend', ['c', 'llvm', 'fortran'])
def test_source_only_retains_source_without_abi(source_request, backend):
    source_request['backend'] = backend
    stub = StubDivider()
    result = adapter.compile_row_tool(source_request, compiler=stub)
    assert stub.source == source_request['source_text'].encode()
    assert stub.spec['names'] == source_request['source_names']
    assert stub.contract == 'caller-selected-contract'
    assert result['schema'] == adapter.SOURCE_PRODUCT_SCHEMA
    assert result['state'] == 'emitted-not-built'
    assert result['loadable'] is False
    assert result['adoption_receipt'] is None and result['binary'] is None and result['buffer_abi'] is None
    output = Path(source_request['output_dir'])
    assert not (output / adapter.ADOPTION_RECEIPT_NAME).exists()
    assert not (output / 'submitted-source.py').exists()
    assert json.loads((output / 'site-artifact-lifecycle.json').read_text())['state'] == 'emitted-not-built'
    assert any(r['path'].endswith('attempt.json') for r in result['artifacts'])
    for record in result['artifacts']:
        assert record['sha256'] == hashlib.sha256((output / record['path']).read_bytes()).hexdigest()


def test_path_source_preserves_crlf(source_request, tmp_path):
    source = tmp_path / 'source.py'
    source.write_bytes(source_request.pop('source_text').encode())
    source_request['source_path'] = str(source)
    stub = StubDivider()
    adapter.compile_row_tool(source_request, compiler=stub)
    assert stub.source == source.read_bytes()


def test_refusal_is_actual_divider_diagnostic(source_request):
    result = adapter.compile_row_tool(source_request, compiler=StubDivider(refused=True))
    assert result['state'] == 'refused'
    assert result['attempt']['entries'][0]['error'] == 'actual compiler diagnostic: object not supported'
    output = Path(source_request['output_dir'])
    assert json.loads((output / 'site-artifact-lifecycle.json').read_text())['state'] == 'refused'
    assert not (output / adapter.ADOPTION_RECEIPT_NAME).exists()


@pytest.mark.parametrize('patch', [
    {'row_entry': 'fake'}, {'columns': []}, {'global_scope': False}, {'ssa_composition': {}},
    {'allow_uncompiled': True}, {'unstable_source_only': 'true'}, {'unstable_source_only': 1},
    {'unstable_source_only': False}, {'source_names': []}, {'source_names': ['a', 'a']},
    {'source_names': ['A.b', 'A_b']}, {'source_names': ['../escape']}, {'source_names': 'entry'},
    {'source_names': [7]}, {'backend': 'wasm'}, {'source_text': 12},
    {'extraction_contract': None}, {'turing_root': None}, {'site_artifact_context': None},
    {'site_artifact_context': {}}, {'output_dir': None}, {'source_path': 'both.py'},
])
def test_reject_before_artifact_writes(source_request, patch):
    output = Path(source_request['output_dir'])
    source_request.update(patch)
    with pytest.raises((ValueError, TypeError, adapter.ProductError)):
        adapter.compile_row_tool(source_request, compiler=StubDivider())
    assert not output.exists()


def test_dataclass_cannot_bypass_context_or_mixed_mode(source_request):
    parsed = adapter.RowToolRequest.from_mapping(source_request)
    for forged in [dataclasses.replace(parsed, site_artifact_context=None), dataclasses.replace(parsed, row_entry='fake')]:
        with pytest.raises(ValueError):
            adapter.compile_row_tool(forged, compiler=StubDivider())
    assert not Path(source_request['output_dir']).exists()


def test_no_stale_product_deletion_or_reuse(source_request):
    output = Path(source_request['output_dir']); output.mkdir()
    receipt = output / adapter.ADOPTION_RECEIPT_NAME; receipt.write_text('previous')
    with pytest.raises(adapter.ProductError, match='must be empty'):
        adapter.compile_row_tool(source_request, compiler=StubDivider())
    assert receipt.read_text() == 'previous'
    assert len(list(output.iterdir())) == 1


def test_missing_contract_does_not_write(source_request):
    Path(source_request['extraction_contract']).unlink()
    with pytest.raises(FileNotFoundError):
        adapter.compile_row_tool(source_request, compiler=StubDivider())
    assert not Path(source_request['output_dir']).exists()


def test_cli_refused_returns_failure(source_request, tmp_path, monkeypatch):
    path = tmp_path / 'source_request.json'; path.write_text(json.dumps(source_request))
    monkeypatch.setattr(adapter, 'SourceDividerBoundary', lambda root: StubDivider(refused=True))
    assert adapter.main(['--request', str(path)]) == 1


def test_cli_emission_success_without_adoption(source_request, tmp_path, monkeypatch):
    path = tmp_path / 'source_request.json'; path.write_text(json.dumps(source_request))
    monkeypatch.setattr(adapter, 'SourceDividerBoundary', lambda root: StubDivider())
    assert adapter.main(['--request', str(path)]) == 0




@pytest.mark.parametrize('name', [
    'main.<locals>._doc_id', 'Owner.method.<locals>.inner',
    'outer.<locals>.inner.<locals>.deep', 'Outer.Inner.method',
    'outer.<locals>.LocalClass.method', 'Owner.method.<locals>.inner.<locals>.deep',
])
def test_authored_nested_names_preserved(source_request, name):
    source_request['source_names'] = [name]
    stub = StubDivider()
    result = adapter.compile_row_tool(source_request, compiler=stub)
    assert stub.spec['names'] == [name]
    assert result['source_names'] == [name]
    assert result['attempt']['entries'][0]['name'] == name
    assert stub.source == source_request['source_text'].encode()


@pytest.mark.parametrize('name', [
    '', '.entry', 'entry.', 'Owner..method', '<locals>', '<locals>.entry',
    'outer.<locals>', 'outer.<locals>.<locals>.inner', 'outer.<locals>..inner',
    'outer.<local>.inner', 'outer.<lambda>.inner', 'outer.<locals>/inner',
    'outer/<locals>/inner', 'outer.\\inner', '../outer', 'C:/outer',
    'outer.<locals>.../inner', 'outer.<locals>.inner\x00',
])
def test_malformed_authored_names_rejected_before_writes(source_request, name):
    source_request['source_names'] = [name]
    with pytest.raises(ValueError, match='source_names'):
        adapter.compile_row_tool(source_request, compiler=StubDivider())
    assert not Path(source_request['output_dir']).exists()


def test_nested_name_stem_collision_rejected(source_request):
    source_request['source_names'] = ['main.<locals>._doc_id', 'main_locals__doc_id']
    with pytest.raises(ValueError, match='collide'):
        adapter.compile_row_tool(source_request, compiler=StubDivider())
    assert not Path(source_request['output_dir']).exists()
