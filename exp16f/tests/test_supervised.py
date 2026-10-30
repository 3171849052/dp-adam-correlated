import pytest
from exp16f.runtime import supervised
from exp16f.config import save_json,file_hash

def test_official_manifest_verification(tmp_path,monkeypatch):
    p=tmp_path/'tokens.pt';p.write_bytes(b'fixture')
    monkeypatch.setattr(supervised,'RESULTS',tmp_path)
    monkeypatch.setattr(supervised,'ROOT',tmp_path)
    save_json(tmp_path/'official_local_assets.json',dict(historical_tensor_equal=True,files={'tokens.pt':file_hash(p)}))
    supervised.verify_official(True)
    p.write_bytes(b'corrupted')
    with pytest.raises(AssertionError):supervised.verify_official(True)

def test_official_only():
    with pytest.raises(AssertionError):supervised.verify_official(False)
