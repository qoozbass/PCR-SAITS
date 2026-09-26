from pcrsaits import BackboneAdapter, BRITSBackbone, SAITSBackbone

def test_public_imports():
    assert BackboneAdapter is not None
    assert SAITSBackbone is not None
    assert BRITSBackbone is not None
