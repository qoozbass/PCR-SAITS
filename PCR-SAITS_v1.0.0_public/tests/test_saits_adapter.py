import numpy as np
import pcrsaits.backbones.saits as mod
from pcrsaits.backbones import BackboneAdapter, SAITSBackbone

class FakeSAITS:
    instances = []
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fit_calls = []
        self.impute_calls = []
        self.saved = []
        self.loaded = []
        FakeSAITS.instances.append(self)
    def fit(self, *args):
        self.fit_calls.append(args)
    def impute(self, data):
        self.impute_calls.append(data)
        return np.asarray(data["X"]) + 1.0
    def save(self, path):
        self.saved.append(path)
    def load(self, path):
        self.loaded.append(path)

def make(monkeypatch):
    FakeSAITS.instances.clear()
    monkeypatch.setattr(mod, "SAITS", FakeSAITS)
    monkeypatch.setattr(mod, "choose_device", lambda: "cpu")
    return SAITSBackbone(
        n_steps=48, n_features=11, epochs=10, batch_size=32, patience=3,
        d_model=64, d_ffn=128, n_heads=4, n_layers=2, dropout=0.1,
        verbose=False,
    )

def test_constructor_matches_legacy_v71(monkeypatch):
    obj = make(monkeypatch)
    assert isinstance(obj, BackboneAdapter)
    assert obj.model.kwargs == {
        "n_steps": 48, "n_features": 11, "n_layers": 2,
        "d_model": 64, "d_ffn": 128, "n_heads": 4,
        "d_k": 16, "d_v": 16, "dropout": 0.1,
        "batch_size": 32, "epochs": 10, "patience": 3,
        "num_workers": 0, "device": "cpu",
    }

def test_fit_float32_and_val_reuse(monkeypatch):
    obj = make(monkeypatch)
    train = np.ones((2,48,11), dtype=np.float64)
    val = np.full((1,48,11), 2.0, dtype=np.float64)
    obj.fit(train, val)
    train_set, val_set = obj.model.fit_calls[-1]
    assert train_set["X"].dtype == np.float32
    assert val_set["X"].dtype == np.float32
    assert val_set["X_ori"].dtype == np.float32
    np.testing.assert_array_equal(val_set["X"], val_set["X_ori"])

def test_impute_and_checkpoint_passthrough(monkeypatch, tmp_path):
    obj = make(monkeypatch)
    x = np.zeros((1,48,11), dtype=np.float64)
    out = obj.impute(x)
    assert obj.model.impute_calls[-1]["X"].dtype == np.float32
    assert out.dtype == np.dtype(float)
    p = tmp_path/"saits.pypots"
    obj.save(p)
    assert obj.model.saved == [str(p)]
    kwargs = dict(n_steps=48,n_features=11,epochs=10,batch_size=32,patience=3,
                  d_model=64,d_ffn=128,n_heads=4,n_layers=2,dropout=0.1,
                  verbose=False)
    loaded = SAITSBackbone.load_from_checkpoint(p, **kwargs)
    assert loaded.model.loaded == [str(p)]
