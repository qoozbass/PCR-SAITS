import numpy as np
import pcrsaits.backbones.brits as mod
from pcrsaits.backbones import BackboneAdapter, BRITSBackbone

class FakeBRITS:
    instances = []
    fail_validation_once = False
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fit_calls = []
        self.impute_calls = []
        self.saved = []
        self.loaded = []
        self.failed = False
        FakeBRITS.instances.append(self)
    def fit(self, *args):
        self.fit_calls.append(args)
        if FakeBRITS.fail_validation_once and len(args)==2 and not self.failed:
            self.failed = True
            raise RuntimeError("synthetic validation incompatibility")
    def impute(self, data):
        self.impute_calls.append(data)
        return np.asarray(data["X"]) + 2.0
    def save(self, path):
        self.saved.append(path)
    def load(self, path):
        self.loaded.append(path)

def make(monkeypatch):
    FakeBRITS.instances.clear()
    FakeBRITS.fail_validation_once = False
    monkeypatch.setattr(mod, "BRITS", FakeBRITS)
    monkeypatch.setattr(mod, "choose_device", lambda: "cpu")
    return BRITSBackbone(
        n_steps=48,n_features=11,epochs=10,batch_size=32,patience=3,
        rnn_hidden_size=64,verbose=False,
    )

def test_constructor_matches_legacy(monkeypatch):
    obj = make(monkeypatch)
    assert isinstance(obj, BackboneAdapter)
    assert obj.model.kwargs == {
        "n_steps":48,"n_features":11,"rnn_hidden_size":64,
        "batch_size":32,"epochs":10,"patience":3,"num_workers":0,
        "device":"cpu","verbose":False,"saving_path":None,
    }

def test_validation_fit_path(monkeypatch):
    obj = make(monkeypatch)
    train=np.ones((2,48,11),dtype=np.float64)
    val=np.ones((1,48,11),dtype=np.float64)
    ori=np.ones((1,48,11),dtype=np.float64)
    obj.fit(train,val,ori)
    train_set,val_set=obj.model.fit_calls[0]
    assert train_set["X"].dtype==np.float32
    assert val_set["X"].dtype==np.float32
    assert val_set["X_ori"].dtype==np.float32

def test_validation_failure_falls_back(monkeypatch):
    obj = make(monkeypatch)
    FakeBRITS.fail_validation_once = True
    x=np.ones((1,48,11),dtype=np.float64)
    obj.fit(x,x,x)
    assert len(obj.model.fit_calls)==2
    assert len(obj.model.fit_calls[0])==2
    assert len(obj.model.fit_calls[1])==1

def test_incomplete_validation_pair_uses_train_only(monkeypatch):
    obj = make(monkeypatch)
    x=np.ones((1,48,11),dtype=np.float64)
    obj.fit(x,x,None)
    assert len(obj.model.fit_calls)==1
    assert len(obj.model.fit_calls[0])==1

def test_impute_and_checkpoint_passthrough(monkeypatch,tmp_path):
    obj=make(monkeypatch)
    x=np.zeros((1,48,11),dtype=np.float64)
    out=obj.impute(x)
    assert obj.model.impute_calls[-1]["X"].dtype==np.float32
    assert out.dtype==np.dtype(float)
    p=tmp_path/"brits.pypots"
    obj.save(p)
    assert obj.model.saved==[str(p)]
    kwargs=dict(n_steps=48,n_features=11,epochs=10,batch_size=32,patience=3,
                rnn_hidden_size=64,verbose=False)
    loaded=BRITSBackbone.load_from_checkpoint(p,**kwargs)
    assert loaded.model.loaded==[str(p)]
