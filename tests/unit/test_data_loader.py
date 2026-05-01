import util.data_loader as dl

def test_vocab_look_up():
    token = ["Hi", "Hi", "ok", "nice", "nice"]
    vc = dl.Vocab(token, min_freq=2)
    unk_id = vc.lookup("<unk>")
    assert vc.lookup("Hi") != unk_id
    assert vc.lookup("ok") == unk_id
    assert vc.lookup("nice") != unk_id
    assert vc.lookup("none") == unk_id
