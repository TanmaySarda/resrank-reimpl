import pytest
import torch

from conftest import FakeTokenizer, make_model, make_sample, tiny_backbone
from resrank.model.diagnostics import dump_intermediates, grad_flow_report, norm_report, overfit
from resrank.model.encoder import PassageEncoder
from resrank.model.pooling import last_token_pool
from resrank.model.reranker import RerankerLM
from resrank.model.resrank import ResRankConfig, ResRankModel
from resrank.model.sequence import pad_stack


# ---- pooling ---------------------------------------------------------------
def test_last_token_pool_left_and_right_padding():
    h = torch.arange(2 * 4 * 3, dtype=torch.float).view(2, 4, 3)
    left = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]])
    assert torch.equal(last_token_pool(h, left), h[:, -1])
    right = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    pooled = last_token_pool(h, right)
    assert torch.equal(pooled[0], h[0, 1]) and torch.equal(pooled[1], h[1, 3])


def test_pad_stack():
    out = pad_stack([torch.ones(2, 3), torch.ones(4, 3)])
    assert out.shape == (2, 4, 3) and out[0, 2:].abs().sum() == 0


# ---- encoder ---------------------------------------------------------------
def test_encoder_shapes_and_query_vs_passage_formatting():
    enc = PassageEncoder(tiny_backbone(1), FakeTokenizer(), max_length=64, query_max_length=64)
    docs = enc.encode(["alpha beta", "gamma"])
    assert docs.shape == (2, 32)
    q_plain = enc.encode(["alpha beta"], is_query=False)
    q_inst = enc.encode(["alpha beta"], is_query=True)
    assert not torch.allclose(q_plain, q_inst)   # the Instruct/Query wrapper changes the vector


def test_encoder_batch_independent_of_padding_and_chunking():
    enc = PassageEncoder(tiny_backbone(1), FakeTokenizer(), max_length=64).eval()
    texts = ["a short one", "a considerably longer passage text here", "mid length text"]
    with torch.no_grad():
        full = enc.encode(texts)
        chunked = enc.encode(texts, chunk_size=1)
        alone = enc.encode([texts[0]])
    assert torch.allclose(full, chunked, atol=1e-5)
    assert torch.allclose(full[0], alone[0], atol=1e-5)   # left padding must not change the result


# ---- model wiring ----------------------------------------------------------
def test_hidden_size_mismatch_is_rejected():
    from transformers import Qwen3Config, Qwen3Model
    big = Qwen3Model(Qwen3Config(hidden_size=48, intermediate_size=64, num_hidden_layers=1,
                                 num_attention_heads=4, num_key_value_heads=2, head_dim=12,
                                 vocab_size=260))
    with pytest.raises(ValueError, match="hidden size"):
        ResRankModel(PassageEncoder(big, FakeTokenizer()), RerankerLM(tiny_backbone(2), FakeTokenizer()))


def test_config_validation_and_from_dict_ignores_unknown():
    with pytest.raises(ValueError):
        ResRankConfig(fusion_mode="nope")
    with pytest.raises(ValueError):
        ResRankConfig(inject_mode="nope")
    assert ResRankConfig.from_dict({"lam": 0.5, "unknown": 1}).lam == 0.5


def test_rerank_shapes_and_sequence_layout(model, samples):
    model.eval()
    with torch.no_grad():
        e_list = model.encode_passages([s.passages for s in samples])
        out = model.rerank_from_embeddings([s.query for s in samples], e_list, return_intermediates=True)
    assert out["scores"].shape == (3, 5) and out["passage_mask"].all()
    layout = out["layouts"][0]
    # The passage positions of the input hold exactly the injected passage vectors.
    injected = model.inject(e_list[0])
    assert torch.equal(out["inputs_embeds"][0, layout.passage_slice], injected)
    # The last real position is the EOS token embedding.
    eos = model.reranker.token_embeddings([model.reranker.eos_id])[0]
    assert torch.equal(out["inputs_embeds"][0, layout.eos_index], eos)
    # Fusion is the residual sum.
    assert torch.allclose(out["r"], out["h_p"] + out["e"])


def test_passage_hidden_states_are_causal(model):
    """h_i may depend on earlier passages but never on later ones (causal attention)."""
    model.eval()
    with torch.no_grad():
        e = model.encode_passages([make_sample(4).passages])[0]
        q = ["what is topic 0"]
        base = model.rerank_from_embeddings(q, [e], return_intermediates=True)["h_p"][0]
        e_changed_last = e.clone(); e_changed_last[3] += 1.0
        after_last = model.rerank_from_embeddings(q, [e_changed_last], return_intermediates=True)["h_p"][0]
        e_changed_first = e.clone(); e_changed_first[0] += 1.0
        after_first = model.rerank_from_embeddings(q, [e_changed_first], return_intermediates=True)["h_p"][0]
    assert torch.allclose(base[:3], after_last[:3], atol=1e-5)        # earlier unaffected by a later change
    assert not torch.allclose(base[3], after_last[3], atol=1e-5)
    assert not torch.allclose(base[1:], after_first[1:], atol=1e-5)    # later ones see the earlier change


def test_batching_with_different_list_lengths_matches_single(model):
    a, b = make_sample(4, seed=1), make_sample(6, seed=2)
    model.eval()
    with torch.no_grad():
        batched = model.rerank([a.query, b.query], [a.passages, b.passages])
        alone = model.rerank([a.query], [a.passages])
    assert batched["passage_mask"].sum(dim=1).tolist() == [4, 6]
    assert torch.allclose(batched["scores"][0, :4], alone["scores"][0], atol=1e-4)


def test_inject_modes(model):
    e = torch.randn(3, 32) * 5
    model.cfg.inject_mode, model.cfg.inject_scale = "unit", 2.0
    assert torch.allclose(model.inject(e).norm(dim=-1), torch.full((3,), 2.0), atol=1e-5)
    model.cfg.inject_mode, model.cfg.inject_scale = "raw", 1.0
    assert torch.allclose(model.inject(e), e)


@pytest.mark.parametrize("mode", ["residual", "no_residual", "no_hidden"])
def test_fusion_modes_run(mode, samples):
    m = make_model(fusion_mode=mode)
    losses = m.training_losses(samples)
    assert torch.isfinite(losses["total"])


# ---- training signals ------------------------------------------------------
def test_losses_finite_and_gradients_reach_both_models(model, samples):
    report = grad_flow_report(model, samples)
    assert all(v == v and abs(v) != float("inf") for v in report["losses"].values())
    assert report["encoder"]["grad_norm"] > 0 and report["reranker"]["grad_norm"] > 0
    assert report["encoder"]["params_with_grad"] > 0
    assert report["encoder_weights_changed"] is True


def test_encoder_gets_gradient_through_reranker_input_alone():
    """With the residual off, the encoder must still learn through the injected tokens."""
    m = make_model(fusion_mode="no_residual", lam=0.0)   # lam=0 removes the InfoNCE route too
    s = [make_sample(5, seed=3)]
    report = grad_flow_report(m, s, take_step=False)
    assert report["encoder"]["grad_norm"] > 0


def test_overfit_small_batch(samples):
    m = make_model()
    history = overfit(m, samples[:2], steps=60, lr=3e-3)
    assert history[-1] < 0.5 * history[0]


def test_norm_report_and_dump(tmp_path, model, samples):
    meta = dump_intermediates(model, samples[:2], tmp_path / "dbg")
    for name in ("e", "inputs_embeds", "attention_mask", "H", "h_p", "h_eos", "r", "scores", "meta"):
        assert any((tmp_path / "dbg").glob(f"{name}.*")), name
    assert meta["shapes"]["scores"] == [2, 5]
    assert meta["norms"]["e_norm_mean"] > 0 and meta["norms"]["h_over_e"] > 0
    assert meta["layouts"][0]["eos_index"] == meta["layouts"][0]["passage_end"] + meta["layouts"][0]["anchor_len"]


def test_save_pretrained_writes_both_parts(tmp_path, model):
    model.save_pretrained(tmp_path / "ck")
    assert (tmp_path / "ck/encoder/config.json").exists()
    assert (tmp_path / "ck/reranker/config.json").exists()
    assert (tmp_path / "ck/resrank_config.json").exists()
