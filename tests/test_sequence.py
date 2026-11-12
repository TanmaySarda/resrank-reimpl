import pytest
import torch

from resrank.model.sequence import (SequenceLayout, build_inputs_embeds,
                                    collate_sequences, gather_hidden)

D = 6


def make(prefix=4, n=3, anchor=2):
    p = torch.randn(prefix, D)
    e = torch.randn(n, D)
    a = torch.randn(anchor, D)
    eos = torch.randn(D)
    return p, e, a, eos


def test_layout_positions_match_plan_example():
    # 12 prefix tokens, 3 passages, 3 anchor tokens, 1 EOS -> positions from the conversation
    x, layout = build_inputs_embeds(*make(prefix=12, n=3, anchor=3))
    assert x.shape == (19, D)
    assert (layout.passage_start, layout.passage_end) == (12, 15)
    assert layout.eos_index == 18 and layout.seq_len == 19


def test_sequence_content_order():
    p, e, a, eos = make()
    x, layout = build_inputs_embeds(p, e, a, eos)
    assert torch.equal(x[:4], p)
    assert torch.equal(x[layout.passage_slice], e)
    assert torch.equal(x[layout.anchor_start:layout.eos_index], a)
    assert torch.equal(x[layout.eos_index], eos)


def test_hundred_passages_match_input_size_in_plan():
    x, layout = build_inputs_embeds(*make(prefix=12, n=100, anchor=3))
    assert x.shape[0] == 116 and layout.passage_slice == slice(12, 112)


def test_build_rejects_mismatched_hidden_size():
    p, e, a, eos = make()
    with pytest.raises(ValueError):
        build_inputs_embeds(p, torch.randn(3, D + 1), a, eos)


def test_collate_pads_right_and_masks():
    s1 = build_inputs_embeds(*make(prefix=4, n=3, anchor=2))
    s2 = build_inputs_embeds(*make(prefix=5, n=4, anchor=2))
    batch, mask, layouts = collate_sequences([s1, s2])
    assert batch.shape == (2, s2[0].shape[0], D)
    assert mask[0].sum().item() == s1[0].shape[0]
    assert mask[1].sum().item() == s2[0].shape[0]
    assert torch.equal(batch[0, : s1[0].shape[0]], s1[0])
    assert batch[0, s1[0].shape[0]:].abs().sum() == 0
    assert layouts[0] == s1[1]


def test_gather_hidden_reads_the_right_positions():
    s1 = build_inputs_embeds(*make(prefix=4, n=3, anchor=2))
    s2 = build_inputs_embeds(*make(prefix=5, n=4, anchor=2))
    batch, _, layouts = collate_sequences([s1, s2])
    # Use the inputs themselves as stand-in "hidden states" so we can check positions exactly.
    h_p, pmask, h_eos = gather_hidden(batch, layouts)
    assert h_p.shape == (2, 4, D) and pmask.tolist() == [[True] * 3 + [False], [True] * 4]
    assert torch.equal(h_p[0, :3], s1[0][s1[1].passage_slice])
    assert h_p[0, 3].abs().sum() == 0
    assert torch.equal(h_eos[0], s1[0][-1]) and torch.equal(h_eos[1], s2[0][-1])


def test_layout_to_dict_roundtrip_fields():
    layout = SequenceLayout(prefix_len=12, n_passages=100, anchor_len=3)
    d = layout.to_dict()
    assert d["passage_start"] == 12 and d["eos_index"] == 115 and d["seq_len"] == 116


def test_scores_end_to_end_with_fusion_and_masking():
    """A tiny pipeline: sequence -> fake hidden -> gather -> fuse -> cosine -> RankNet."""
    from resrank.model.fusion import fuse
    from resrank.model.losses import ranknet_loss
    from resrank.model.scoring import cosine_scores

    p, e, a, eos = make(n=4)
    x, layout = build_inputs_embeds(p, e, a, eos)
    batch, _, layouts = collate_sequences([(x, layout)])
    hidden = batch.clone().requires_grad_(True)       # stand-in for reranker output
    h_p, pmask, h_eos = gather_hidden(hidden, layouts)
    r = fuse(h_p, e.unsqueeze(0))
    scores = cosine_scores(h_eos, r)
    assert scores.shape == (1, 4)
    ranknet_loss(scores, torch.tensor([[0, 1, 2, 3]]), mask=pmask).backward()
    assert hidden.grad is not None and hidden.grad.abs().sum() > 0
