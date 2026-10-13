"""build_example: text layout and which tokens count as target."""
from prefixlab.data.formats import Cond, build_example


def _target_tokens(tok, ex):
    return [tok.decode([t]) for t, m in zip(ex.input_ids, ex.target_mask) if m]


def test_plain_layout_is_input_newline_target(tok):
    ex = build_example(tok, "The cat sat.", "Кошка сидела.", Cond())
    assert ex.text == "The cat sat.\nКошка сидела."


def test_target_tokens_cover_exactly_the_target_text(tok):
    ex = build_example(tok, "The cat sat.", "Кошка сидела.", Cond())
    decoded = "".join(_target_tokens(tok, ex))
    assert decoded.strip() == "Кошка сидела."
    # nothing from the input is scored
    assert "cat" not in decoded


def test_language_label_sits_before_target_and_is_not_scored(tok):
    ex = build_example(tok, "The cat sat.", "Кошка сидела.", Cond(lang_label="Russian:"))
    assert ex.text == "The cat sat.\nRussian: Кошка сидела."
    decoded = "".join(_target_tokens(tok, ex))
    assert "Russian" not in decoded
    assert decoded.strip() == "Кошка сидела."


def test_task_instruction_sits_before_input_and_is_not_scored(tok):
    ex = build_example(tok, "The cat sat.", "Кошка сидела.", Cond(task_instruction="Translate:"))
    assert ex.text == "Translate: The cat sat.\nКошка сидела."
    decoded = "".join(_target_tokens(tok, ex))
    assert "Translate" not in decoded


def test_no_eos_is_appended(tok):
    ex = build_example(tok, "a", "b", Cond())
    assert ex.input_ids[-1] != tok.eos_token_id
    assert len(ex.input_ids) == len(ex.target_mask)


def test_first_target_token_may_carry_leading_space_but_not_input_text(tok):
    ex = build_example(tok, "Hello there", "Привет", Cond(lang_label="Russian:"))
    first = _target_tokens(tok, ex)[0]
    assert first.lstrip(" ").startswith("П")
    assert ex.n_target_tokens == sum(ex.target_mask)


def test_same_line_separator_puts_label_at_start(tok):
    ex = build_example(tok, "Dante was born in", "Флоренция", Cond(lang_label="Russian:"), sep=" ")
    assert ex.text == "Russian: Dante was born in Флоренция"
    decoded = "".join(_target_tokens(tok, ex))
    assert decoded.strip() == "Флоренция"
    assert ex.boundary_clean


def test_same_line_with_instruction_and_label(tok):
    ex = build_example(tok, "Dante was born in", "Флоренция", Cond(lang_label="Russian:", task_instruction="Complete:"), sep=" ")
    assert ex.text == "Complete: Russian: Dante was born in Флоренция"


def test_copy_mask_marks_target_tokens_that_occur_in_the_input(tok):
    ex = build_example(tok, "Paris is big.", "Париж большой. Paris", Cond())
    toks = [tok.decode([t]) for t in ex.input_ids]
    copy_targets = [toks[i].strip() for i, (m, c) in enumerate(zip(ex.target_mask, ex.copy_mask)) if m and c]
    assert "Paris" in "".join(copy_targets)
    assert all(not c for c, m in zip(ex.copy_mask, ex.target_mask) if not m)  # never set off-target
    assert not any(("Париж" in toks[i]) and ex.copy_mask[i] for i in range(len(toks)))
