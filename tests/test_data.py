from __future__ import annotations

import numpy as np
import pytest

from scripts.dataloader import Sub


def test_invalid_subject_raises_value_error():
    with pytest.raises(ValueError, match="отсутствуют снимки фМРТ"):
        Sub("99")


def test_subs_with_fmri_not_empty():
    assert len(Sub.subs_with_fmri) > 0
    assert all(isinstance(s, str) for s in Sub.subs_with_fmri)


def test_known_subjects_in_list():
    for sid in ["04", "09", "13"]:
        assert sid in Sub.subs_with_fmri
