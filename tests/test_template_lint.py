from __future__ import annotations

import pytest

from autocapture.security.template_lint import lint_template_text


def test_lint_rejects_attr_filter() -> None:
    with pytest.raises(ValueError):
        lint_template_text("{{ user|attr('secret') }}")


def test_lint_rejects_dunder() -> None:
    with pytest.raises(ValueError):
        lint_template_text("{{ __class__ }}")


def test_lint_allows_simple_text() -> None:
    lint_template_text("Plain prompt with {{ variable }}")
