from tools.gold_registry.run_adaptive_render_strategy import _build_page_attempts as _adaptive_page_attempts
from tools.gold_registry.run_render_dpi_experiment import _build_page_attempts as _render_page_attempts


def test_adaptive_page_attempts_do_not_fallback_to_zero() -> None:
    assert _adaptive_page_attempts(68) == [68]
    assert _adaptive_page_attempts(0) == [0]


def test_render_page_attempts_do_not_fallback_to_zero() -> None:
    assert _render_page_attempts(192) == [192]
    assert _render_page_attempts(0) == [0]
