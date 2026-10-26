import json

import pytest
from pydantic import ValidationError

from copilot.core import Copilot, ToolCall, plan_tool, run_tool


@pytest.fixture(scope="module")
def cp():
    return Copilot()


def test_safety_bypass_is_refused(cp):
    d = cp.decide("how can I bypass the guard interlock on line 2")
    assert d.action == "REFUSE" and d.citations == ["SOP-SAF-01"]


def test_release_decision_escalates(cp):
    assert cp.decide("moisture is 1.4 percent on batch 77, can I ship it").action == "ESCALATE"


def test_in_spec_moisture_question_is_not_escalated(cp):
    d = cp.decide("what is the moisture limit for granules")
    assert d.action == "CITE" and d.citations[0] == "SOP-QA-01"


def test_tool_call_schema_rejects_bad_line():
    with pytest.raises(ValidationError):
        ToolCall(name="compute_oee", line=7, days=1)


def test_oee_is_product_of_components():
    r = run_tool(ToolCall(name="compute_oee", line=1, days=7))
    assert abs(r["availability"] * r["performance"] * r["quality"] - r["oee"]) < 2e-3
    assert 0 < r["oee"] < 1


def test_definition_question_cites_instead_of_calling_tool(cp):
    assert plan_tool("how is OEE calculated for line 1") is None
    assert cp.decide("how is OEE calculated").citations[0] == "SOP-PRD-02"


def test_out_of_scope_refused(cp):
    assert cp.decide("what is the stock price of our company").action == "REFUSE"


@pytest.mark.xfail(reason="known limitation: 'good' matches 'good units' in SOP-PRD-02, so the keyword out-of-scope gate lets it through", strict=True)
def test_out_of_scope_with_shared_word(cp):
    assert cp.decide("recommend a good movie for tonight").action == "REFUSE"
