"""Tests for the remember/forget tool schemas and their decoders.

The schema/decoder agreement tests are the regression guard that keeps
slice 8's MCP server — which is documented to reuse this module verbatim —
from silently drifting: every property a schema promises must be consumed
by its decoder, or a client using the schema could send a field the decoder
ignores.
"""

from __future__ import annotations

import pytest

from memgit.agent.tools import (
    FORGET_SCHEMA,
    RECALL_SCHEMA,
    REMEMBER_SCHEMA,
    TOOL_SCHEMAS,
    ForgetCall,
    RecallCall,
    RememberCall,
    ToolCallError,
    decode_forget,
    decode_recall,
    decode_remember,
    tool_schemas,
)


class TestSchemas:
    @pytest.mark.parametrize("schema", [REMEMBER_SCHEMA, FORGET_SCHEMA, RECALL_SCHEMA])
    def test_strict_and_closed(self, schema):
        assert schema["strict"] is True
        assert schema["input_schema"]["additionalProperties"] is False

    @pytest.mark.parametrize("schema", [REMEMBER_SCHEMA, FORGET_SCHEMA, RECALL_SCHEMA])
    def test_every_property_is_required(self, schema):
        # A precondition for `strict: true` actually guaranteeing full input.
        properties = schema["input_schema"]["properties"]
        assert set(schema["input_schema"]["required"]) == set(properties)

    def test_tool_schemas_lists_both_write_tools(self):
        names = {schema["name"] for schema in TOOL_SCHEMAS}
        assert names == {"remember", "forget"}

    def test_tool_schemas_helper_omits_recall_by_default(self):
        names = {schema["name"] for schema in tool_schemas(recall=False)}
        assert names == {"remember", "forget"}

    def test_tool_schemas_helper_includes_recall_when_asked(self):
        names = {schema["name"] for schema in tool_schemas(recall=True)}
        assert names == {"remember", "forget", "recall"}


class TestDecodeRemember:
    def valid_payload(self, **overrides):
        payload = dict(
            subject="user",
            predicate="prefers_language",
            object="Python",
            confidence=0.9,
            source_text="I really like Python.",
        )
        payload.update(overrides)
        return payload

    def test_decodes_into_a_fact(self):
        call = decode_remember(self.valid_payload(), source="session:s/turn:1")
        assert isinstance(call, RememberCall)
        assert call.fact.triple == ("user", "prefers_language", "Python")
        assert call.fact.confidence == 0.9
        assert call.fact.source == "session:s/turn:1"
        assert call.fact.source_text == "I really like Python."

    def test_missing_field_raises_tool_call_error(self):
        payload = self.valid_payload()
        del payload["confidence"]
        with pytest.raises(ToolCallError):
            decode_remember(payload, source="s")

    def test_confidence_out_of_range_raises_tool_call_error(self):
        with pytest.raises(ToolCallError):
            decode_remember(self.valid_payload(confidence=1.4), source="s")

    def test_empty_subject_raises_tool_call_error(self):
        with pytest.raises(ToolCallError):
            decode_remember(self.valid_payload(subject="  "), source="s")

    def test_source_is_runtime_supplied_not_model_authored(self):
        payload = self.valid_payload()
        assert "source" not in payload  # not a schema property at all
        call = decode_remember(payload, source="session:x/turn:9")
        assert call.fact.source == "session:x/turn:9"


class TestDecodeForget:
    def test_decodes_whole_key_retraction(self):
        call = decode_forget({"subject": "user", "predicate": "has_pet", "object": None, "reason": "no more pet"})
        assert isinstance(call, ForgetCall)
        assert call.key == ("user", "has_pet")
        assert call.object is None
        assert call.reason == "no more pet"

    def test_decodes_single_value_retraction(self):
        call = decode_forget(
            {"subject": "user", "predicate": "likes", "object": "chess", "reason": "no longer true"}
        )
        assert call.object == "chess"

    def test_missing_field_raises_tool_call_error(self):
        with pytest.raises(ToolCallError):
            decode_forget({"subject": "user", "predicate": "likes", "object": None})

    def test_empty_subject_raises_tool_call_error(self):
        with pytest.raises(ToolCallError):
            decode_forget({"subject": "", "predicate": "likes", "object": None, "reason": "x"})


class TestDecodeRecall:
    def test_decodes_a_query(self):
        call = decode_recall({"query": "what editor do I use", "subject": None, "limit": 5})
        assert isinstance(call, RecallCall)
        assert call.query == "what editor do I use"
        assert call.subject is None
        assert call.limit == 5

    def test_decodes_with_a_subject_restriction(self):
        call = decode_recall({"query": "editor", "subject": "user", "limit": 3})
        assert call.subject == "user"

    def test_missing_field_raises_tool_call_error(self):
        with pytest.raises(ToolCallError):
            decode_recall({"query": "editor", "subject": None})

    def test_empty_query_raises_tool_call_error(self):
        with pytest.raises(ToolCallError):
            decode_recall({"query": "  ", "subject": None, "limit": 5})

    def test_non_string_subject_raises_tool_call_error(self):
        with pytest.raises(ToolCallError):
            decode_recall({"query": "editor", "subject": 5, "limit": 5})

    def test_non_positive_limit_raises_tool_call_error(self):
        with pytest.raises(ToolCallError):
            decode_recall({"query": "editor", "subject": None, "limit": 0})

    def test_bool_limit_raises_tool_call_error(self):
        with pytest.raises(ToolCallError):
            decode_recall({"query": "editor", "subject": None, "limit": True})
