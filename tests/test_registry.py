"""Tests for vask.core.registry.Registry."""

import pytest

from vask.core.registry import Registry


class TestRegistry:
    def test_register_and_create_provider(self):
        reg = Registry()
        factory = lambda: "created_instance"
        reg.register("llm", "test", factory)
        result = reg.create("llm", "test")
        assert result == "created_instance"

    def test_create_unknown_provider_raises_key_error(self):
        reg = Registry()
        with pytest.raises(KeyError, match="No 'llm' provider named 'nonexistent'"):
            reg.create("llm", "nonexistent")

    def test_register_unknown_category_raises_value_error(self):
        reg = Registry()
        with pytest.raises(ValueError, match="Unknown category 'bogus'"):
            reg.register("bogus", "test", lambda: None)

    def test_list_providers_returns_registered_names(self):
        reg = Registry()
        reg.register("input", "mic", lambda: None)
        reg.register("input", "file", lambda: None)
        names = reg.list_providers("input")
        assert sorted(names) == ["file", "mic"]

    def test_list_providers_empty_category(self):
        reg = Registry()
        assert reg.list_providers("llm") == []

    def test_list_providers_unknown_category(self):
        reg = Registry()
        assert reg.list_providers("nonexistent") == []

    def test_create_with_config(self):
        from vask.config import ProviderConfig

        reg = Registry()
        received = {}

        def factory(config):
            received["config"] = config
            return "instance"

        reg.register("llm", "test", factory)
        config = ProviderConfig(name="test", type="llm")
        result = reg.create("llm", "test", config)
        assert result == "instance"
        assert received["config"] is config
