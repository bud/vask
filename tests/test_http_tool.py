"""Tests for vask.tools.http.HttpTool."""

from unittest.mock import AsyncMock, patch

import pytest

from vask.tools.http import HttpTool


class TestHttpToolBlocking:
    """Test URL blocking logic."""

    def setup_method(self):
        self.tool = HttpTool()

    @pytest.mark.parametrize(
        "ip",
        ["127.0.0.1", "10.0.0.1", "192.168.1.1", "172.16.0.1"],
    )
    def test_blocks_private_ips(self, ip):
        assert self.tool._is_blocked_url(f"http://{ip}/path") is True

    def test_blocks_metadata_endpoint(self):
        assert self.tool._is_blocked_url("http://169.254.169.254/latest/meta-data") is True

    def test_blocks_google_metadata(self):
        assert self.tool._is_blocked_url("http://metadata.google.internal/computeMetadata") is True

    @pytest.mark.parametrize("scheme", ["file", "ftp", "gopher"])
    def test_blocks_non_http_schemes(self, scheme):
        assert self.tool._is_blocked_url(f"{scheme}:///etc/passwd") is True

    def test_blocks_localhost(self):
        assert self.tool._is_blocked_url("http://localhost:8080/api") is True

    def test_blocks_zero_address(self):
        assert self.tool._is_blocked_url("http://0.0.0.0/test") is True

    def test_allows_public_url(self):
        assert self.tool._is_blocked_url("https://api.example.com/v1/data") is False


class TestHttpToolExecute:
    """Test execution with mocked httpx."""

    @pytest.mark.asyncio
    async def test_blocked_url_returns_message(self):
        tool = HttpTool()
        result = await tool.execute({"method": "GET", "url": "http://127.0.0.1/secret"})
        assert "Blocked" in result

    @pytest.mark.asyncio
    async def test_allows_public_url_with_mock(self):
        tool = HttpTool()

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.text = '{"ok": true}'

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("vask.tools.http.httpx.AsyncClient", return_value=mock_client):
            result = await tool.execute({"method": "GET", "url": "https://api.example.com/data"})

        assert "200" in result
        assert '{"ok": true}' in result

    @pytest.mark.asyncio
    async def test_block_private_networks_can_be_disabled(self):
        from vask.config import ProviderConfig

        config = ProviderConfig(
            name="http",
            type="tool",
            extra={"block_private_networks": False},
        )
        tool = HttpTool(config=config)
        # Private IPs should no longer be blocked (but metadata endpoints still are)
        assert tool._is_blocked_url("http://192.168.1.1/api") is False
        # Metadata endpoints are always blocked
        assert tool._is_blocked_url("http://169.254.169.254/latest") is True
