import httpx
import asyncio
from typing import Dict, Any, Optional


class HttpxClient:
    """
    基于 httpx 的 HTTP 请求工具类，支持同步和异步请求
    """

    def __init__(self, base_url: str = "", timeout: int = 30, headers: Dict[str, str] = None):
        """
        初始化 HTTP 客户端

        Args:
            base_url: API 基础 URL
            timeout: 请求超时时间(秒)
            headers: 默认请求头
        """
        self.base_url = base_url
        self.timeout = timeout
        self.default_headers = headers or {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    def _build_url(self, endpoint: str) -> str:
        """构建完整的 URL"""
        if self.base_url and not endpoint.startswith(('http://', 'https://')):
            return f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        return endpoint

    def _merge_headers(self, headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """合并请求头"""
        if headers:
            return {**self.default_headers, **headers}
        return self.default_headers

    def get(self, endpoint: str, params: Dict[str, Any] = None, headers: Dict[str, str] = None) -> httpx.Response:
        """
        发送 GET 请求

        Args:
            endpoint: API 端点或完整 URL
            params: 查询参数
            headers: 请求头

        Returns:
            httpx.Response 对象
        """
        url = self._build_url(endpoint)

        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(url, params=params,
                                  headers=self._merge_headers(headers))
            response.raise_for_status()
            return response

    def post(self, endpoint: str, data: Dict[str, Any] = None, json_data: Dict[str, Any] = None,
             params: Dict[str, Any] = None, headers: Dict[str, str] = None) -> httpx.Response:
        """
        发送 POST 请求

        Args:
            endpoint: API 端点或完整 URL
            data: 表单数据
            json_data: JSON 数据
            params: 查询参数
            headers: 请求头

        Returns:
            httpx.Response 对象
        """
        url = self._build_url(endpoint)

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                url,
                data=data,
                json=json_data,
                params=params,
                headers=self._merge_headers(headers)
            )
            response.raise_for_status()
            return response

    def put(self, endpoint: str, data: Dict[str, Any] = None, json_data: Dict[str, Any] = None,
            params: Dict[str, Any] = None, headers: Dict[str, str] = None) -> httpx.Response:
        """
        发送 PUT 请求

        Args:
            endpoint: API 端点或完整 URL
            data: 表单数据
            json_data: JSON 数据
            params: 查询参数
            headers: 请求头

        Returns:
            httpx.Response 对象
        """
        url = self._build_url(endpoint)

        with httpx.Client(timeout=self.timeout) as client:
            response = client.put(
                url,
                data=data,
                json=json_data,
                params=params,
                headers=self._merge_headers(headers)
            )
            response.raise_for_status()
            return response

    def delete(self, endpoint: str, params: Dict[str, Any] = None,
               headers: Dict[str, str] = None) -> httpx.Response:
        """
        发送 DELETE 请求

        Args:
            endpoint: API 端点或完整 URL
            params: 查询参数
            headers: 请求头

        Returns:
            httpx.Response 对象
        """
        url = self._build_url(endpoint)

        with httpx.Client(timeout=self.timeout) as client:
            response = client.delete(
                url, params=params, headers=self._merge_headers(headers))
            response.raise_for_status()
            return response

    # 异步方法
    async def async_get(self, endpoint: str, params: Dict[str, Any] = None,
                        headers: Dict[str, str] = None) -> httpx.Response:
        """
        异步发送 GET 请求

        Args:
            endpoint: API 端点或完整 URL
            params: 查询参数
            headers: 请求头

        Returns:
            httpx.Response 对象
        """
        url = self._build_url(endpoint)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url, params=params, headers=self._merge_headers(headers))
            response.raise_for_status()
            return response

    async def async_post(self, endpoint: str, data: Dict[str, Any] = None,
                         json_data: Dict[str, Any] = None, params: Dict[str, Any] = None,
                         headers: Dict[str, str] = None) -> httpx.Response:
        """
        异步发送 POST 请求

        Args:
            endpoint: API 端点或完整 URL
            data: 表单数据
            json_data: JSON 数据
            params: 查询参数
            headers: 请求头

        Returns:
            httpx.Response 对象
        """
        url = self._build_url(endpoint)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                url,
                data=data,
                json=json_data,
                params=params,
                headers=self._merge_headers(headers)
            )
            response.raise_for_status()
            return response

    async def async_put(self, endpoint: str, data: Dict[str, Any] = None,
                        json_data: Dict[str, Any] = None, params: Dict[str, Any] = None,
                        headers: Dict[str, str] = None) -> httpx.Response:
        """
        异步发送 PUT 请求

        Args:
            endpoint: API 端点或完整 URL
            data: 表单数据
            json_data: JSON 数据
            params: 查询参数
            headers: 请求头

        Returns:
            httpx.Response 对象
        """
        url = self._build_url(endpoint)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.put(
                url,
                data=data,
                json=json_data,
                params=params,
                headers=self._merge_headers(headers)
            )
            response.raise_for_status()
            return response

    async def async_delete(self, endpoint: str, params: Dict[str, Any] = None,
                           headers: Dict[str, str] = None) -> httpx.Response:
        """
        异步发送 DELETE 请求

        Args:
            endpoint: API 端点或完整 URL
            params: 查询参数
            headers: 请求头

        Returns:
            httpx.Response 对象
        """
        url = self._build_url(endpoint)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.delete(url, params=params, headers=self._merge_headers(headers))
            response.raise_for_status()
            return response
