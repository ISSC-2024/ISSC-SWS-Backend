import pytest
from httpx import AsyncClient
from main import app
from tortoise.contrib.test import initializer, finalizer


@pytest.fixture(scope="module", autouse=True)
def initialize_tests():
    initializer(["app.models"])
    yield
    finalizer()


@pytest.mark.asyncio
async def test_algorithm2_endpoint():
    async with AsyncClient(app=app, base_url="http://localhost") as client:
        response = await client.post("/api/algorithm2", json={"input_data": "test"})
        assert response.status_code == 200
        assert "result" in response.json()
