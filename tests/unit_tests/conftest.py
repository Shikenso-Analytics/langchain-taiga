import json

import pytest
from httpx import Response

_USERS = {
    "alice": {"auth_token": "alice_jwt", "refresh": "alice_ref", "id": 1},
    "bob": {"auth_token": "bob_jwt", "refresh": "bob_ref", "id": 2},
}


@pytest.fixture
def taiga_auth_mock(respx_mock):
    """Answer ``POST {TAIGA_API}/api/v1/auth`` for alice and bob, else 401."""

    from tests.unit_tests.remote_app import TAIGA_API

    def handler(request):
        username = json.loads(request.content.decode()).get("username")
        user = _USERS.get(username)
        if user is None:
            return Response(401)
        return Response(200, json={**user, "username": username})

    respx_mock.post(f"{TAIGA_API}/api/v1/auth").mock(side_effect=handler)
    return respx_mock
