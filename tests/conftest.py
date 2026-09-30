"""pytest 公共夹具：临时数据目录 + FastAPI TestClient。"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.doa.geometry import ULA
from app.doa.synthesis import synthesize


@pytest.fixture
def client(tmp_path):
    app = create_app(str(tmp_path / "data"))
    with TestClient(app) as c:
        yield c


@pytest.fixture
def array8():
    return ULA(8, 0.5, 1.0)


@pytest.fixture
def make_data(array8):
    def _make(angles, snr=20.0, k=500, seed=42, powers=None,
              coherent=None):
        powers = powers if powers is not None else [1.0] * len(angles)
        return synthesize(array8, angles, powers, snr, k, seed,
                          coherent_flags=coherent)
    return _make


def create_session(client, m=8, spacing=0.5, wavelength=1.0,
                   position_errors=None, update=None, estimate=None):
    body = {
        "array": {
            "m": m, "spacing": spacing, "wavelength": wavelength,
            "position_errors": position_errors,
        },
    }
    if update is not None:
        body["update"] = update
    if estimate is not None:
        body["estimate"] = estimate
    r = client.post("/sessions", json=body)
    assert r.status_code == 201, r.text
    return r.json()["session_id"]


def append_synth(client, sid, X, batch_id):
    r = client.post(
        f"/sessions/{sid}/batches",
        json={
            "batch_id": batch_id,
            "real": X.real.tolist(),
            "imag": X.imag.tolist(),
        },
    )
    return r


def do_estimate(client, sid, **overrides):
    return client.post(f"/sessions/{sid}/estimate", json=overrides)
