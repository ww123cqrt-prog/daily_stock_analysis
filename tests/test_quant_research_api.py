# -*- coding: utf-8 -*-
"""
Tests for the quant research API and repository.
"""

from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path

import pytest

from src.models.algorithm_registry import (
    algorithm_to_dict,
    get_algorithm,
    list_algorithms,
)
from src.repositories.quant_research_repo import QuantResearchRepository
from src.schemas.quant_research import Signal


# === Repository Tests ===

@pytest.fixture
def repo(tmp_path):
    """Create a temporary repository for testing."""
    db_path = tmp_path / "test_quant_research.db"
    return QuantResearchRepository(db_path=db_path)


class TestQuantResearchRepository:
    def test_create_and_get_research(self, repo):
        research = repo.create_research(
            name="Test Research",
            description="A test research entry",
            research_type="stock_selection",
            config={"universe": "csi800"},
        )
        assert research["name"] == "Test Research"
        assert research["status"] == "draft"
        assert research["research_type"] == "stock_selection"
        assert research["config"]["universe"] == "csi800"

        fetched = repo.get_research(research["id"])
        assert fetched is not None
        assert fetched["id"] == research["id"]

    def test_list_research(self, repo):
        initial = len(repo.list_research(limit=100))
        repo.create_research(name="R1")
        repo.create_research(name="R2")
        repo.create_research(name="R3")

        results = repo.list_research(limit=100)
        assert len(results) == initial + 3

    def test_update_research(self, repo):
        research = repo.create_research(name="Original")
        updated = repo.update_research(research["id"], name="Updated", status="active")
        assert updated["name"] == "Updated"
        assert updated["status"] == "active"

    def test_delete_research(self, repo):
        research = repo.create_research(name="To Delete")
        repo.delete_research(research["id"])
        assert repo.get_research(research["id"]) is None

    def test_delete_research_with_draft_experiments_no_runs(self, repo):
        research = repo.create_research(name="R")
        repo.create_experiment(research_id=research["id"], name="E1", method_kind="rule")
        repo.create_experiment(research_id=research["id"], name="E2", method_kind="rule")
        repo.delete_research(research["id"])
        assert repo.get_research(research["id"]) is None

    def test_delete_research_with_frozen_experiment_rejected(self, repo):
        research = repo.create_research(name="R")
        exp = repo.create_experiment(research_id=research["id"], name="E", method_kind="rule")
        repo.update_experiment(exp["id"], status="submitted")
        with pytest.raises(ValueError, match="frozen"):
            repo.delete_research(research["id"])

    def test_delete_research_with_runs_rejected(self, repo):
        research = repo.create_research(name="R")
        exp = repo.create_experiment(research_id=research["id"], name="E", method_kind="rule")
        repo.update_experiment(exp["id"], status="submitted")
        repo.create_run(experiment_id=exp["id"])
        with pytest.raises(ValueError, match="run"):
            repo.delete_research(research["id"])

    def test_create_experiment(self, repo):
        research = repo.create_research(name="Parent Research")
        experiment = repo.create_experiment(
            research_id=research["id"],
            name="LightGBM Experiment",
            experiment_type="stock_selection",
            method_kind="algorithm",
            algorithm_id="lightgbm",
            protocol={"dataset": "ashare_daily_qfq", "train_days": 504},
        )
        assert experiment["name"] == "LightGBM Experiment"
        assert experiment["algorithm_id"] == "lightgbm"
        assert experiment["status"] == "draft"

    def test_list_experiments(self, repo):
        research = repo.create_research(name="Research")
        repo.create_experiment(research_id=research["id"], name="E1")
        repo.create_experiment(research_id=research["id"], name="E2")

        experiments = repo.list_experiments(research["id"])
        assert len(experiments) == 2

    def test_delete_draft_experiment(self, repo):
        research = repo.create_research(name="R")
        exp = repo.create_experiment(research_id=research["id"], name="E", method_kind="rule")
        repo.delete_experiment(exp["id"])
        assert repo.get_experiment(exp["id"]) is None

    def test_delete_frozen_experiment_rejected(self, repo):
        research = repo.create_research(name="R")
        exp = repo.create_experiment(research_id=research["id"], name="E", method_kind="rule")
        repo.update_experiment(exp["id"], status="submitted")
        with pytest.raises(ValueError, match="frozen"):
            repo.delete_experiment(exp["id"])

    def test_delete_experiment_with_runs_rejected(self, repo):
        research = repo.create_research(name="R")
        exp = repo.create_experiment(research_id=research["id"], name="E", method_kind="rule")
        repo.update_experiment(exp["id"], status="submitted")
        repo.create_run(experiment_id=exp["id"])
        with pytest.raises(ValueError, match="run"):
            repo.delete_experiment(exp["id"])

    def test_create_run(self, repo):
        research = repo.create_research(name="R")
        experiment = repo.create_experiment(research_id=research["id"], name="E", method_kind="rule")
        repo.update_experiment(experiment["id"], status="submitted")
        run = repo.create_run(
            experiment_id=experiment["id"],
            run_label="First run",
            mlflow_run_id="mlflow-123",
        )
        assert run["status"] == "pending"
        assert run["run_label"] == "First run"
        assert run["mlflow_run_id"] == "mlflow-123"

    def test_create_run_draft_rejected(self, repo):
        research = repo.create_research(name="R")
        experiment = repo.create_experiment(research_id=research["id"], name="E", method_kind="rule")
        assert experiment["status"] == "draft"
        with pytest.raises(ValueError, match="frozen"):
            repo.create_run(experiment_id=experiment["id"])

    def test_update_run(self, repo):
        research = repo.create_research(name="R")
        experiment = repo.create_experiment(research_id=research["id"], name="E", method_kind="rule")
        repo.update_experiment(experiment["id"], status="submitted")
        run = repo.create_run(experiment_id=experiment["id"])

        updated = repo.update_run(
            run["id"],
            status="completed",
            metrics={"sharpe": 1.5, "max_drawdown": -0.15},
        )
        assert updated["status"] == "completed"
        assert updated["metrics"]["sharpe"] == 1.5

    def test_get_research_with_experiments(self, repo):
        research = repo.create_research(name="Full Research")
        exp = repo.create_experiment(research_id=research["id"], name="E1", method_kind="rule")
        repo.update_experiment(exp["id"], status="submitted")
        repo.create_run(experiment_id=exp["id"], run_label="R1")
        repo.create_run(experiment_id=exp["id"], run_label="R2")

        full = repo.get_research_with_experiments(research["id"])
        assert full is not None
        assert len(full["experiments"]) == 1
        assert len(full["experiments"][0]["runs"]) == 2

    def test_get_overview(self, repo):
        initial = repo.get_overview()
        repo.create_research(name="R1")
        repo.create_research(name="R2")
        overview = repo.get_overview()
        assert overview["total_research"] == initial["total_research"] + 2
        assert overview["total_experiments"] == 0
        assert overview["total_runs"] == 0

    def test_legacy_run(self, repo):
        research = repo.create_research(name="R")
        experiment = repo.create_experiment(research_id=research["id"], name="E", method_kind="rule")
        repo.update_experiment(experiment["id"], status="submitted")
        run = repo.create_run(experiment_id=experiment["id"], is_legacy=True)
        assert run["is_legacy"] is True

    def test_protocol_hash_computed_on_submit(self, repo):
        research = repo.create_research(name="R")
        experiment = repo.create_experiment(
            research_id=research["id"],
            name="E",
            experiment_type="stock_selection",
            method_kind="algorithm",
            algorithm_id="lightgbm",
            protocol={"universe": "csi800", "train_days": 504},
        )
        assert experiment["protocol_hash"] is None
        assert experiment["status"] == "draft"

        submitted = repo.update_experiment(experiment["id"], status="submitted")
        assert submitted["status"] == "submitted"
        assert submitted["protocol_hash"] is not None
        assert len(submitted["protocol_hash"]) == 64

    def test_protocol_freeze_rejects_protocol_change(self, repo):
        research = repo.create_research(name="R")
        experiment = repo.create_experiment(
            research_id=research["id"],
            name="E",
            method_kind="rule",
            protocol={"universe": "csi800"},
        )
        repo.update_experiment(experiment["id"], status="submitted")

        with pytest.raises(ValueError, match="frozen"):
            repo.update_experiment(experiment["id"], protocol={"universe": "csi500"})

    def test_frozen_rejects_experiment_type_change(self, repo):
        research = repo.create_research(name="R")
        experiment = repo.create_experiment(
            research_id=research["id"],
            name="E",
            method_kind="rule",
            experiment_type="stock_selection",
        )
        repo.update_experiment(experiment["id"], status="submitted")

        with pytest.raises(ValueError, match="frozen"):
            repo.update_experiment(experiment["id"], experiment_type="strategy")

    def test_frozen_rejects_method_kind_change(self, repo):
        research = repo.create_research(name="R")
        experiment = repo.create_experiment(
            research_id=research["id"],
            name="E",
            method_kind="rule",
        )
        repo.update_experiment(experiment["id"], status="submitted")

        with pytest.raises(ValueError, match="frozen"):
            repo.update_experiment(experiment["id"], method_kind="algorithm")

    def test_frozen_rejects_algorithm_id_change(self, repo):
        research = repo.create_research(name="R")
        experiment = repo.create_experiment(
            research_id=research["id"],
            name="E",
            method_kind="algorithm",
            algorithm_id="lightgbm",
        )
        repo.update_experiment(experiment["id"], status="submitted")

        with pytest.raises(ValueError, match="frozen"):
            repo.update_experiment(experiment["id"], algorithm_id="xgboost")

    def test_frozen_rejects_status_rollback_to_draft(self, repo):
        research = repo.create_research(name="R")
        experiment = repo.create_experiment(research_id=research["id"], name="E", method_kind="rule")
        repo.update_experiment(experiment["id"], status="submitted")

        with pytest.raises(ValueError, match="draft"):
            repo.update_experiment(experiment["id"], status="draft")

    def test_frozen_allows_name_description_change(self, repo):
        research = repo.create_research(name="R")
        experiment = repo.create_experiment(research_id=research["id"], name="E", method_kind="rule")
        repo.update_experiment(experiment_id=experiment["id"], status="submitted")

        updated = repo.update_experiment(experiment["id"], name="New Name", description="New desc")
        assert updated["name"] == "New Name"
        assert updated["description"] == "New desc"

    def test_hash_includes_all_research_definition_fields(self, repo):
        research = repo.create_research(name="R")
        e1 = repo.create_experiment(
            research_id=research["id"],
            name="E1",
            experiment_type="stock_selection",
            method_kind="algorithm",
            algorithm_id="lightgbm",
            protocol={"universe": "csi800"},
        )
        repo.update_experiment(e1["id"], status="submitted")

        e2 = repo.create_experiment(
            research_id=research["id"],
            name="E2",
            experiment_type="strategy",
            method_kind="rule",
            algorithm_id=None,
            protocol={"universe": "csi800"},
        )
        repo.update_experiment(e2["id"], status="submitted")

        e1_fresh = repo.get_experiment(e1["id"])
        e2_fresh = repo.get_experiment(e2["id"])
        assert e1_fresh["protocol_hash"] != e2_fresh["protocol_hash"]

    def test_hash_key_order_stability(self, repo):
        from src.repositories.quant_research_repo import _compute_research_definition_hash
        h1 = _compute_research_definition_hash("stock_selection", "algorithm", "lightgbm", {"b": 2, "a": 1})
        h2 = _compute_research_definition_hash("stock_selection", "algorithm", "lightgbm", {"a": 1, "b": 2})
        assert h1 == h2

    def test_tampering_detected_on_integrity_check(self, repo):
        research = repo.create_research(name="R")
        experiment = repo.create_experiment(
            research_id=research["id"],
            name="E",
            algorithm_id="lightgbm",
            protocol={"universe": "csi800"},
        )
        repo.update_experiment(experiment["id"], status="submitted")

        conn = repo._get_connection()
        conn.execute(
            "UPDATE experiment SET algorithm_id = ? WHERE id = ?",
            ("xgboost", experiment["id"]),
        )
        conn.commit()

        assert repo.verify_experiment_integrity(experiment["id"]) is False

    def test_tampering_blocks_run_creation(self, repo):
        research = repo.create_research(name="R")
        experiment = repo.create_experiment(
            research_id=research["id"],
            name="E",
            algorithm_id="lightgbm",
        )
        repo.update_experiment(experiment["id"], status="submitted")

        conn = repo._get_connection()
        conn.execute(
            "UPDATE experiment SET algorithm_id = ? WHERE id = ?",
            ("xgboost", experiment["id"]),
        )
        conn.commit()

        with pytest.raises(ValueError, match="integrity"):
            repo.create_run(experiment_id=experiment["id"])

    def test_tampering_detected_on_read(self, repo):
        research = repo.create_research(name="R")
        experiment = repo.create_experiment(
            research_id=research["id"],
            name="E",
            algorithm_id="lightgbm",
        )
        repo.update_experiment(experiment["id"], status="submitted")

        conn = repo._get_connection()
        conn.execute(
            "UPDATE experiment SET algorithm_id = ? WHERE id = ?",
            ("xgboost", experiment["id"]),
        )
        conn.commit()

        with pytest.raises(ValueError, match="integrity"):
            repo.get_experiment_checked(experiment["id"])

    def test_tampering_detected_on_list(self, repo):
        research = repo.create_research(name="R")
        experiment = repo.create_experiment(
            research_id=research["id"],
            name="E",
            algorithm_id="lightgbm",
        )
        repo.update_experiment(experiment["id"], status="submitted")

        conn = repo._get_connection()
        conn.execute(
            "UPDATE experiment SET algorithm_id = ? WHERE id = ?",
            ("xgboost", experiment["id"]),
        )
        conn.commit()

        with pytest.raises(ValueError, match="integrity"):
            repo.get_research_with_experiments(research["id"])


# === Algorithm Registry Tests ===

class TestAlgorithmRegistry:
    def test_list_algorithms(self):
        algos = list_algorithms()
        assert len(algos) >= 6
        ids = [a.id for a in algos]
        assert "rule" in ids
        assert "linear_ols" in ids
        assert "lightgbm" in ids
        assert "xgboost" in ids
        assert "catboost" in ids

    def test_get_algorithm(self):
        algo = get_algorithm("lightgbm")
        assert algo is not None
        assert algo.name == "LightGBM"
        assert algo.name_zh == "LightGBM"
        assert algo.category == "tree_ensemble"

    def test_get_algorithm_not_found(self):
        assert get_algorithm("nonexistent") is None

    def test_filter_by_category(self):
        tree_algos = list_algorithms(category="tree_ensemble")
        assert all(a.category == "tree_ensemble" for a in tree_algos)
        assert len(tree_algos) >= 3

    def test_algorithm_to_dict(self):
        algo = get_algorithm("catboost")
        assert algo is not None
        d = algorithm_to_dict(algo)
        assert d["id"] == "catboost"
        assert "capability" in d
        assert "config_schema" in d
        assert "default_params" in d

    def test_algorithms_have_config_schema(self):
        for algo in list_algorithms():
            assert algo.config_schema is not None
            assert isinstance(algo.config_schema.parameters, dict)


# === Signal Contract Tests ===

class TestSignalContract:
    def test_create_signal(self):
        signal = Signal(
            date="2025-06-30",
            symbol="SH600519",
            score=0.85,
            rank=1,
            expected_return=0.03,
            confidence=0.72,
            producer_id="abc-123",
            producer_type="algorithm",
            algorithm_id="lightgbm",
        )
        assert signal.date == "2025-06-30"
        assert signal.symbol == "SH600519"
        assert signal.score == 0.85

    def test_signal_json_serialization(self):
        signal = Signal(date="2025-01-01", symbol="SZ000001", score=0.5)
        data = signal.model_dump()
        assert data["date"] == "2025-01-01"
        assert data["symbol"] == "SZ000001"

    def test_signal_minimal(self):
        signal = Signal(date="2025-01-01", symbol="X", score=0.0)
        assert signal.rank is None
        assert signal.expected_return is None


# === API Integration Tests ===

@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient
    from api.v1.router import router as v1_router
    from fastapi import FastAPI

    import api.v1.endpoints.quant_research as qr_endpoint
    from src.repositories.quant_research_repo import QuantResearchRepository

    test_repo = QuantResearchRepository(db_path=tmp_path / "test.db")
    qr_endpoint._repo = test_repo

    app = FastAPI()
    app.include_router(v1_router, prefix="/api/v1")
    app.state._test_repo = test_repo
    return TestClient(app)


class TestQuantResearchAPI:
    def test_overview(self, client):
        response = client.get("/api/v1/quant-research/overview")
        assert response.status_code == 200
        data = response.json()
        assert "total_research" in data
        assert "total_experiments" in data
        assert "total_runs" in data

    def test_create_research(self, client):
        response = client.post("/api/v1/quant-research/research", json={
            "name": "My Research",
            "description": "Testing",
            "research_type": "stock_selection",
        })
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "My Research"
        assert data["status"] == "draft"

    def test_list_research(self, client):
        client.post("/api/v1/quant-research/research", json={"name": "R1"})
        client.post("/api/v1/quant-research/research", json={"name": "R2"})

        response = client.get("/api/v1/quant-research/research")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 2

    def test_get_research_not_found(self, client):
        response = client.get("/api/v1/quant-research/research/nonexistent")
        assert response.status_code == 404

    def test_create_experiment(self, client):
        r = client.post("/api/v1/quant-research/research", json={"name": "R"})
        research_id = r.json()["id"]

        response = client.post(
            f"/api/v1/quant-research/research/{research_id}/experiments",
            json={
                "name": "LightGBM Test",
                "experiment_type": "stock_selection",
                "method_kind": "algorithm",
                "algorithm_id": "lightgbm",
                "protocol": {"train_days": 504},
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "LightGBM Test"
        assert data["algorithm_id"] == "lightgbm"

    def test_create_experiment_invalid_algorithm(self, client):
        r = client.post("/api/v1/quant-research/research", json={"name": "R"})
        research_id = r.json()["id"]

        response = client.post(
            f"/api/v1/quant-research/research/{research_id}/experiments",
            json={
                "name": "Bad Algo",
                "method_kind": "algorithm",
                "algorithm_id": "nonexistent_algo",
            },
        )
        assert response.status_code == 400

    def test_create_run(self, client):
        r = client.post("/api/v1/quant-research/research", json={"name": "R"})
        research_id = r.json()["id"]

        e = client.post(
            f"/api/v1/quant-research/research/{research_id}/experiments",
            json={"name": "E", "method_kind": "rule"},
        )
        assert e.status_code == 201, e.json()
        experiment_id = e.json()["id"]

        client.patch(
            f"/api/v1/quant-research/experiments/{experiment_id}",
            json={"status": "submitted"},
        )

        response = client.post(
            f"/api/v1/quant-research/experiments/{experiment_id}/runs",
            json={"run_label": "Test Run"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "pending"

    def test_list_algorithms(self, client):
        response = client.get("/api/v1/quant-research/algorithms")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 6

    def test_get_algorithm(self, client):
        response = client.get("/api/v1/quant-research/algorithms/lightgbm")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "lightgbm"

    def test_get_algorithm_not_found(self, client):
        response = client.get("/api/v1/quant-research/algorithms/nonexistent")
        assert response.status_code == 404

    def test_delete_research(self, client):
        r = client.post("/api/v1/quant-research/research", json={"name": "ToDelete"})
        research_id = r.json()["id"]

        response = client.delete(f"/api/v1/quant-research/research/{research_id}")
        assert response.status_code == 204

        response = client.get(f"/api/v1/quant-research/research/{research_id}")
        assert response.status_code == 404

    def test_delete_research_with_frozen_experiment_rejected(self, client):
        r = client.post("/api/v1/quant-research/research", json={"name": "R"})
        research_id = r.json()["id"]
        e = client.post(
            f"/api/v1/quant-research/research/{research_id}/experiments",
            json={"name": "E", "method_kind": "rule"},
        )
        experiment_id = e.json()["id"]
        client.patch(f"/api/v1/quant-research/experiments/{experiment_id}", json={"status": "submitted"})

        response = client.delete(f"/api/v1/quant-research/research/{research_id}")
        assert response.status_code == 409
        assert "frozen" in response.json()["detail"]

    def test_delete_research_with_runs_rejected(self, client):
        r = client.post("/api/v1/quant-research/research", json={"name": "R"})
        research_id = r.json()["id"]
        e = client.post(
            f"/api/v1/quant-research/research/{research_id}/experiments",
            json={"name": "E", "method_kind": "rule"},
        )
        experiment_id = e.json()["id"]
        client.patch(f"/api/v1/quant-research/experiments/{experiment_id}", json={"status": "submitted"})
        client.post(f"/api/v1/quant-research/experiments/{experiment_id}/runs", json={"run_label": "R1"})

        response = client.delete(f"/api/v1/quant-research/research/{research_id}")
        assert response.status_code == 409
        assert "run" in response.json()["detail"]

    def test_delete_frozen_experiment_rejected(self, client):
        r = client.post("/api/v1/quant-research/research", json={"name": "R"})
        research_id = r.json()["id"]
        e = client.post(
            f"/api/v1/quant-research/research/{research_id}/experiments",
            json={"name": "E", "method_kind": "rule"},
        )
        experiment_id = e.json()["id"]
        client.patch(f"/api/v1/quant-research/experiments/{experiment_id}", json={"status": "submitted"})

        response = client.delete(f"/api/v1/quant-research/experiments/{experiment_id}")
        assert response.status_code == 409
        assert "frozen" in response.json()["detail"]

    def test_delete_experiment_with_runs_rejected(self, client):
        r = client.post("/api/v1/quant-research/research", json={"name": "R"})
        research_id = r.json()["id"]
        e = client.post(
            f"/api/v1/quant-research/research/{research_id}/experiments",
            json={"name": "E", "method_kind": "rule"},
        )
        experiment_id = e.json()["id"]
        client.patch(f"/api/v1/quant-research/experiments/{experiment_id}", json={"status": "submitted"})
        client.post(f"/api/v1/quant-research/experiments/{experiment_id}/runs", json={"run_label": "R1"})

        response = client.delete(f"/api/v1/quant-research/experiments/{experiment_id}")
        assert response.status_code == 409
        assert "run" in response.json()["detail"]

    def test_delete_research_not_found(self, client):
        response = client.delete("/api/v1/quant-research/research/nonexistent-id")
        assert response.status_code == 404

    def test_delete_experiment_not_found(self, client):
        response = client.delete("/api/v1/quant-research/experiments/nonexistent-id")
        assert response.status_code == 404

    def test_delete_draft_experiment_succeeds(self, client):
        r = client.post("/api/v1/quant-research/research", json={"name": "R"})
        research_id = r.json()["id"]
        e = client.post(
            f"/api/v1/quant-research/research/{research_id}/experiments",
            json={"name": "E", "method_kind": "rule"},
        )
        experiment_id = e.json()["id"]

        response = client.delete(f"/api/v1/quant-research/experiments/{experiment_id}")
        assert response.status_code == 204

        response = client.get(f"/api/v1/quant-research/experiments/{experiment_id}")
        assert response.status_code == 404

    def test_submit_experiment_computes_hash(self, client):
        r = client.post("/api/v1/quant-research/research", json={"name": "R"})
        research_id = r.json()["id"]

        e = client.post(
            f"/api/v1/quant-research/research/{research_id}/experiments",
            json={"name": "E", "method_kind": "rule", "protocol": {"universe": "csi800"}},
        )
        assert e.status_code == 201, e.json()
        experiment_id = e.json()["id"]
        assert e.json()["protocol_hash"] is None

        response = client.patch(
            f"/api/v1/quant-research/experiments/{experiment_id}",
            json={"status": "submitted"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "submitted"
        assert data["protocol_hash"] is not None

    def test_frozen_experiment_rejects_protocol_change(self, client):
        r = client.post("/api/v1/quant-research/research", json={"name": "R"})
        research_id = r.json()["id"]

        e = client.post(
            f"/api/v1/quant-research/research/{research_id}/experiments",
            json={"name": "E", "method_kind": "rule", "protocol": {"universe": "csi800"}},
        )
        assert e.status_code == 201, e.json()
        experiment_id = e.json()["id"]

        client.patch(
            f"/api/v1/quant-research/experiments/{experiment_id}",
            json={"status": "submitted"},
        )

        response = client.patch(
            f"/api/v1/quant-research/experiments/{experiment_id}",
            json={"protocol": {"universe": "csi500"}},
        )
        assert response.status_code == 409

    def test_frozen_experiment_rejects_algorithm_id_change(self, client):
        r = client.post("/api/v1/quant-research/research", json={"name": "R"})
        research_id = r.json()["id"]

        e = client.post(
            f"/api/v1/quant-research/research/{research_id}/experiments",
            json={"name": "E", "algorithm_id": "lightgbm", "method_kind": "algorithm"},
        )
        assert e.status_code == 201, e.json()
        experiment_id = e.json()["id"]

        client.patch(
            f"/api/v1/quant-research/experiments/{experiment_id}",
            json={"status": "submitted"},
        )

        response = client.patch(
            f"/api/v1/quant-research/experiments/{experiment_id}",
            json={"algorithm_id": "xgboost"},
        )
        assert response.status_code == 409

    def test_frozen_experiment_rejects_status_rollback(self, client):
        r = client.post("/api/v1/quant-research/research", json={"name": "R"})
        research_id = r.json()["id"]

        e = client.post(
            f"/api/v1/quant-research/research/{research_id}/experiments",
            json={"name": "E", "method_kind": "rule"},
        )
        assert e.status_code == 201, e.json()
        experiment_id = e.json()["id"]

        client.patch(
            f"/api/v1/quant-research/experiments/{experiment_id}",
            json={"status": "submitted"},
        )

        response = client.patch(
            f"/api/v1/quant-research/experiments/{experiment_id}",
            json={"status": "draft"},
        )
        assert response.status_code == 409

    def test_tampering_blocks_run_creation(self, client):
        r = client.post("/api/v1/quant-research/research", json={"name": "R"})
        research_id = r.json()["id"]

        e = client.post(
            f"/api/v1/quant-research/research/{research_id}/experiments",
            json={"name": "E", "algorithm_id": "lightgbm", "method_kind": "algorithm"},
        )
        experiment_id = e.json()["id"]

        client.patch(
            f"/api/v1/quant-research/experiments/{experiment_id}",
            json={"status": "submitted"},
        )

        repo = client.app.state._test_repo
        conn = repo._get_connection()
        conn.execute(
            "UPDATE experiment SET algorithm_id = ? WHERE id = ?",
            ("xgboost", experiment_id),
        )
        conn.commit()

        response = client.post(
            f"/api/v1/quant-research/experiments/{experiment_id}/runs",
            json={"run_label": "Should Fail"},
        )
        assert response.status_code == 409

    def test_create_experiment_algorithm_requires_id(self, client):
        r = client.post("/api/v1/quant-research/research", json={"name": "R"})
        research_id = r.json()["id"]

        response = client.post(
            f"/api/v1/quant-research/research/{research_id}/experiments",
            json={"name": "E", "method_kind": "algorithm"},
        )
        assert response.status_code == 400
        assert "algorithm_id is required" in response.json()["detail"]

    def test_create_experiment_rule_rejects_algorithm_id(self, client):
        r = client.post("/api/v1/quant-research/research", json={"name": "R"})
        research_id = r.json()["id"]

        response = client.post(
            f"/api/v1/quant-research/research/{research_id}/experiments",
            json={"name": "E", "method_kind": "rule", "algorithm_id": "lightgbm"},
        )
        assert response.status_code == 400
        assert "must be empty" in response.json()["detail"]

    def test_update_draft_to_algorithm_without_id_rejected(self, client):
        r = client.post("/api/v1/quant-research/research", json={"name": "R"})
        research_id = r.json()["id"]
        e = client.post(
            f"/api/v1/quant-research/research/{research_id}/experiments",
            json={"name": "E", "method_kind": "rule"},
        )
        experiment_id = e.json()["id"]

        response = client.patch(
            f"/api/v1/quant-research/experiments/{experiment_id}",
            json={"method_kind": "algorithm"},
        )
        assert response.status_code == 400
        assert "algorithm_id is required" in response.json()["detail"]

    def test_update_draft_to_unknown_algorithm_rejected(self, client):
        r = client.post("/api/v1/quant-research/research", json={"name": "R"})
        research_id = r.json()["id"]
        e = client.post(
            f"/api/v1/quant-research/research/{research_id}/experiments",
            json={"name": "E", "method_kind": "algorithm", "algorithm_id": "lightgbm"},
        )
        experiment_id = e.json()["id"]

        response = client.patch(
            f"/api/v1/quant-research/experiments/{experiment_id}",
            json={"algorithm_id": "nonexistent_algo"},
        )
        assert response.status_code == 400
        assert "Unknown algorithm" in response.json()["detail"]

    def test_update_draft_algorithm_to_rule_with_leftover_id_rejected(self, client):
        r = client.post("/api/v1/quant-research/research", json={"name": "R"})
        research_id = r.json()["id"]
        e = client.post(
            f"/api/v1/quant-research/research/{research_id}/experiments",
            json={"name": "E", "method_kind": "algorithm", "algorithm_id": "lightgbm"},
        )
        experiment_id = e.json()["id"]

        response = client.patch(
            f"/api/v1/quant-research/experiments/{experiment_id}",
            json={"method_kind": "rule"},
        )
        assert response.status_code == 400
        assert "must be empty" in response.json()["detail"]

    def test_update_draft_algorithm_id_and_method_together_succeeds(self, client):
        r = client.post("/api/v1/quant-research/research", json={"name": "R"})
        research_id = r.json()["id"]
        e = client.post(
            f"/api/v1/quant-research/research/{research_id}/experiments",
            json={"name": "E", "method_kind": "algorithm", "algorithm_id": "lightgbm"},
        )
        experiment_id = e.json()["id"]

        response = client.patch(
            f"/api/v1/quant-research/experiments/{experiment_id}",
            json={"method_kind": "algorithm", "algorithm_id": "xgboost"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["method_kind"] == "algorithm"
        assert data["algorithm_id"] == "xgboost"

    def test_update_algorithm_to_rule_clears_algorithm_id(self, client):
        r = client.post("/api/v1/quant-research/research", json={"name": "R"})
        research_id = r.json()["id"]
        e = client.post(
            f"/api/v1/quant-research/research/{research_id}/experiments",
            json={"name": "E", "method_kind": "algorithm", "algorithm_id": "lightgbm"},
        )
        experiment_id = e.json()["id"]

        response = client.patch(
            f"/api/v1/quant-research/experiments/{experiment_id}",
            json={"method_kind": "rule", "algorithm_id": None},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["method_kind"] == "rule"
        assert data["algorithm_id"] is None

    def test_create_run_from_draft_rejected(self, client):
        r = client.post("/api/v1/quant-research/research", json={"name": "R"})
        research_id = r.json()["id"]

        e = client.post(
            f"/api/v1/quant-research/research/{research_id}/experiments",
            json={"name": "E", "method_kind": "rule"},
        )
        assert e.status_code == 201, e.json()
        experiment_id = e.json()["id"]

        response = client.post(
            f"/api/v1/quant-research/experiments/{experiment_id}/runs",
            json={"run_label": "Draft Run"},
        )
        assert response.status_code == 409

    def test_tampering_detected_on_get_experiment(self, client):
        r = client.post("/api/v1/quant-research/research", json={"name": "R"})
        research_id = r.json()["id"]

        e = client.post(
            f"/api/v1/quant-research/research/{research_id}/experiments",
            json={"name": "E", "algorithm_id": "lightgbm", "method_kind": "algorithm"},
        )
        experiment_id = e.json()["id"]

        client.patch(
            f"/api/v1/quant-research/experiments/{experiment_id}",
            json={"status": "submitted"},
        )

        repo = client.app.state._test_repo
        conn = repo._get_connection()
        conn.execute(
            "UPDATE experiment SET algorithm_id = ? WHERE id = ?",
            ("xgboost", experiment_id),
        )
        conn.commit()

        response = client.get(f"/api/v1/quant-research/experiments/{experiment_id}")
        assert response.status_code == 409

    def test_tampering_detected_on_list_experiments(self, client):
        r = client.post("/api/v1/quant-research/research", json={"name": "R"})
        research_id = r.json()["id"]

        e = client.post(
            f"/api/v1/quant-research/research/{research_id}/experiments",
            json={"name": "E", "algorithm_id": "lightgbm", "method_kind": "algorithm"},
        )
        experiment_id = e.json()["id"]

        client.patch(
            f"/api/v1/quant-research/experiments/{experiment_id}",
            json={"status": "submitted"},
        )

        repo = client.app.state._test_repo
        conn = repo._get_connection()
        conn.execute(
            "UPDATE experiment SET algorithm_id = ? WHERE id = ?",
            ("xgboost", experiment_id),
        )
        conn.commit()

        response = client.get(f"/api/v1/quant-research/research/{research_id}/experiments")
        assert response.status_code == 409

    def test_tampering_detected_on_get_research(self, client):
        r = client.post("/api/v1/quant-research/research", json={"name": "R"})
        research_id = r.json()["id"]

        e = client.post(
            f"/api/v1/quant-research/research/{research_id}/experiments",
            json={"name": "E", "algorithm_id": "lightgbm", "method_kind": "algorithm"},
        )
        experiment_id = e.json()["id"]

        client.patch(
            f"/api/v1/quant-research/experiments/{experiment_id}",
            json={"status": "submitted"},
        )

        repo = client.app.state._test_repo
        conn = repo._get_connection()
        conn.execute(
            "UPDATE experiment SET algorithm_id = ? WHERE id = ?",
            ("xgboost", experiment_id),
        )
        conn.commit()

        response = client.get(f"/api/v1/quant-research/research/{research_id}")
        assert response.status_code == 409

    def test_status_only_freeze_rejects_algorithm_without_id(self, client):
        r = client.post("/api/v1/quant-research/research", json={"name": "R"})
        research_id = r.json()["id"]

        repo = client.app.state._test_repo
        conn = repo._get_connection()
        conn.execute(
            """INSERT INTO experiment (id, research_id, name, description, experiment_type, method_kind, algorithm_id, protocol_json, protocol_hash, status, created_at, updated_at)
               VALUES (?, ?, 'Old Draft', '', 'stock_selection', 'algorithm', NULL, '{}', NULL, 'draft', '2025-01-01', '2025-01-01')""",
            (str(uuid.uuid4()), research_id),
        )
        conn.commit()

        experiments = repo.list_experiments(research_id)
        old_draft = [e for e in experiments if e["name"] == "Old Draft"][0]

        response = client.patch(
            f"/api/v1/quant-research/experiments/{old_draft['id']}",
            json={"status": "submitted"},
        )
        assert response.status_code == 400
        assert "algorithm_id is required" in response.json()["detail"]

    def test_status_only_freeze_rejects_rule_with_algorithm_id(self, client):
        r = client.post("/api/v1/quant-research/research", json={"name": "R"})
        research_id = r.json()["id"]

        repo = client.app.state._test_repo
        conn = repo._get_connection()
        conn.execute(
            """INSERT INTO experiment (id, research_id, name, description, experiment_type, method_kind, algorithm_id, protocol_json, protocol_hash, status, created_at, updated_at)
               VALUES (?, ?, 'Old Rule Draft', '', 'stock_selection', 'rule', 'lightgbm', '{}', NULL, 'draft', '2025-01-01', '2025-01-01')""",
            (str(uuid.uuid4()), research_id),
        )
        conn.commit()

        experiments = repo.list_experiments(research_id)
        old_draft = [e for e in experiments if e["name"] == "Old Rule Draft"][0]

        response = client.patch(
            f"/api/v1/quant-research/experiments/{old_draft['id']}",
            json={"status": "submitted"},
        )
        assert response.status_code == 400
        assert "must be empty" in response.json()["detail"]

    def test_semantic_invalid_frozen_rejected_on_get(self, client):
        r = client.post("/api/v1/quant-research/research", json={"name": "R"})
        research_id = r.json()["id"]

        repo = client.app.state._test_repo
        conn = repo._get_connection()
        exp_id = str(uuid.uuid4())
        from src.repositories.quant_research_repo import _compute_research_definition_hash
        h = _compute_research_definition_hash("stock_selection", "algorithm", None, {})
        conn.execute(
            """INSERT INTO experiment (id, research_id, name, description, experiment_type, method_kind, algorithm_id, protocol_json, protocol_hash, status, created_at, updated_at)
               VALUES (?, ?, 'Bad Frozen', '', 'stock_selection', 'algorithm', NULL, '{}', ?, 'submitted', '2025-01-01', '2025-01-01')""",
            (exp_id, research_id, h),
        )
        conn.commit()

        response = client.get(f"/api/v1/quant-research/experiments/{exp_id}")
        assert response.status_code == 409
        assert "algorithm_id is required" in response.json()["detail"]

    def test_semantic_invalid_frozen_rejected_on_list(self, client):
        r = client.post("/api/v1/quant-research/research", json={"name": "R"})
        research_id = r.json()["id"]

        repo = client.app.state._test_repo
        conn = repo._get_connection()
        exp_id = str(uuid.uuid4())
        from src.repositories.quant_research_repo import _compute_research_definition_hash
        h = _compute_research_definition_hash("stock_selection", "rule", "lightgbm", {})
        conn.execute(
            """INSERT INTO experiment (id, research_id, name, description, experiment_type, method_kind, algorithm_id, protocol_json, protocol_hash, status, created_at, updated_at)
               VALUES (?, ?, 'Bad Rule Frozen', '', 'stock_selection', 'rule', 'lightgbm', '{}', ?, 'submitted', '2025-01-01', '2025-01-01')""",
            (exp_id, research_id, h),
        )
        conn.commit()

        response = client.get(f"/api/v1/quant-research/research/{research_id}/experiments")
        assert response.status_code == 409
        assert "must be empty" in response.json()["detail"]

    def test_semantic_invalid_frozen_rejected_on_get_research(self, client):
        r = client.post("/api/v1/quant-research/research", json={"name": "R"})
        research_id = r.json()["id"]

        repo = client.app.state._test_repo
        conn = repo._get_connection()
        exp_id = str(uuid.uuid4())
        from src.repositories.quant_research_repo import _compute_research_definition_hash
        h = _compute_research_definition_hash("stock_selection", "algorithm", None, {})
        conn.execute(
            """INSERT INTO experiment (id, research_id, name, description, experiment_type, method_kind, algorithm_id, protocol_json, protocol_hash, status, created_at, updated_at)
               VALUES (?, ?, 'Bad Frozen', '', 'stock_selection', 'algorithm', NULL, '{}', ?, 'submitted', '2025-01-01', '2025-01-01')""",
            (exp_id, research_id, h),
        )
        conn.commit()

        response = client.get(f"/api/v1/quant-research/research/{research_id}")
        assert response.status_code == 409

    def test_semantic_invalid_frozen_rejected_on_create_run(self, client):
        r = client.post("/api/v1/quant-research/research", json={"name": "R"})
        research_id = r.json()["id"]

        repo = client.app.state._test_repo
        conn = repo._get_connection()
        exp_id = str(uuid.uuid4())
        from src.repositories.quant_research_repo import _compute_research_definition_hash
        h = _compute_research_definition_hash("stock_selection", "algorithm", None, {})
        conn.execute(
            """INSERT INTO experiment (id, research_id, name, description, experiment_type, method_kind, algorithm_id, protocol_json, protocol_hash, status, created_at, updated_at)
               VALUES (?, ?, 'Bad Frozen', '', 'stock_selection', 'algorithm', NULL, '{}', ?, 'submitted', '2025-01-01', '2025-01-01')""",
            (exp_id, research_id, h),
        )
        conn.commit()

        response = client.post(
            f"/api/v1/quant-research/experiments/{exp_id}/runs",
            json={"run_label": "Should Fail"},
        )
        assert response.status_code == 409
        assert "algorithm_id is required" in response.json()["detail"]
