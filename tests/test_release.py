"""Standalone deployment checks: python -m unittest discover -s tests -v."""
import json
import unittest
from pathlib import Path

from server.engines.gnn import GnnEngine
from server.schemas.engines import EngineStatus
from server.services.normalize import normalize


class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = GnnEngine()
        cls.fixture = json.loads((Path(__file__).parent / "v14_parity.json").read_text())

    def test_saved_prediction_parity_and_confidence(self):
        self.assertEqual(self.engine.version, "v14.0")
        self.assertTrue(self.engine.status()["ready"])
        for prop in ("mp", "bp"):
            rows = [row for row in self.fixture["predictions"] if row["property"] == prop]
            results = self.engine.predict_batch([normalize(row["smiles"]) for row in rows], prop)
            for row, result in zip(rows, results, strict=True):
                self.assertEqual(result.status, EngineStatus.LOW_CONFIDENCE)
                self.assertAlmostEqual(result.value, row["value"], delta=0.001)
                self.assertIsNone(result.uncertainty)
                self.assertEqual(result.raw["selection_lock_sha256"], self.fixture["selection_lock_sha256"])
                self.assertEqual(sum(map(len, result.raw["member_predictions"].values())), 9)

    def test_api_version_and_frontend(self):
        from app import create_app, get_registry
        # Use the same resident ensemble for the API checks.
        registry = get_registry()
        registry.by_name["gnn"] = self.engine
        client = create_app().test_client()
        self.assertEqual(client.get("/version").json["gnn"], "v14.0")
        self.assertIsNotNone(client.get("/version").json["gnn_checkpoint_digest"])
        self.assertIsNotNone(client.get("/version").json["gnn_dataset_manifest_digest"])
        page = client.get("/")
        self.assertEqual(page.status_code, 200)
        page.close()
        engine = next(item for item in client.get("/engines").json["engines"] if item["name"] == "gnn")
        self.assertEqual(engine["properties"]["mp"]["members"], 9)
        row = self.fixture["predictions"][0]
        response = client.post("/predict", json={
            "smiles": row["smiles"], "properties": [row["property"]], "engines": ["gnn"],
        })
        self.assertEqual(response.status_code, 200)
        prediction = response.json["predictions"][0]
        self.assertEqual(prediction["method"], "gnn")
        self.assertEqual(prediction["confidence"], "low")
        self.assertAlmostEqual(prediction["value"], row["value"], delta=0.001)
        self.assertIsNone(prediction["uncertainty"])

    def test_domain_failure(self):
        blend = self.engine._blend
        old_threshold = blend._manifest["ood_threshold"]
        try:
            blend._manifest["ood_threshold"] = 1.1
            result = self.engine.predict(normalize("CCO"), "mp")
        finally:
            blend._manifest["ood_threshold"] = old_threshold
        self.assertEqual(result.status, EngineStatus.OUT_OF_DOMAIN)
        self.assertIsNone(result.value)


if __name__ == "__main__":
    unittest.main()
