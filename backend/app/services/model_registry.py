"""
Model registry service for managing uploaded and built-in models.

Responsibilities:
- Register models in SQLite
- Load models dynamically
- List available models
- Validate model files and metadata
"""

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from torchvision.models import resnet18, vgg16

from app.config import settings
from app.core.models.simple_cnn import SimpleCNN


SUPPORTED_ARCHITECTURES = {"simple_cnn", "resnet18", "vgg16"}
SUPPORTED_DATASETS = {"cifar10", "mnist"}


class ModelRegistry:
    """Manages model registration, loading, and listing."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.models_dir = Path(settings.results_dir) / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize models table if not exists."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                path TEXT NOT NULL,
                architecture TEXT NOT NULL,
                dataset TEXT NOT NULL,
                uploaded_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
        conn.close()

    def register_model(
        self, file_path: str, model_name: str, architecture: str, dataset: str
    ) -> str:
        """
        Register a model in the registry.

        Args:
            file_path: Path to uploaded model file
            model_name: User-friendly model name
            architecture: Model architecture (e.g., 'simple_cnn', 'resnet18', 'vgg16')
            dataset: Dataset model was trained on (e.g., 'cifar10', 'mnist')

        Returns:
            Model ID (UUID)

        Raises:
            ValueError: If model name already exists or validation fails
        """
        # Validate inputs
        if not model_name.strip():
            raise ValueError("model_name cannot be empty")
        if architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(f"Unsupported architecture: {architecture}")
        if dataset not in SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset: {dataset}")

        # Validate model file is loadable
        try:
            _validate_model_file(file_path, architecture)
        except Exception as e:
            raise ValueError(f"Model validation failed: {str(e)}")

        # Generate model ID and destination path
        model_id = str(uuid.uuid4())
        dest_path = self.models_dir / f"{model_id}.pt"

        # Move file to models directory
        import shutil

        shutil.move(file_path, str(dest_path))

        # Register in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO models (id, name, path, architecture, dataset, uploaded_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    model_id,
                    model_name,
                    str(dest_path),
                    architecture,
                    dataset,
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            # Clean up file if registration fails
            dest_path.unlink(missing_ok=True)
            raise ValueError(f"Model name '{model_name}' already exists")
        finally:
            conn.close()

        return model_id

    def get_model(self, model_id: str) -> Optional[dict]:
        """
        Get model metadata by ID.

        Returns:
            Dict with id, name, path, architecture, dataset, uploaded_at
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, name, path, architecture, dataset, uploaded_at FROM models WHERE id = ?",
            (model_id,),
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            return dict(row)
        return None

    def list_models(self, limit: int = 50) -> list[dict]:
        """
        List all registered models.

        Returns:
            List of model metadata dicts, newest first
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, name, path, architecture, dataset, uploaded_at
            FROM models
            ORDER BY uploaded_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def load_model(self, model_id: str, device: str = "cpu"):
        """
        Load a registered model.

        Args:
            model_id: Model UUID
            device: torch device (cpu, cuda, etc.)

        Returns:
            PyTorch model on specified device

        Raises:
            ValueError: If model not found or load fails
        """
        metadata = self.get_model(model_id)
        if not metadata:
            raise ValueError(f"Model {model_id} not found")

        try:
            model_path = metadata["path"]
            architecture = metadata["architecture"]
            dataset = metadata["dataset"]
            num_classes = _get_num_classes(dataset)
            in_channels = _get_in_channels(dataset)

            # Instantiate model based on architecture
            model = _instantiate_model(
                architecture=architecture,
                num_classes=num_classes,
                in_channels=in_channels,
            )

            # Load weights
            loaded = torch.load(model_path, map_location=device)
            state_dict = loaded["state_dict"] if isinstance(loaded, dict) and "state_dict" in loaded else loaded
            model.load_state_dict(state_dict)
            model = model.to(device)
            model.eval()

            return model
        except Exception as e:
            raise ValueError(f"Failed to load model {model_id}: {str(e)}")

    def delete_model(self, model_id: str) -> bool:
        """
        Delete a registered model.

        Returns:
            True if deleted, False if not found
        """
        metadata = self.get_model(model_id)
        if not metadata:
            return False

        # Delete file
        Path(metadata["path"]).unlink(missing_ok=True)

        # Delete from database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM models WHERE id = ?", (model_id,))
        conn.commit()
        conn.close()

        return True


def _validate_model_file(file_path: str, architecture: str) -> None:
    """Validate that model file is a valid PyTorch model."""
    try:
        loaded = torch.load(file_path, map_location="cpu")
        state_dict = loaded["state_dict"] if isinstance(loaded, dict) and "state_dict" in loaded else loaded
        if not isinstance(state_dict, dict):
            raise ValueError("Model file must contain a state_dict (torch.load should return dict)")
    except Exception as e:
        raise ValueError(f"Invalid PyTorch model: {str(e)}")


def _get_num_classes(dataset: str) -> int:
    if dataset in SUPPORTED_DATASETS:
        return 10
    raise ValueError(f"Unsupported dataset: {dataset}")


def _get_in_channels(dataset: str) -> int:
    if dataset == "mnist":
        return 1
    if dataset == "cifar10":
        return 3
    raise ValueError(f"Unsupported dataset: {dataset}")


def _instantiate_model(architecture: str, num_classes: int, in_channels: int):
    if architecture == "simple_cnn":
        return SimpleCNN(in_channels=in_channels, num_classes=num_classes)
    if architecture == "resnet18":
        return resnet18(weights=None, num_classes=num_classes)
    if architecture == "vgg16":
        return vgg16(weights=None, num_classes=num_classes)
    raise ValueError(f"Unknown architecture: {architecture}")


# Global registry instance
_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """Get or create global model registry instance."""
    global _registry
    if _registry is None:
        db_path = str(Path(settings.results_dir) / "a3dl.db")
        _registry = ModelRegistry(db_path)
    return _registry
