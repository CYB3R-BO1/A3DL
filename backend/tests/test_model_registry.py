"""
Unit tests for the model registry service.

Tests:
- Successful model registration
- Duplicate name prevention
- Model loading with weights verification
- Model listing and pagination
- Model deletion with file cleanup
- Invalid file handling
- Invalid architecture/dataset validation
"""

import sqlite3
import tempfile
import os
from pathlib import Path
from typing import Generator

import pytest
import torch
import torch.nn as nn
from torchvision.models import resnet18, vgg16

from app.services.model_registry import ModelRegistry, _validate_model_file
from app.core.models.simple_cnn import SimpleCNN


def _temp_pt_path() -> str:
    """Create a temporary .pt path that can be reopened on Windows."""
    fd, path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    return path


# Test fixtures

@pytest.fixture
def temp_db_path() -> Generator[str, None, None]:
    """Create a temporary database file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def temp_models_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for storing models."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def registry(temp_db_path, temp_models_dir, monkeypatch) -> Generator[ModelRegistry, None, None]:
    """Create a model registry instance with temporary storage."""
    # Monkey-patch settings to use temp directories
    import app.config
    monkeypatch.setattr(app.config.settings, "results_dir", str(temp_models_dir.parent))
    
    # Create temp models subdirectory
    models_dir = temp_models_dir.parent / "models"
    models_dir.mkdir(exist_ok=True)
    (models_dir).mkdir(parents=True, exist_ok=True)
    
    registry = ModelRegistry(temp_db_path)
    registry.models_dir = models_dir
    
    yield registry


@pytest.fixture
def valid_model_file(temp_models_dir) -> Generator[str, None, None]:
    """Create a valid PyTorch model file for testing."""
    model = SimpleCNN(in_channels=3, num_classes=10)
    state_dict = model.state_dict()
    
    model_path = temp_models_dir / "test_model.pt"
    torch.save(state_dict, model_path)
    
    yield str(model_path)
    
    # Cleanup
    model_path.unlink(missing_ok=True)


@pytest.fixture
def invalid_model_file(temp_models_dir) -> Generator[str, None, None]:
    """Create an invalid model file (not a proper state_dict)."""
    invalid_path = temp_models_dir / "invalid_model.pt"
    torch.save([1, 2, 3], invalid_path)  # List instead of dict
    
    yield str(invalid_path)
    
    # Cleanup
    invalid_path.unlink(missing_ok=True)


@pytest.fixture
def valid_resnet18_model_file(temp_models_dir) -> Generator[str, None, None]:
    """Create a valid ResNet18 model file for testing."""
    model = resnet18(weights=None, num_classes=10)
    model_path = temp_models_dir / "resnet18_model.pt"
    torch.save(model.state_dict(), model_path)

    yield str(model_path)

    model_path.unlink(missing_ok=True)


@pytest.fixture
def valid_vgg16_model_file(temp_models_dir) -> Generator[str, None, None]:
    """Create a valid VGG16 model file for testing."""
    model = vgg16(weights=None, num_classes=10)
    model_path = temp_models_dir / "vgg16_model.pt"
    torch.save(model.state_dict(), model_path)

    yield str(model_path)

    model_path.unlink(missing_ok=True)


# Tests: Registration

def test_register_model_success(registry, valid_model_file):
    """Test successful model registration."""
    model_id = registry.register_model(
        file_path=valid_model_file,
        model_name="test_model",
        architecture="simple_cnn",
        dataset="cifar10",
    )
    
    assert model_id is not None
    assert len(model_id) > 0
    
    # Verify in database
    metadata = registry.get_model(model_id)
    assert metadata is not None
    assert metadata["name"] == "test_model"
    assert metadata["architecture"] == "simple_cnn"
    assert metadata["dataset"] == "cifar10"


def test_register_model_duplicate_name_raises_error(registry, valid_model_file):
    """Test that duplicate model names are rejected."""
    # Register first model
    registry.register_model(
        file_path=valid_model_file,
        model_name="duplicate_test",
        architecture="simple_cnn",
        dataset="cifar10",
    )
    
    # Try to register with same name (should fail)
    tmp_path = _temp_pt_path()
    try:
        torch.save(torch.nn.Linear(5, 5).state_dict(), tmp_path)

        with pytest.raises(ValueError, match="already exists"):
            registry.register_model(
                file_path=tmp_path,
                model_name="duplicate_test",
                architecture="simple_cnn",
                dataset="cifar10",
            )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_register_model_empty_name_raises_error(registry, valid_model_file):
    """Test that empty model names are rejected."""
    with pytest.raises(ValueError, match="cannot be empty"):
        registry.register_model(
            file_path=valid_model_file,
            model_name="",
            architecture="simple_cnn",
            dataset="cifar10",
        )


def test_register_model_invalid_architecture_raises_error(registry, valid_model_file):
    """Test that unsupported architectures are rejected."""
    with pytest.raises(ValueError, match="Unsupported architecture"):
        registry.register_model(
            file_path=valid_model_file,
            model_name="test_model",
            architecture="unsupported_arch",
            dataset="cifar10",
        )


def test_register_model_invalid_dataset_raises_error(registry, valid_model_file):
    """Test that unsupported datasets are rejected."""
    with pytest.raises(ValueError, match="Unsupported dataset"):
        registry.register_model(
            file_path=valid_model_file,
            model_name="test_model",
            architecture="simple_cnn",
            dataset="imagenet",
        )


def test_register_model_invalid_file_raises_error(registry, invalid_model_file):
    """Test that invalid PyTorch files are rejected."""
    with pytest.raises(ValueError, match="Model validation failed"):
        registry.register_model(
            file_path=invalid_model_file,
            model_name="test_model",
            architecture="simple_cnn",
            dataset="cifar10",
        )


# Tests: Retrieval

def test_get_model_returns_metadata(registry, valid_model_file):
    """Test retrieving model metadata by ID."""
    model_id = registry.register_model(
        file_path=valid_model_file,
        model_name="retrieve_test",
        architecture="simple_cnn",
        dataset="mnist",
    )
    
    metadata = registry.get_model(model_id)
    assert metadata is not None
    assert metadata["id"] == model_id
    assert metadata["name"] == "retrieve_test"
    assert metadata["architecture"] == "simple_cnn"
    assert metadata["dataset"] == "mnist"
    assert "uploaded_at" in metadata
    assert "path" in metadata


def test_get_model_nonexistent_returns_none(registry):
    """Test that retrieving nonexistent model returns None."""
    metadata = registry.get_model("nonexistent_id")
    assert metadata is None


def test_list_models_returns_models(registry, valid_model_file):
    """Test listing models returns all registered models."""
    # Register multiple models
    id1 = registry.register_model(
        file_path=valid_model_file,
        model_name="model_1",
        architecture="simple_cnn",
        dataset="cifar10",
    )
    
    tmp_path = _temp_pt_path()
    try:
        torch.save(torch.nn.Linear(5, 5).state_dict(), tmp_path)
        id2 = registry.register_model(
            file_path=tmp_path,
            model_name="model_2",
            architecture="simple_cnn",
            dataset="cifar10",
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    
    models = registry.list_models(limit=50)
    assert len(models) == 2
    model_ids = [m["id"] for m in models]
    assert id1 in model_ids
    assert id2 in model_ids


def test_list_models_respects_limit(registry, valid_model_file):
    """Test that list_models respects the limit parameter."""
    # Register multiple models
    for i in range(5):
        tmp_path = _temp_pt_path()
        try:
            torch.save(torch.nn.Linear(5, 5).state_dict(), tmp_path)
            registry.register_model(
                file_path=tmp_path,
                model_name=f"model_{i}",
                architecture="simple_cnn",
                dataset="cifar10",
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    models = registry.list_models(limit=3)
    assert len(models) == 3


def test_list_models_ordered_by_newest(registry, valid_model_file):
    """Test that models are returned newest first."""
    import time
    
    ids = []
    for i in range(3):
        tmp_path = _temp_pt_path()
        try:
            torch.save(torch.nn.Linear(5, 5).state_dict(), tmp_path)
            model_id = registry.register_model(
                file_path=tmp_path,
                model_name=f"model_{i}",
                architecture="simple_cnn",
                dataset="cifar10",
            )
            ids.append(model_id)
            time.sleep(0.01)  # Ensure different timestamps
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    models = registry.list_models(limit=50)
    returned_ids = [m["id"] for m in models]
    
    # Newest should be last registered (reverse order)
    assert returned_ids[0] == ids[-1]  # Most recent
    assert returned_ids[-1] == ids[0]   # Oldest


# Tests: Loading

def test_load_model_returns_torch_model(registry, valid_model_file):
    """Test that loading a model returns a valid PyTorch model."""
    model_id = registry.register_model(
        file_path=valid_model_file,
        model_name="load_test",
        architecture="simple_cnn",
        dataset="cifar10",
    )
    
    model = registry.load_model(model_id, device="cpu")
    
    assert model is not None
    assert isinstance(model, nn.Module)
    assert model.training == False  # Should be in eval mode


def test_load_model_nonexistent_raises_error(registry):
    """Test that loading nonexistent model raises ValueError."""
    with pytest.raises(ValueError, match="not found"):
        registry.load_model("nonexistent_id", device="cpu")


def test_load_model_maps_to_correct_device(registry, valid_model_file):
    """Test that loaded model is on correct device."""
    model_id = registry.register_model(
        file_path=valid_model_file,
        model_name="device_test",
        architecture="simple_cnn",
        dataset="cifar10",
    )
    
    model = registry.load_model(model_id, device="cpu")
    
    # Check that model parameters are on CPU
    for param in model.parameters():
        assert param.device.type == "cpu"


def test_load_resnet18_model_returns_torch_model(registry, valid_resnet18_model_file):
    """Test that loading a resnet18 model returns a valid PyTorch model."""
    model_id = registry.register_model(
        file_path=valid_resnet18_model_file,
        model_name="resnet18_load_test",
        architecture="resnet18",
        dataset="cifar10",
    )

    model = registry.load_model(model_id, device="cpu")

    assert model is not None
    assert isinstance(model, nn.Module)
    assert model.training is False


def test_load_vgg16_model_returns_torch_model(registry, valid_vgg16_model_file):
    """Test that loading a vgg16 model returns a valid PyTorch model."""
    model_id = registry.register_model(
        file_path=valid_vgg16_model_file,
        model_name="vgg16_load_test",
        architecture="vgg16",
        dataset="cifar10",
    )

    model = registry.load_model(model_id, device="cpu")

    assert model is not None
    assert isinstance(model, nn.Module)
    assert model.training is False


# Tests: Deletion

def test_delete_model_removes_file(registry, valid_model_file):
    """Test that deleting a model removes the file."""
    model_id = registry.register_model(
        file_path=valid_model_file,
        model_name="delete_test",
        architecture="simple_cnn",
        dataset="cifar10",
    )
    
    metadata = registry.get_model(model_id)
    file_path = Path(metadata["path"])
    assert file_path.exists()
    
    # Delete model
    deleted = registry.delete_model(model_id)
    assert deleted == True
    assert not file_path.exists()


def test_delete_model_removes_from_database(registry, valid_model_file):
    """Test that deleting a model removes it from the database."""
    model_id = registry.register_model(
        file_path=valid_model_file,
        model_name="delete_db_test",
        architecture="simple_cnn",
        dataset="cifar10",
    )
    
    # Verify exists
    metadata = registry.get_model(model_id)
    assert metadata is not None
    
    # Delete
    deleted = registry.delete_model(model_id)
    assert deleted == True
    
    # Verify no longer exists
    metadata = registry.get_model(model_id)
    assert metadata is None


def test_delete_nonexistent_model_returns_false(registry):
    """Test that deleting nonexistent model returns False."""
    deleted = registry.delete_model("nonexistent_id")
    assert deleted == False


# Tests: File Validation

def test_validate_model_file_with_valid_state_dict():
    """Test validation of valid PyTorch state_dict."""
    tmp_path = _temp_pt_path()
    try:
        state_dict = {"layer1.weight": torch.randn(10, 5)}
        torch.save(state_dict, tmp_path)

        # Should not raise
        _validate_model_file(tmp_path, "simple_cnn")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_validate_model_file_with_invalid_format():
    """Test validation rejects non-dict state_dicts."""
    tmp_path = _temp_pt_path()
    try:
        # Save a list instead of dict
        torch.save([1, 2, 3], tmp_path)

        with pytest.raises(ValueError, match="state_dict"):
            _validate_model_file(tmp_path, "simple_cnn")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_validate_model_file_corrupted_raises_error():
    """Test validation fails on corrupted files."""
    with tempfile.NamedTemporaryFile(suffix=".pt", mode="w") as f:
        f.write("this is not a valid pytorch file")
        f.flush()
        
        with pytest.raises(ValueError, match="Invalid PyTorch model"):
            _validate_model_file(f.name, "simple_cnn")


# Tests: Database Integrity

def test_database_table_created_on_init(temp_db_path, temp_models_dir):
    """Test that registry creates models table on initialization."""
    registry = ModelRegistry(temp_db_path)
    registry.models_dir = temp_models_dir
    
    # Check table exists
    conn = sqlite3.connect(temp_db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='models'"
    )
    result = cursor.fetchone()
    conn.close()
    
    assert result is not None


def test_database_schema_correct(temp_db_path, temp_models_dir):
    """Test that registry creates correct database schema."""
    registry = ModelRegistry(temp_db_path)
    registry.models_dir = temp_models_dir
    
    conn = sqlite3.connect(temp_db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(models)")
    columns = {row[1]: row[2] for row in cursor.fetchall()}
    conn.close()
    
    expected_columns = {"id", "name", "path", "architecture", "dataset", "uploaded_at"}
    assert set(columns.keys()) == expected_columns


# Tests: Concurrency (Basic)

def test_concurrent_registrations_thread_safe(registry, temp_models_dir):
    """Test that concurrent registrations don't cause database errors."""
    import threading
    
    def register_model(index):
        tmp_path = _temp_pt_path()
        try:
            torch.save(torch.nn.Linear(5, 5).state_dict(), tmp_path)
            registry.register_model(
                file_path=tmp_path,
                model_name=f"concurrent_model_{index}",
                architecture="simple_cnn",
                dataset="cifar10",
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    threads = [threading.Thread(target=register_model, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    models = registry.list_models(limit=50)
    assert len(models) == 5
