from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


AttackType = Literal["fgsm", "pgd"]
DatasetName = Literal["cifar10", "mnist"]
ModelName = Literal["simple_cnn"]


class AttackRequest(BaseModel):
    attack_type: AttackType = Field(description="Attack algorithm to run")
    dataset: DatasetName = Field(default="cifar10")
    model_name: ModelName = Field(default="simple_cnn")
    epsilon: float = Field(default=8.0 / 255.0, gt=0.0, le=1.0)
    alpha: float = Field(default=2.0 / 255.0, gt=0.0, le=1.0)
    steps: int = Field(default=10, ge=1, le=200)
    sample_limit: int = Field(default=64, ge=1, le=2048)
    batch_size: int = Field(default=32, ge=1, le=256)
    checkpoint_path: str | None = None
    model_id: str | None = Field(default=None, description="UUID of uploaded model (takes precedence over checkpoint_path)")


class PredictionInfo(BaseModel):
    label: int
    confidence: float


class AttackSampleResult(BaseModel):
    sample_index: int
    true_label: int
    original: PredictionInfo
    adversarial: PredictionInfo
    attack_success: bool
    original_image_path: str
    adversarial_image_path: str
    perturbation_image_path: str


class AttackMetrics(BaseModel):
    clean_accuracy: float
    adversarial_accuracy: float
    attack_success_rate: float
    avg_confidence_drop: float


class AttackResponse(BaseModel):
    run_id: str
    attack_type: AttackType
    dataset: DatasetName
    model_name: ModelName
    epsilon: float
    alpha: float
    steps: int
    metrics: AttackMetrics
    samples: list[AttackSampleResult]


class DetectRequest(BaseModel):
    run_id: str
    confidence_drop_threshold: float = Field(default=0.15, ge=0.0, le=1.0)


class DetectResult(BaseModel):
    sample_index: int
    detection_probability: float
    label: Literal["clean", "adversarial"]


class DetectResponse(BaseModel):
    run_id: str
    threshold: float
    results: list[DetectResult]


class DefendRequest(BaseModel):
    run_id: str
    dataset: DatasetName = Field(default="cifar10")
    gaussian_sigma: float = Field(default=0.03, ge=0.0, le=1.0)
    bit_depth_bits: int = Field(default=4, ge=1, le=8)
    model_id: str | None = Field(default=None, description="UUID of model to use (optional)")


class DefendResponse(BaseModel):
    run_id: str
    dataset: DatasetName
    gaussian_sigma: float
    bit_depth_bits: int
    adversarial_accuracy: float
    defended_accuracy: float
    robustness_score: float


class ReportRequest(BaseModel):
    run_id: str
    include_detection: bool = True
    include_defense: bool = True


class ReportResponse(BaseModel):
    report_id: str
    generated_at: str
    model_details: dict
    attack: dict
    detection: dict | None = None
    defense: dict | None = None
    vulnerability_explanation: str
    agent_recommendation: dict
    summary: str


class RecommendationRequest(BaseModel):
    attack_success_rate: float = Field(ge=0.0, le=1.0)


class RecommendationResponse(BaseModel):
    recommended_attack: Literal["fgsm", "pgd"]
    recommended_epsilon: float
    recommended_defense: str
    reason: str


class TrainRequest(BaseModel):
    dataset: DatasetName = Field(default="cifar10")
    epochs: int = Field(default=1, ge=1, le=50)
    batch_size: int = Field(default=64, ge=1, le=512)
    learning_rate: float = Field(default=1e-3, gt=0.0, le=1.0)
    max_batches_per_epoch: int = Field(default=100, ge=1, le=2000)
    model_id: str | None = Field(default=None, description="UUID of model to fine-tune (optional)")


class TrainResponse(BaseModel):
    checkpoint_id: str
    checkpoint_path: str
    dataset: DatasetName
    model_name: ModelName
    epochs: int
    batch_size: int
    learning_rate: float
    epoch_losses: list[float]


class ExperimentSummary(BaseModel):
    run_id: str
    attack_type: AttackType
    dataset: DatasetName
    model_name: ModelName
    epsilon: float
    alpha: float
    steps: int
    clean_accuracy: float
    adversarial_accuracy: float
    attack_success_rate: float
    created_at: str


class ExperimentListResponse(BaseModel):
    experiments: list[ExperimentSummary]


class CheckpointSummary(BaseModel):
    checkpoint_id: str
    path: str
    dataset: DatasetName
    model_name: ModelName
    created_at: str
    size_mb: float


class CheckpointListResponse(BaseModel):
    checkpoints: list[CheckpointSummary]


class JobStatusEnum(str, Enum):
    """Job execution status"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobTypeEnum(str, Enum):
    """Job type"""

    TRAIN = "train"
    ATTACK = "attack"
    DEFEND = "defend"


class JobResponse(BaseModel):
    """Response when job is submitted"""

    job_id: str
    status: JobStatusEnum


class JobStatusResponse(BaseModel):
    """Response with job status and progress"""

    job_id: str
    job_type: JobTypeEnum
    status: JobStatusEnum
    progress: int = Field(ge=0, le=100)
    error: str | None = None


class JobResultWrapper(BaseModel):
    """Response with job result"""

    job_id: str | None = None
    status: JobStatusEnum | None = None
    result: dict | None = None
    error: str | None = None
    code: int | None = None
    message: str | None = None


# Model Registry Schemas


class ModelUploadResponse(BaseModel):
    """Response when model is uploaded"""

    model_id: str
    name: str
    architecture: str
    dataset: DatasetName
    path: str
    uploaded_at: str


class ModelMetadata(BaseModel):
    """Model metadata"""

    id: str
    name: str
    path: str
    architecture: str
    dataset: DatasetName
    uploaded_at: str


class ModelListResponse(BaseModel):
    """List of models"""

    models: list[ModelMetadata]

