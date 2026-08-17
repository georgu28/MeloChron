"""Request and response models for the serving API.

Validation lives here rather than in the handlers so that the bounds are
declared in one place and appear in the generated OpenAPI schema. The caps are
not ceremony: ``k`` and inline history length are the two fields an
unauthenticated caller could use to make the service do unbounded work, so both
carry hard limits rather than being trusted.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

#: Inline histories are for trying the API without uploading a file. The model
#: reads at most ``max_len`` (200) events, so this is already far more than
#: scoring uses; it exists to bound request size, not to bound the model.
MAX_INLINE_HISTORY = 5_000
MAX_K = 100
MAX_BATCH = 32


class Play(BaseModel):
    """One listening event, as a client supplies it."""

    artist: str = Field(min_length=1, max_length=512)
    track: str = Field(min_length=1, max_length=512)
    ts: int = Field(gt=0, description="Unix timestamp in seconds, UTC")


class RecommendRequest(BaseModel):
    """Either a parsed upload (``job_id``) or an inline history, never both."""

    model_config = ConfigDict(protected_namespaces=())

    job_id: str | None = None
    history: list[Play] | None = Field(default=None, max_length=MAX_INLINE_HISTORY)
    k: int = Field(default=20, ge=1, le=MAX_K)
    #: Off by default, matching the evaluation protocol: replaying a known
    #: track is the most common real outcome, so filtering it would make the
    #: product disagree with the metrics. On is the discovery surface.
    exclude_history: bool = False
    model: str | None = Field(default=None, description="Which loaded variant to score with")


class BatchRecommendRequest(BaseModel):
    """Several histories scored in one forward pass.

    Exists because ``SASRecScorer.score`` is already batched: sending histories
    together amortises the item-vector materialisation, which for a projected
    text representation is the dominant cost of a single request.
    """

    model_config = ConfigDict(protected_namespaces=())

    histories: list[list[Play]] = Field(min_length=1, max_length=MAX_BATCH)
    k: int = Field(default=20, ge=1, le=MAX_K)
    exclude_history: bool = False
    model: str | None = None


class RecommendationOut(BaseModel):
    item_id: int
    key: str
    artist: str
    track: str
    score: float
    #: True when this track already appears in the submitted history. The
    #: repeat/novel split is the project's headline honesty result, so the
    #: product surface reports it per item rather than only in the README.
    repeat: bool


class Coverage(BaseModel):
    """How much of the upload the model actually recognised."""

    coverage: float = Field(description="In-vocabulary fraction of the history")
    history_length: int
    matched: int
    cold_start: bool = Field(
        description="Coverage fell below the threshold; results are weakly grounded"
    )


class ModelCard(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    name: str
    version: str
    variant: str
    trained: bool
    catalog_size: int
    max_len: int
    d_model: int | None = None
    use_time: bool = True
    metrics: dict = Field(default_factory=dict)
    load_seconds: float | None = None


class RecommendResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    items: list[RecommendationOut]
    coverage: Coverage
    model: ModelCard
    inference_ms: float


class BatchRecommendResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    results: list[RecommendResponse]
    inference_ms: float


class UploadAccepted(BaseModel):
    job_id: str
    state: str
    filename: str
    size_bytes: int


class HealthResponse(BaseModel):
    status: str
    models_loaded: int
    active_model: str | None = None
    error: str | None = None
