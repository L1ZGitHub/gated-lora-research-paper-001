"""
Training configuration classes - Extended for Gated LoRA v2.

NOTE (Phase B import): copied as-is from ensicompute_harder_multitask.
Will be replaced by a thin YAML-loader in Phase G — keep imports stable.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    model_name: str = "microsoft/phi-2"
    model_type: str = "baseline"  # "baseline" or "gated"
    load_in_4bit: bool = False  # Disabled for gated LoRA (interferes with gating)
    load_in_8bit: bool = False  # Disabled for gated LoRA
    torch_dtype: str = "bfloat16"  # "float32", "float16", "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = True
    freeze_base: bool = True
    gradient_checkpointing: bool = False  # Can cause issues with hooks

    # LoRA config (baseline)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "dense"]
    )

    # Gated LoRA specific - Expert configuration
    expert_ranks: List[int] = field(default_factory=lambda: [8, 16, 32])
    expert_alphas: List[int] = field(default_factory=lambda: [16, 32, 64])

    # Gated LoRA specific - Gating configuration
    gating_hidden_dim: int = 256
    gating_dropout: float = 0.1
    per_layer_gating: bool = True  # True = separate gate per layer
    gating_temperature: float = 1.0  # Lower = sharper routing
    use_layer_embedding: bool = True  # Add layer index embedding to gating (NEW for ablation)
    gated_layers: Optional[List[int]] = None  # Only apply gating to these layers (None = all layers)

    # Gated LoRA specific - Routing configuration
    use_top_k: bool = False  # True = sparse routing
    top_k: int = 2  # Number of experts to use if use_top_k=True

    # Gated LoRA specific - Load balancing
    use_load_balancing: bool = True
    load_balancing_weight: float = 0.001  # Reduced from 0.01

    # Gated LoRA specific - L1 regularization on gates (encourages sparsity)
    use_l1_gate_regularization: bool = True
    l1_gate_weight: float = 0.01  # Weight for L1 penalty on gate weights


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Basic training
    num_epochs: int = 4
    batch_size: int = 4
    gradient_accumulation_steps: int = 8  # Effective batch = 32
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 0  # Will use warmup_ratio instead
    warmup_ratio: float = 0.1  # 10% of training for warmup
    max_steps: int = -1  # -1 means train for num_epochs

    # Optimization
    optimizer: str = "adamw"  # "adamw", "sgd", "adafactor"
    scheduler: str = "cosine"  # "linear", "cosine", "constant"
    max_grad_norm: float = 1.0  # gradient clipping
    fp16: bool = False
    bf16: bool = True

    # Data
    max_length: int = 512
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True

    # Checkpointing
    save_steps: int = 500
    save_total_limit: int = 3
    save_strategy: str = "steps"  # "steps", "epoch", "no"
    load_best_model_at_end: bool = True

    # Evaluation
    eval_steps: int = 500
    eval_strategy: str = "steps"  # "steps", "epoch", "no"
    eval_accumulation_steps: Optional[int] = None

    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0

    # Logging
    logging_steps: int = 10
    log_level: str = "info"
    log_routing_stats: bool = True  # Log gating statistics

    # Memory management
    vram_limit_gb: float = 44.0  # A40 has ~46GB
    oom_retry: bool = True
    min_batch_size: int = 1

    # Gating-specific training
    gating_warmup_steps: int = 0  # Steps to train gating only (0 = disabled)
    gating_warmup_epochs: int = 0  # Epochs to train gating only (takes priority over steps if > 0)
    freeze_experts_during_warmup: bool = True  # Freeze LoRA during gating warmup

    # Periodic routing analysis
    routing_analysis_steps: int = 500  # Run detailed routing analysis every N steps (0 = disabled)


@dataclass
class DataConfig:
    """Configuration for data."""

    # Dataset source
    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None
    dataset_path: Optional[str] = None
    train_file: Optional[str] = None
    val_file: Optional[str] = None
    test_file: Optional[str] = None

    # Multi-task configuration
    use_multi_task: bool = False
    task_datasets: List[str] = field(default_factory=list)  # ["squad", "imdb", "conll2003"]
    task_weights: List[float] = field(default_factory=list)  # [0.4, 0.3, 0.3]
    task_column: str = "task"  # Column name for task identifier

    # Text processing
    text_column: str = "text"
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    preprocessing_num_workers: int = 4

    # Data mixing
    shuffle_seed: int = 42


@dataclass
class WandbConfig:
    """Configuration for wandb logging."""

    enabled: bool = True
    project: str = "gated-lora-research"
    entity: Optional[str] = None
    name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    mode: str = "online"  # "online", "offline", "disabled"
    log_model: bool = False  # Whether to log model artifacts


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    # Sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    # Experiment metadata
    experiment_name: str = "gated-lora-exp"
    output_dir: str = "./outputs"
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)

    def save(self, path: str):
        """Save config to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            config_dict = json.load(f)

        # Reconstruct sub-configs
        model_config = ModelConfig(**config_dict.pop("model"))
        training_config = TrainingConfig(**config_dict.pop("training"))
        data_config = DataConfig(**config_dict.pop("data"))
        wandb_config = WandbConfig(**config_dict.pop("wandb"))

        return cls(
            model=model_config,
            training=training_config,
            data=data_config,
            wandb=wandb_config,
            **config_dict,
        )

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Validate batch size
        if self.training.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.training.batch_size}")

        # Validate model type
        if self.model.model_type not in ["baseline", "gated"]:
            raise ValueError(f"model_type must be 'baseline' or 'gated', got {self.model.model_type}")

        # Validate quantization for gated models
        if self.model.model_type == "gated":
            if self.model.load_in_4bit or self.model.load_in_8bit:
                print("WARNING: Quantization disabled for gated LoRA (interferes with gating)")
                self.model.load_in_4bit = False
                self.model.load_in_8bit = False

        # Validate expert configuration
        if self.model.model_type == "gated":
            if len(self.model.expert_ranks) != len(self.model.expert_alphas):
                raise ValueError(
                    f"expert_ranks ({len(self.model.expert_ranks)}) must match "
                    f"expert_alphas ({len(self.model.expert_alphas)})"
                )

        # Validate top-k
        if self.model.use_top_k:
            if self.model.top_k > len(self.model.expert_ranks):
                self.model.top_k = len(self.model.expert_ranks)
                print(f"WARNING: top_k reduced to {self.model.top_k} (num_experts)")


# ============================================================================
# Experiment Configuration Factories
# ============================================================================

def create_exp1_baseline_config() -> ExperimentConfig:
    """
    Experiment 1: Baseline LoRA (r=16, fixed)

    Standard LoRA without gating. Reference for comparison.
    """
    config = ExperimentConfig()

    # Model
    config.model.model_type = "baseline"
    config.model.lora_r = 16
    config.model.lora_alpha = 32
    config.model.lora_target_modules = ["q_proj", "k_proj", "v_proj", "dense"]
    config.model.load_in_4bit = False
    config.model.load_in_8bit = False

    # Training
    config.training.num_epochs = 4
    config.training.batch_size = 8
    config.training.gradient_accumulation_steps = 4
    config.training.learning_rate = 2e-4
    config.training.eval_steps = 100  # Eval more frequently

    # Data
    config.data.use_multi_task = True
    config.data.task_datasets = ["squad", "imdb", "conll2003", "wikitext"]
    config.data.task_weights = [0.3, 0.25, 0.25, 0.2]
    config.data.max_train_samples = 5000

    # Metadata
    config.experiment_name = "exp1_baseline_lora"
    config.wandb.tags = ["baseline", "lora", "r16", "experiment1", "multi-task"]
    config.wandb.notes = "Baseline LoRA r=16 - Reference experiment"

    return config


def create_exp2_gated_2experts_config() -> ExperimentConfig:
    """
    Experiment 2: Gated LoRA with 2 experts (r=8, r=16)

    Simple gated setup to test basic routing.
    """
    config = ExperimentConfig()

    # Model
    config.model.model_type = "gated"
    config.model.expert_ranks = [8, 16]
    config.model.expert_alphas = [16, 32]
    config.model.lora_target_modules = ["q_proj", "k_proj", "v_proj", "dense"]
    config.model.per_layer_gating = True
    config.model.use_top_k = False
    config.model.use_load_balancing = False

    # Training
    config.training.num_epochs = 4
    config.training.batch_size = 8
    config.training.gradient_accumulation_steps = 4
    config.training.learning_rate = 2e-4
    config.training.eval_steps = 100  # Eval more frequently
    config.training.gating_warmup_epochs = 0  # No warmup - train everything together
    config.training.freeze_experts_during_warmup = False

    # Data
    config.data.use_multi_task = True
    config.data.task_datasets = ["squad", "imdb", "conll2003", "wikitext"]
    config.data.task_weights = [0.3, 0.25, 0.25, 0.2]
    config.data.max_train_samples = 5000

    # Metadata
    config.experiment_name = "exp2_gated_2experts"
    config.wandb.tags = ["gated", "2experts", "r8-r16", "experiment2", "multi-task"]
    config.wandb.notes = "Gated LoRA with 2 experts (r=8, r=16) - Basic gating"

    return config


def create_exp3_gated_3experts_config() -> ExperimentConfig:
    """
    Experiment 3: Gated LoRA with 3 experts (r=8, r=16, r=32)

    Full gated setup with hierarchical capacity.
    """
    config = ExperimentConfig()

    # Model
    config.model.model_type = "gated"
    config.model.expert_ranks = [8, 16, 32]
    config.model.expert_alphas = [16, 32, 64]
    config.model.lora_target_modules = ["q_proj", "k_proj", "v_proj", "dense"]
    config.model.per_layer_gating = True
    config.model.use_top_k = False
    config.model.use_load_balancing = False

    # Training
    config.training.num_epochs = 4
    config.training.batch_size = 8
    config.training.gradient_accumulation_steps = 4
    config.training.learning_rate = 2e-4
    config.training.eval_steps = 100  # Eval more frequently
    config.training.gating_warmup_epochs = 0  # No warmup - train everything together
    config.training.freeze_experts_during_warmup = False

    # Data
    config.data.use_multi_task = True
    config.data.task_datasets = ["squad", "imdb", "conll2003", "wikitext"]
    config.data.task_weights = [0.3, 0.25, 0.25, 0.2]
    config.data.max_train_samples = 5000

    # Metadata
    config.experiment_name = "exp3_gated_3experts"
    config.wandb.tags = ["gated", "3experts", "r8-r16-r32", "experiment3", "multi-task"]
    config.wandb.notes = "Gated LoRA with 3 experts (r=8, r=16, r=32) - Full hierarchy"

    return config


def create_exp4_gated_loadbalancing_config() -> ExperimentConfig:
    """
    Experiment 4: Gated LoRA 3 experts + Load Balancing

    Tests if load balancing improves expert diversity.
    """
    config = ExperimentConfig()

    # Model
    config.model.model_type = "gated"
    config.model.expert_ranks = [8, 16, 32]
    config.model.expert_alphas = [16, 32, 64]
    config.model.lora_target_modules = ["q_proj", "k_proj", "v_proj", "dense"]
    config.model.per_layer_gating = True
    config.model.use_top_k = False
    config.model.use_load_balancing = True
    config.model.load_balancing_weight = 0.001  # Conservative

    # Training
    config.training.num_epochs = 4
    config.training.batch_size = 8
    config.training.gradient_accumulation_steps = 4
    config.training.learning_rate = 2e-4
    config.training.eval_steps = 100  # Eval more frequently
    config.training.gating_warmup_epochs = 0  # No warmup - train everything together
    config.training.freeze_experts_during_warmup = False

    # Data
    config.data.use_multi_task = True
    config.data.task_datasets = ["squad", "imdb", "conll2003", "wikitext"]
    config.data.task_weights = [0.3, 0.25, 0.25, 0.2]
    config.data.max_train_samples = 5000

    # Metadata
    config.experiment_name = "exp4_gated_loadbalancing"
    config.wandb.tags = ["gated", "3experts", "load-balancing", "experiment4", "multi-task"]
    config.wandb.notes = "Gated LoRA 3 experts + Load Balancing (weight=0.001)"

    return config


def create_exp5_gated_topk_config() -> ExperimentConfig:
    """
    Experiment 5: Gated LoRA 3 experts + Top-K Routing

    Sparse routing - only top 2 experts active per token.
    """
    config = ExperimentConfig()

    # Model
    config.model.model_type = "gated"
    config.model.expert_ranks = [8, 16, 32]
    config.model.expert_alphas = [16, 32, 64]
    config.model.lora_target_modules = ["q_proj", "k_proj", "v_proj", "dense"]
    config.model.per_layer_gating = True
    config.model.use_top_k = True
    config.model.top_k = 2
    config.model.use_load_balancing = True
    config.model.load_balancing_weight = 0.001

    # Training
    config.training.num_epochs = 4
    config.training.batch_size = 8
    config.training.gradient_accumulation_steps = 4
    config.training.learning_rate = 2e-4
    config.training.eval_steps = 100  # Eval more frequently
    config.training.gating_warmup_epochs = 0  # No warmup - train everything together
    config.training.freeze_experts_during_warmup = False

    # Data
    config.data.use_multi_task = True
    config.data.task_datasets = ["squad", "imdb", "conll2003", "wikitext"]
    config.data.task_weights = [0.3, 0.25, 0.25, 0.2]
    config.data.max_train_samples = 5000

    # Metadata
    config.experiment_name = "exp5_gated_topk"
    config.wandb.tags = ["gated", "3experts", "top-k", "sparse", "experiment5", "multi-task"]
    config.wandb.notes = "Gated LoRA 3 experts + Top-K (k=2) sparse routing"

    return config


def create_multitask_data_config() -> DataConfig:
    """
    Create data configuration for multi-task learning.

    Uses: SQuAD (QA), IMDB (sentiment), CoNLL-2003 (NER), WikiText-2 (LM)
    """
    return DataConfig(
        use_multi_task=True,
        task_datasets=["squad", "imdb", "conll2003", "wikitext"],
        task_weights=[0.3, 0.25, 0.25, 0.2],  # QA weighted higher
        preprocessing_num_workers=4,
        shuffle_seed=42,
    )


# ============================================================================
# Helper Functions
# ============================================================================

def get_experiment_config(experiment_id: int) -> ExperimentConfig:
    """Get configuration for a specific experiment number (1-5)."""
    factories = {
        1: create_exp1_baseline_config,
        2: create_exp2_gated_2experts_config,
        3: create_exp3_gated_3experts_config,
        4: create_exp4_gated_loadbalancing_config,
        5: create_exp5_gated_topk_config,
    }

    if experiment_id not in factories:
        raise ValueError(f"Unknown experiment_id: {experiment_id}. Must be 1-5.")

    return factories[experiment_id]()


def list_experiments() -> List[dict]:
    """List all available experiments."""
    return [
        {"id": 1, "name": "exp1_baseline_lora", "description": "Baseline LoRA r=16"},
        {"id": 2, "name": "exp2_gated_2experts", "description": "Gated LoRA 2 experts (r=8, r=16)"},
        {"id": 3, "name": "exp3_gated_3experts", "description": "Gated LoRA 3 experts (r=8, r=16, r=32)"},
        {"id": 4, "name": "exp4_gated_loadbalancing", "description": "Gated 3 experts + Load Balancing"},
        {"id": 5, "name": "exp5_gated_topk", "description": "Gated 3 experts + Top-K Routing"},
    ]


# =============================================================================
# Backward Compatibility (for old train.py)
# =============================================================================

def create_default_config() -> ExperimentConfig:
    """Create default config (alias for baseline)."""
    return create_exp1_baseline_config()


def create_baseline_config() -> ExperimentConfig:
    """Create baseline LoRA config (alias)."""
    return create_exp1_baseline_config()


def create_gated_config() -> ExperimentConfig:
    """Create gated LoRA config (alias for 3 experts)."""
    return create_exp3_gated_3experts_config()


if __name__ == "__main__":
    # Test config creation
    print("Available experiments:")
    for exp in list_experiments():
        print(f"  {exp['id']}: {exp['name']} - {exp['description']}")

    print("\n--- Experiment 1 (Baseline) ---")
    config1 = get_experiment_config(1)
    print(f"Model type: {config1.model.model_type}")
    print(f"LoRA r: {config1.model.lora_r}")

    print("\n--- Experiment 5 (Top-K) ---")
    config5 = get_experiment_config(5)
    print(f"Model type: {config5.model.model_type}")
    print(f"Expert ranks: {config5.model.expert_ranks}")
    print(f"Top-K: {config5.model.use_top_k}, k={config5.model.top_k}")
    print(f"Load balancing: {config5.model.use_load_balancing}")

    # Save example config
    config5.save("test_exp5_config.json")
    print("\nSaved exp5 config to test_exp5_config.json")
