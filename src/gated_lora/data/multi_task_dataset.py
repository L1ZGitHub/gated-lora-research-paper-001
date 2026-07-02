"""
Multi-Task Dataset Loader for Gated LoRA experiments - HARDER MULTITASK VERSION.

Supports 8 diverse tasks spanning different cognitive capabilities:
- SQuAD (Question Answering - Reading Comprehension)
- IMDB (Sentiment Analysis - Document Classification)
- CoNLL-2003 (Named Entity Recognition - Token Classification)
- WikiText-2 (Language Modeling - Text Continuation)
- GSM8K (Math Reasoning - Step-by-step Problem Solving) [NEW]
- XSum (Summarization - Extreme Compression) [NEW]
- CommonsenseQA (Commonsense Reasoning - Multiple Choice) [NEW]
- MNLI (Natural Language Inference - Entailment) [NEW]

Each task is formatted as text for causal language modeling.
"""

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
import random

try:
    from datasets import load_dataset, DatasetDict
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

try:
    from transformers import AutoTokenizer, PreTrainedTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class TaskDataset(Dataset):
    """
    Base dataset for a single task, formatted for causal LM.
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        task_name: str = "generic",
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_name = task_name

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Causal LM labels: mask padding positions with -100 so the loss is
        # not computed on pad tokens (pad_token == eos_token for Phi-2 & co,
        # so without this the model is trained to spam EOS after the answer
        # and perplexity numbers are polluted by trivial pad predictions).
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "task": self.task_name,
        }


# =============================================================================
# ORIGINAL TASKS (from multirun)
# =============================================================================

def format_squad_example(example: Dict) -> str:
    """Format SQuAD example as text for causal LM."""
    context = example.get("context", "")
    question = example.get("question", "")
    answers = example.get("answers", {})

    if isinstance(answers, dict):
        answer_text = answers.get("text", [""])[0] if answers.get("text") else ""
    else:
        answer_text = answers[0]["text"] if answers else ""

    # Format as QA prompt
    text = f"Context: {context}\n\nQuestion: {question}\n\nAnswer: {answer_text}"
    return text


def format_imdb_example(example: Dict) -> str:
    """Format IMDB example as text for causal LM."""
    text = example.get("text", "")
    label = example.get("label", 0)
    sentiment = "positive" if label == 1 else "negative"

    # Format as sentiment analysis
    formatted = f"Review: {text}\n\nSentiment: {sentiment}"
    return formatted


def format_conll_example(example: Dict) -> str:
    """Format CoNLL-2003 example as text for causal LM."""
    tokens = example.get("tokens", [])
    ner_tags = example.get("ner_tags", [])

    # NER tag mapping (simplified)
    tag_names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

    # Format as NER annotation
    text = " ".join(tokens)

    # Extract entities
    entities = []
    current_entity = []
    current_type = None

    for token, tag_id in zip(tokens, ner_tags):
        tag = tag_names[tag_id] if tag_id < len(tag_names) else "O"

        if tag.startswith("B-"):
            if current_entity:
                entities.append((" ".join(current_entity), current_type))
            current_entity = [token]
            current_type = tag[2:]
        elif tag.startswith("I-") and current_type == tag[2:]:
            current_entity.append(token)
        else:
            if current_entity:
                entities.append((" ".join(current_entity), current_type))
            current_entity = []
            current_type = None

    if current_entity:
        entities.append((" ".join(current_entity), current_type))

    # Format output
    entity_str = ", ".join([f"{e[0]} ({e[1]})" for e in entities]) if entities else "None"
    formatted = f"Text: {text}\n\nEntities: {entity_str}"
    return formatted


def format_wikitext_example(example: Dict) -> str:
    """Format WikiText example (already plain text)."""
    text = example.get("text", "")
    # Filter out empty lines and headers
    if text.strip() and not text.strip().startswith("="):
        return text.strip()
    return ""


# =============================================================================
# NEW HARDER TASKS
# =============================================================================

def format_gsm8k_example(example: Dict) -> str:
    """
    Format GSM8K math problem for causal LM.
    GSM8K contains grade school math word problems with step-by-step solutions.
    """
    question = example.get("question", "")
    answer = example.get("answer", "")

    # Format as math problem solving
    formatted = f"Math Problem: {question}\n\nSolution: {answer}"
    return formatted


def format_xsum_example(example: Dict) -> str:
    """
    Format XSum summarization example for causal LM.
    XSum is extreme summarization - single sentence summaries of news articles.
    """
    document = example.get("document", "")
    summary = example.get("summary", "")

    # Format as summarization task
    formatted = f"Article: {document}\n\nSummary: {summary}"
    return formatted


def format_commonsenseqa_example(example: Dict) -> str:
    """
    Format CommonsenseQA example for causal LM.
    Multiple choice questions requiring commonsense reasoning.
    """
    question = example.get("question", "")
    choices = example.get("choices", {})
    answer_key = example.get("answerKey", "")

    # Format choices
    choice_labels = choices.get("label", [])
    choice_texts = choices.get("text", [])

    choices_str = ""
    correct_answer = ""
    for label, text in zip(choice_labels, choice_texts):
        choices_str += f"\n  {label}) {text}"
        if label == answer_key:
            correct_answer = f"{label}) {text}"

    # Format as MCQ
    formatted = f"Question: {question}\n\nChoices:{choices_str}\n\nAnswer: {correct_answer}"
    return formatted


def format_mnli_example(example: Dict) -> str:
    """
    Format MNLI (Multi-Genre NLI) example for causal LM.
    Natural language inference: entailment, contradiction, or neutral.
    """
    premise = example.get("premise", "")
    hypothesis = example.get("hypothesis", "")
    label = example.get("label", -1)

    # Label mapping
    label_names = {0: "entailment", 1: "neutral", 2: "contradiction"}
    label_str = label_names.get(label, "unknown")

    # Skip examples with unknown labels (-1)
    if label == -1:
        return ""

    # Format as NLI task
    formatted = f"Premise: {premise}\n\nHypothesis: {hypothesis}\n\nRelationship: {label_str}"
    return formatted


# =============================================================================
# DATASET LOADER
# =============================================================================

class MultiTaskDatasetLoader:
    """
    Loads and combines multiple task datasets.
    Enhanced version with 8 diverse tasks for harder multi-task learning.
    """

    # Task categories for analysis
    TASK_CATEGORIES = {
        "squad": "reading_comprehension",
        "imdb": "classification",
        "conll2003": "token_classification",
        "wikitext": "language_modeling",
        "gsm8k": "reasoning",
        "xsum": "generation",
        "commonsenseqa": "reasoning",
        "mnli": "classification",
    }

    # Task complexity estimates (higher = more complex)
    TASK_COMPLEXITY = {
        "squad": 0.6,
        "imdb": 0.3,
        "conll2003": 0.5,
        "wikitext": 0.4,
        "gsm8k": 0.9,       # Math reasoning is hard
        "xsum": 0.7,        # Summarization requires abstraction
        "commonsenseqa": 0.8,  # Reasoning required
        "mnli": 0.6,        # Understanding relationships
    }

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        task_datasets: List[str] = None,
        task_weights: List[float] = None,
        max_samples_per_task: Optional[int] = None,
        seed: int = 42,
        strict: bool = True,
        max_eval_samples_per_task: int = 1000,
    ):
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets library required: pip install datasets")

        self.tokenizer = tokenizer
        self.max_length = max_length
        # strict=True: a task that fails to load ABORTS the run instead of
        # being silently dropped. Silent dropping is doubly wrong: (a) the
        # experiment no longer trains on the advertised task mix, and (b) the
        # task_weights list gets misaligned with the surviving tasks.
        self.strict = strict
        self.max_eval_samples_per_task = max_eval_samples_per_task

        # Default: all 8 tasks
        self.task_datasets = task_datasets or [
            "squad", "imdb", "conll2003", "wikitext",
            "gsm8k", "xsum", "commonsenseqa", "mnli"
        ]

        # Default weights - slightly favor harder tasks to balance difficulty
        if task_weights is None:
            self.task_weights = [0.12, 0.10, 0.12, 0.10, 0.15, 0.14, 0.14, 0.13]
        else:
            self.task_weights = task_weights

        self.max_samples_per_task = max_samples_per_task
        self.seed = seed

        # Normalize weights
        total_weight = sum(self.task_weights)
        self.task_weights = [w / total_weight for w in self.task_weights]

        logger.info(f"MultiTaskDatasetLoader (HARDER VERSION) initialized:")
        logger.info(f"  Tasks: {self.task_datasets}")
        logger.info(f"  Weights: {self.task_weights}")
        logger.info(f"  Max samples per task: {self.max_samples_per_task}")

    # =========================================================================
    # Original task loaders
    # =========================================================================

    def _load_squad(self, split: str = "train") -> List[str]:
        """Load SQuAD dataset."""
        try:
            dataset = load_dataset("squad", split=split)
            texts = [format_squad_example(ex) for ex in dataset]
            texts = [t for t in texts if t.strip()]

            if self.max_samples_per_task:
                texts = texts[:self.max_samples_per_task]

            logger.info(f"Loaded {len(texts)} SQuAD examples")
            return texts
        except Exception as e:
            logger.warning(f"Failed to load SQuAD: {e}")
            return []

    def _load_imdb(self, split: str = "train") -> List[str]:
        """Load IMDB dataset."""
        try:
            dataset = load_dataset("imdb", split=split)
            texts = [format_imdb_example(ex) for ex in dataset]
            texts = [t for t in texts if t.strip()]

            if self.max_samples_per_task:
                texts = texts[:self.max_samples_per_task]

            logger.info(f"Loaded {len(texts)} IMDB examples")
            return texts
        except Exception as e:
            logger.warning(f"Failed to load IMDB: {e}")
            return []

    def _load_conll(self, split: str = "train") -> List[str]:
        """Load CoNLL-2003 dataset.

        The canonical "conll2003" repo is script-based, which modern
        `datasets` (>=3) refuses to load ("Dataset scripts are no longer
        supported"). We try parquet-native mirrors first.
        """
        candidates = ["eriktks/conll2003", "conll2003"]
        last_err: Optional[Exception] = None
        for repo in candidates:
            try:
                dataset = load_dataset(repo, split=split)
                texts = [format_conll_example(ex) for ex in dataset]
                texts = [t for t in texts if t.strip()]

                if self.max_samples_per_task:
                    texts = texts[:self.max_samples_per_task]

                logger.info(f"Loaded {len(texts)} CoNLL-2003 examples (from {repo})")
                return texts
            except Exception as e:
                last_err = e
                logger.warning(f"CoNLL-2003 load failed from {repo}: {e}")
        logger.warning(f"Failed to load CoNLL-2003 from all sources: {last_err}")
        return []

    def _load_wikitext(self, split: str = "train") -> List[str]:
        """Load WikiText-2 dataset."""
        try:
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
            texts = [format_wikitext_example(ex) for ex in dataset]
            texts = [t for t in texts if t.strip() and len(t) > 50]

            if self.max_samples_per_task:
                texts = texts[:self.max_samples_per_task]

            logger.info(f"Loaded {len(texts)} WikiText examples")
            return texts
        except Exception as e:
            logger.warning(f"Failed to load WikiText: {e}")
            return []

    # =========================================================================
    # NEW task loaders
    # =========================================================================

    def _load_gsm8k(self, split: str = "train") -> List[str]:
        """Load GSM8K math reasoning dataset."""
        try:
            # GSM8K has train and test splits
            actual_split = "train" if split == "train" else "test"
            dataset = load_dataset("gsm8k", "main", split=actual_split)
            texts = [format_gsm8k_example(ex) for ex in dataset]
            texts = [t for t in texts if t.strip()]

            if self.max_samples_per_task:
                texts = texts[:self.max_samples_per_task]

            logger.info(f"Loaded {len(texts)} GSM8K examples")
            return texts
        except Exception as e:
            logger.warning(f"Failed to load GSM8K: {e}")
            return []

    def _load_xsum(self, split: str = "train") -> List[str]:
        """Load XSum summarization dataset."""
        try:
            dataset = load_dataset("EdinburghNLP/xsum", split=split)
            texts = [format_xsum_example(ex) for ex in dataset]
            texts = [t for t in texts if t.strip()]

            if self.max_samples_per_task:
                texts = texts[:self.max_samples_per_task]

            logger.info(f"Loaded {len(texts)} XSum examples")
            return texts
        except Exception as e:
            logger.warning(f"Failed to load XSum: {e}")
            return []

    def _load_commonsenseqa(self, split: str = "train") -> List[str]:
        """Load CommonsenseQA reasoning dataset."""
        try:
            # CommonsenseQA has train, validation splits (test has no labels)
            actual_split = split if split in ["train", "validation"] else "validation"
            dataset = load_dataset("commonsense_qa", split=actual_split)
            texts = [format_commonsenseqa_example(ex) for ex in dataset]
            texts = [t for t in texts if t.strip()]

            if self.max_samples_per_task:
                texts = texts[:self.max_samples_per_task]

            logger.info(f"Loaded {len(texts)} CommonsenseQA examples")
            return texts
        except Exception as e:
            logger.warning(f"Failed to load CommonsenseQA: {e}")
            return []

    def _load_mnli(self, split: str = "train") -> List[str]:
        """Load MNLI natural language inference dataset."""
        try:
            # MNLI has train, validation_matched, validation_mismatched
            if split == "validation":
                actual_split = "validation_matched"
            elif split == "test":
                actual_split = "validation_mismatched"  # Use mismatched for test
            else:
                actual_split = "train"

            dataset = load_dataset("glue", "mnli", split=actual_split)
            texts = [format_mnli_example(ex) for ex in dataset]
            texts = [t for t in texts if t.strip()]

            if self.max_samples_per_task:
                texts = texts[:self.max_samples_per_task]

            logger.info(f"Loaded {len(texts)} MNLI examples")
            return texts
        except Exception as e:
            logger.warning(f"Failed to load MNLI: {e}")
            return []

    # =========================================================================
    # Main loader interface
    # =========================================================================

    def load_task(self, task_name: str, split: str = "train") -> List[str]:
        """Load a specific task dataset."""
        loaders = {
            # Original tasks
            "squad": self._load_squad,
            "imdb": self._load_imdb,
            "conll2003": self._load_conll,
            "conll": self._load_conll,
            "wikitext": self._load_wikitext,
            # New harder tasks
            "gsm8k": self._load_gsm8k,
            "xsum": self._load_xsum,
            "commonsenseqa": self._load_commonsenseqa,
            "mnli": self._load_mnli,
        }

        if task_name.lower() not in loaders:
            if self.strict:
                raise ValueError(f"Unknown task: {task_name}")
            logger.warning(f"Unknown task: {task_name}")
            return []

        texts = loaders[task_name.lower()](split)
        if not texts and self.strict:
            raise RuntimeError(
                f"Task '{task_name}' (split={split}) loaded 0 examples. "
                f"Refusing to continue: the experiment would silently train on "
                f"fewer tasks than configured. Fix the loader or remove the task."
            )
        return texts

    def create_dataset(self, split: str = "train") -> Dataset:
        """
        Create combined multi-task dataset.

        Returns:
            ConcatDataset combining all task datasets
        """
        datasets = []

        for task_name in self.task_datasets:
            texts = self.load_task(task_name, split)

            if texts:
                task_dataset = TaskDataset(
                    texts=texts,
                    tokenizer=self.tokenizer,
                    max_length=self.max_length,
                    task_name=task_name,
                )
                datasets.append(task_dataset)

        if not datasets:
            raise ValueError("No datasets could be loaded!")

        combined = ConcatDataset(datasets)
        logger.info(f"Created combined dataset with {len(combined)} total examples")

        return combined

    def create_weighted_dataloader(
        self,
        split: str = "train",
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> DataLoader:
        """
        Create dataloader with weighted sampling across tasks.

        This ensures balanced sampling according to task_weights.
        """
        datasets = []
        task_sizes = []
        loaded_weights = []  # weights aligned with SUCCESSFULLY loaded tasks

        for task_idx, task_name in enumerate(self.task_datasets):
            texts = self.load_task(task_name, split)

            if texts:
                task_dataset = TaskDataset(
                    texts=texts,
                    tokenizer=self.tokenizer,
                    max_length=self.max_length,
                    task_name=task_name,
                )
                datasets.append(task_dataset)
                task_sizes.append(len(texts))
                # Index into the ORIGINAL weight list: if a task fails to load
                # (strict=False), zipping sizes with the full weight list would
                # silently shift every subsequent task onto the wrong weight.
                loaded_weights.append(self.task_weights[task_idx])

        if not datasets:
            raise ValueError("No datasets could be loaded!")

        combined = ConcatDataset(datasets)

        # Create sample weights for weighted random sampling
        sample_weights = []

        for task_size, task_weight in zip(task_sizes, loaded_weights):
            # Weight per sample in this task
            weight_per_sample = task_weight / task_size
            sample_weights.extend([weight_per_sample] * task_size)

        # Normalize weights
        total_weight = sum(sample_weights)
        sample_weights = [w / total_weight for w in sample_weights]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(combined),
            replacement=True,
        )

        def collate_fn(batch):
            """Custom collate to handle task labels."""
            input_ids = torch.stack([b["input_ids"] for b in batch])
            attention_mask = torch.stack([b["attention_mask"] for b in batch])
            labels = torch.stack([b["labels"] for b in batch])
            tasks = [b["task"] for b in batch]

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "task": tasks,
            }

        dataloader = DataLoader(
            combined,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

        logger.info(f"Created weighted dataloader with batch_size={batch_size}")

        return dataloader

    def create_eval_dataloader(
        self,
        split: str = "validation",
        batch_size: int = 4,
        num_workers: int = 4,
    ) -> DataLoader:
        """Create evaluation dataloader (no weighted sampling)."""
        # Map splits for different datasets
        split_map = {
            "validation": {
                "squad": "validation",
                "imdb": "test",  # IMDB uses test for eval
                "conll2003": "validation",
                "conll": "validation",
                "wikitext": "validation",
                "gsm8k": "test",  # GSM8K only has train/test
                "xsum": "validation",
                "commonsenseqa": "validation",
                "mnli": "validation",
            },
            "test": {
                "squad": "validation",  # SQuAD test is hidden
                "imdb": "test",
                "conll2003": "test",
                "conll": "test",
                "wikitext": "test",
                "gsm8k": "test",
                "xsum": "test",
                "commonsenseqa": "validation",  # Test has no labels
                "mnli": "test",
            },
        }

        datasets = []

        for task_name in self.task_datasets:
            actual_split = split_map.get(split, {}).get(task_name.lower(), split)
            texts = self.load_task(task_name, actual_split)

            # Limit eval samples
            if texts:
                texts = texts[:self.max_eval_samples_per_task]
                task_dataset = TaskDataset(
                    texts=texts,
                    tokenizer=self.tokenizer,
                    max_length=self.max_length,
                    task_name=task_name,
                )
                datasets.append(task_dataset)

        if not datasets:
            raise ValueError("No eval datasets could be loaded!")

        combined = ConcatDataset(datasets)

        def collate_fn(batch):
            input_ids = torch.stack([b["input_ids"] for b in batch])
            attention_mask = torch.stack([b["attention_mask"] for b in batch])
            labels = torch.stack([b["labels"] for b in batch])
            tasks = [b["task"] for b in batch]

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "task": tasks,
            }

        dataloader = DataLoader(
            combined,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        logger.info(f"Created eval dataloader with {len(combined)} samples")

        return dataloader


def create_single_task_dataloader(
    task_name: str,
    tokenizer: PreTrainedTokenizer,
    split: str = "train",
    max_length: int = 512,
    batch_size: int = 4,
    max_samples: Optional[int] = None,
    num_workers: int = 4,
) -> DataLoader:
    """
    Create dataloader for a single task (for baseline comparisons).
    """
    loader = MultiTaskDatasetLoader(
        tokenizer=tokenizer,
        max_length=max_length,
        task_datasets=[task_name],
        task_weights=[1.0],
        max_samples_per_task=max_samples,
    )

    dataset = loader.create_dataset(split)

    def collate_fn(batch):
        input_ids = torch.stack([b["input_ids"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
        labels = torch.stack([b["labels"] for b in batch])
        tasks = [b["task"] for b in batch]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "task": tasks,
        }

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return dataloader


# =============================================================================
# Preset configurations for experiments
# =============================================================================

def get_original_4_tasks():
    """Original 4 tasks from multirun experiments."""
    return {
        "tasks": ["squad", "imdb", "conll2003", "wikitext"],
        "weights": [0.3, 0.25, 0.25, 0.2],
    }


def get_harder_4_tasks():
    """4 new harder tasks only."""
    return {
        "tasks": ["gsm8k", "xsum", "commonsenseqa", "mnli"],
        "weights": [0.3, 0.25, 0.25, 0.2],
    }


def get_all_8_tasks():
    """All 8 tasks combined."""
    return {
        "tasks": ["squad", "imdb", "conll2003", "wikitext", "gsm8k", "xsum", "commonsenseqa", "mnli"],
        "weights": [0.12, 0.10, 0.12, 0.10, 0.15, 0.14, 0.14, 0.13],
    }


def get_diverse_6_tasks():
    """6 most diverse tasks (one per category)."""
    return {
        "tasks": ["squad", "imdb", "conll2003", "gsm8k", "xsum", "commonsenseqa"],
        "weights": [0.17, 0.14, 0.17, 0.20, 0.16, 0.16],
    }


def get_reasoning_focused():
    """Focus on reasoning tasks."""
    return {
        "tasks": ["squad", "gsm8k", "commonsenseqa", "mnli"],
        "weights": [0.25, 0.30, 0.25, 0.20],
    }


if __name__ == "__main__":
    # Test the multi-task dataset loader
    print("Testing MultiTaskDatasetLoader (HARDER VERSION)...")
    print("="*60)

    if not TRANSFORMERS_AVAILABLE:
        print("transformers not available - cannot test")
        exit(1)

    if not DATASETS_AVAILABLE:
        print("datasets not available - cannot test")
        exit(1)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Test each task individually
    print("\n" + "="*60)
    print("Testing individual task loading...")
    print("="*60)

    test_tasks = ["gsm8k", "xsum", "commonsenseqa", "mnli"]

    for task in test_tasks:
        print(f"\n--- Testing {task} ---")
        loader = MultiTaskDatasetLoader(
            tokenizer=tokenizer,
            max_length=256,
            task_datasets=[task],
            task_weights=[1.0],
            max_samples_per_task=5,
        )

        try:
            texts = loader.load_task(task, "train")
            if texts:
                print(f"  Loaded {len(texts)} examples")
                print(f"  Sample (first 300 chars):\n  {texts[0][:300]}...")
            else:
                print(f"  No examples loaded")
        except Exception as e:
            print(f"  Error: {e}")

    # Test combined loading
    print("\n" + "="*60)
    print("Testing combined 8-task loader...")
    print("="*60)

    config = get_all_8_tasks()
    loader = MultiTaskDatasetLoader(
        tokenizer=tokenizer,
        max_length=256,
        task_datasets=config["tasks"],
        task_weights=config["weights"],
        max_samples_per_task=50,
    )

    # Create dataloader
    print("\nCreating train dataloader...")
    try:
        train_loader = loader.create_weighted_dataloader(
            split="train",
            batch_size=2,
            num_workers=0,
        )

        # Test a batch
        print("\nTesting a batch...")
        batch = next(iter(train_loader))
        print(f"input_ids shape: {batch['input_ids'].shape}")
        print(f"attention_mask shape: {batch['attention_mask'].shape}")
        print(f"labels shape: {batch['labels'].shape}")
        print(f"tasks: {batch['task']}")

        # Decode first example
        print("\nFirst example text (truncated):")
        decoded = tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True)
        print(decoded[:300] + "...")

        print("\n" + "="*60)
        print("All tests passed!")
        print("="*60)
    except Exception as e:
        print(f"Error during combined test: {e}")
        import traceback
        traceback.print_exc()
