"""
Base model implementation using Phi-2 from Microsoft.
Optimized for RTX 4070 laptop (8GB VRAM) with memory-efficient loading.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from typing import Optional, Dict, Any, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Phi2BaseModel:
    """
    Base model wrapper for Microsoft Phi-2.

    Features:
    - Memory-efficient loading with device_map="auto"
    - Frozen base model for LoRA fine-tuning
    - Support for 8-bit/16-bit quantization
    - LoRA adapter support via add_lora() method
    - Forward pass with loss computation
    - Text generation interface
    """

    def __init__(
        self,
        model_name: str = "microsoft/phi-2",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
        trust_remote_code: bool = True,
        freeze_base: bool = True,
        gradient_checkpointing: bool = True,
    ):
        """
        Initialize Phi-2 base model.

        Args:
            model_name: HuggingFace model identifier
            load_in_8bit: Enable 8-bit quantization (reduces VRAM to ~3.5GB)
            load_in_4bit: Enable 4-bit quantization (reduces VRAM to ~2GB)
            torch_dtype: Data type for model weights (bfloat16 recommended for RTX 4070)
            device_map: Device mapping strategy ("auto" for automatic distribution)
            trust_remote_code: Allow custom code execution (required for Phi-2)
            freeze_base: Freeze all model parameters (for LoRA training)
            gradient_checkpointing: Enable gradient checkpointing (saves memory but slows training ~20-30%)
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lora_enabled = False
        self.is_quantized = load_in_8bit or load_in_4bit

        logger.info(f"Loading {model_name} on {self.device}")
        logger.info(f"Quantization: 8bit={load_in_8bit}, 4bit={load_in_4bit}")

        # Configure quantization if enabled
        quantization_config = None
        if load_in_8bit or load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                bnb_4bit_compute_dtype=torch_dtype if load_in_4bit else None,
                bnb_4bit_use_double_quant=True if load_in_4bit else False,
                bnb_4bit_quant_type="nf4" if load_in_4bit else None,
            )
            # When using quantization, use float16 instead of bfloat16
            torch_dtype = torch.float16

        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )

        # Set pad token if not exists (common issue with Phi-2)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load model with memory-efficient settings
        logger.info("Loading model...")

        # Force loading to GPU if device_map is "auto" and we have a GPU
        # This prevents the model from being split across CPU/GPU
        if device_map == "auto" and torch.cuda.is_available():
            effective_device_map = {"": 0}  # Load everything to GPU 0
            logger.info("Forcing model to GPU 0 (overriding device_map='auto')")
        else:
            effective_device_map = device_map

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=effective_device_map,
            trust_remote_code=trust_remote_code,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
        )

        # Freeze base model parameters if requested
        if freeze_base:
            self._freeze_base_model()

        # Enable gradient checkpointing for memory efficiency during training
        if gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        elif not gradient_checkpointing:
            logger.info("Gradient checkpointing disabled for faster training")

        self._log_memory_usage()

    def _freeze_base_model(self):
        """Freeze all model parameters to prepare for LoRA fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = False

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())

        logger.info(f"Base model frozen: {trainable_params:,}/{total_params:,} trainable parameters")

    def _log_memory_usage(self):
        """Log current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            logger.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    def add_lora(
        self,
        r: int = 16,
        lora_alpha: int = 32,
        target_modules: list = None,
        lora_dropout: float = 0.1,
        bias: str = "none",
        task_type: str = "CAUSAL_LM",
    ):
        """
        Add LoRA (Low-Rank Adaptation) adapters to the model.

        Args:
            r: LoRA rank (dimension of low-rank matrices)
            lora_alpha: LoRA scaling factor (alpha/r is the actual scaling)
            target_modules: List of module names to apply LoRA to
            lora_dropout: Dropout probability for LoRA layers
            bias: Bias training strategy ("none", "all", "lora_only")
            task_type: Type of task ("CAUSAL_LM" for language modeling)

        Returns:
            None. Modifies self.model in-place.
        """
        if self.lora_enabled:
            logger.warning("LoRA already enabled. Skipping.")
            return

        if target_modules is None:
            target_modules = ["q_proj", "v_proj"]

        logger.info("Adding LoRA adapters to model...")
        logger.info(f"LoRA config: r={r}, alpha={lora_alpha}, dropout={lora_dropout}")
        logger.info(f"Target modules: {target_modules}")

        # Create LoRA configuration
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias=bias,
            task_type=task_type,
        )

        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        self.lora_enabled = True

        # Log trainable parameters
        params = self.get_trainable_params()
        logger.info(
            f"LoRA adapters added: {params['trainable_params']:,}/{params['total_params']:,} "
            f"trainable ({params['trainable_percentage']:.2f}%)"
        )

        # Print trainable parameter details
        self.model.print_trainable_parameters()

        self._log_memory_usage()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Labels for loss computation [batch_size, seq_len]
            **kwargs: Additional arguments passed to model

        Returns:
            Dictionary containing:
                - loss: Language modeling loss (if labels provided)
                - logits: Model logits [batch_size, seq_len, vocab_size]
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

        result = {
            "logits": outputs.logits,
        }

        if labels is not None:
            result["loss"] = outputs.loss

        return result

    def generate(
        self,
        prompt: Union[str, list],
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        repetition_penalty: float = 1.1,
        **kwargs
    ) -> Union[str, list]:
        """
        Generate text completion(s) given a prompt.

        Args:
            prompt: Input text or list of texts
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling probability
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling (vs greedy)
            num_return_sequences: Number of sequences to generate per prompt
            repetition_penalty: Penalty for repeating tokens
            **kwargs: Additional generation parameters

        Returns:
            Generated text(s). Returns string if prompt is string and num_return_sequences=1,
            otherwise returns list of strings.
        """
        # Handle single string input
        is_single = isinstance(prompt, str)
        if is_single:
            prompt = [prompt]

        # Tokenize inputs
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,  # Phi-2 context length
        )

        # Move to device
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )

        # Decode outputs
        generated_texts = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
        )

        # Return format based on input
        if is_single and num_return_sequences == 1:
            return generated_texts[0]
        return generated_texts

    def get_trainable_params(self) -> Dict[str, int]:
        """
        Get count of trainable vs total parameters.

        Returns:
            Dictionary with trainable and total parameter counts.
        """
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())

        return {
            "trainable_params": trainable,
            "total_params": total,
            "trainable_percentage": 100 * trainable / total if total > 0 else 0,
        }

    def save_pretrained(self, output_dir: str):
        """Save model and tokenizer to directory."""
        logger.info(f"Saving model to {output_dir}")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def __call__(self, *args, **kwargs):
        """Allow direct calling of the model."""
        return self.forward(*args, **kwargs)


if __name__ == "__main__":
    # Quick test
    print("Initializing Phi-2 base model...")
    model = Phi2BaseModel(load_in_8bit=True)  # Use 8-bit for RTX 4070 laptop

    print("\nModel info:")
    print(model.get_trainable_params())

    print("\nTesting generation:")
    prompt = "The future of artificial intelligence is"
    output = model.generate(prompt, max_new_tokens=50)
    print(f"\nPrompt: {prompt}")
    print(f"Output: {output}")
