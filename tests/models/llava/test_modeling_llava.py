# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Testing suite for the PyTorch Llava model. """
from typing import Callable, Optional, Dict, List, Tuple, Union, Any, Literal
import torch
from trl import RewardConfig, RewardTrainer, is_peft_available
from trl.trainer import compute_accuracy
from typing import Callable, Optional, Dict, List, Tuple, Union, Any
from torch.utils.data import Dataset
import warnings
from transformers import (
    DataCollator,
    PreTrainedModel,
    TrainingArguments,
    LlavaProcessor,
    PreTrainedTokenizerBase,
)
import json
import torch.nn as nn
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from dataclasses import dataclass
import inspect
import transformers
from dataclasses import dataclass, field
from datasets import load_dataset
import os


if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training


import gc
import unittest
from peft import LoraConfig, PeftModelForCausalLM
import requests

from transformers import (
    AutoProcessor,
    LlavaConfig,
    LlavaForConditionalGeneration,
    is_torch_available,
    is_vision_available,
    BitsAndBytesConfig,
)
from transformers.testing_utils import require_bitsandbytes, require_torch, require_torch_gpu, slow, torch_device, require_torch_fp16

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor

GRAD_ACCUMULATION = 2
BATCH_SIZE = 8


if is_torch_available():
    import torch
else:
    is_torch_greater_or_equal_than_2_0 = False

if is_vision_available():
    from PIL import Image

starting_prompt = """
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
"""

def value_prompt(captions):
    caption_string = ""
    for caption in captions:
        caption_string += f"{caption}\n"
    return f"""
USER: Please evaluate the quality of your last response. There are several dimensions you should consider in your evaluation:

1. Accurate: The AI should provide factual and accurate information from the image, and refrain from making statements that are not supported by the image or inconsistent with the image. Specifically, the AI's response should be fully supported by the combination of the following captions:
{caption_string}
2. Helpful: The AI’s response should precisely serve the user's needs and interests, while grounding the response in the image.
3. Language Natural: The AI should employ language that flows smoothly and is free from repetitive or awkward constructs.
4. Concise: The AI should efficiently address the task or answer the question, communicating the necessary information with brevity and clarity.

A good response should be accurate, helpful, language natural, and concise. ASSISTANT: Following your definitions, the quality score of my last response is
  """

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    # From LLaVA
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    # From AlpacaFarm
    length_column_name: str = field(default="length")
    dataloader_pin_memory: bool = field(default=False)
    bf16: bool = field(default=True)
    half_precision_backend: str = field(default="auto")
    max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be left padded to this length always during training."
        },
    )
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be left padded to this length always during training."
        },
    )
    query_len: int = field(default=None, metadata={"help": "Length of the query."})
    response_len: int = field(
        default=None, metadata={"help": "Length of the response."}
    )
    label_names: List[str] = field(
        default_factory=lambda: ["index_0", "index_1", "choice"],
        metadata={
            "help": "Names of the labels in the dataset. "
            "This is needed to get transformers.Trainer to not throw those tensors away before `compute_loss`."
            "By default, the trainer throws away columns it doesn't recognize when creating the "
            "`train_dataloader` (see `_remove_unused_columns`). "
        },
    )
    padding: Literal["max_length", "longest"] = field(
        default="longest",
        metadata={
            "help": "Padding strategy. If 'max_length', pads to `model_max_length` always; this might lead to some "
            "redundant compute. If 'longest', pads to the longest sequence in the batch, capped by `model_max_length`."
        },
    )
    # From QLoRA
    full_finetune: bool = field(
        default=False, metadata={"help": "Finetune the entire model without adapters."}
    )
    adam8bit: bool = field(default=False, metadata={"help": "Use 8-bit adam."})
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        },
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        },
    )
    bits: int = field(default=4, metadata={"help": "How many bits to use."})
    lora_modules: Optional[List[str]] = field(
        default="q_proj k_proj v_proj o_proj gate_proj up_proj down_proj",
        metadata={
            "help": "Which modules to use LoRA on. If None, will use all linear layers."
        },
    )
    lora_r: int = field(default=64, metadata={"help": "Lora R dimension."})
    lora_alpha: float = field(default=16, metadata={"help": " Lora alpha."})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout."})
    report_to: str = field(
        default="none",
        metadata={"help": "To use wandb or something else for reporting."},
    )
    resume_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory containing the checkpoint to resume."},
    )
    output_dir: str = field(
        default="./output", metadata={"help": "The output dir for logs and checkpoints"}
    )
    optim: str = field(
        default="paged_adamw_32bit", metadata={"help": "The optimizer to be used"}
    )
    per_device_train_batch_size: int = field(
        default=BATCH_SIZE,
        metadata={
            "help": "The training batch size per GPU. Increase for better speed."
        },
    )
    gradient_accumulation_steps: int = field(
        default=GRAD_ACCUMULATION,
        metadata={
            "help": "How many gradients to accumulate before to perform an optimizer step"
        },
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "The L2 weight decay rate of AdamW"}
    )  # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=2e-5, metadata={"help": "The learnign rate"})
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Removed unused columns. Needed to make this codebase work."},
    )
    max_grad_norm: float = field(
        default=0.3,
        metadata={
            "help": "Gradient clipping max norm. This is tuned and works well for all models tested."
        },
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Use gradient checkpointing. You want to use this."},
    )
    do_train: bool = field(
        default=True,
        metadata={"help": "To train or not to train, that is the question?"},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={
            "help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"
        },
    )
    warmup_ratio: float = field(
        default=0.03, metadata={"help": "Fraction of steps to do a warmup for"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "The frequency of update steps after which to log the loss"},
    )
    group_by_length: bool = field(
        default=False,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )
    save_strategy: str = field(
        default="steps", metadata={"help": "When to save checkpoints"}
    )
    save_steps: int = field(default=250, metadata={"help": "How often to save a model"})
    save_total_limit: int = field(
        default=40,
        metadata={
            "help": "How many checkpoints to save before the oldest is overwritten"
        },
    )
    resume_from_training: bool = field(
        default=False, metadata={"help": "Resume from training"}
    )
    ddp_find_unused_parameters: bool = field(
        default=False, metadata={"help": "Find unused parameters"}
    )
    # ddp_backend: str = field(
    #     default="ddp", metadata={"help": "Distributed backend to use"}
    # )

def _get_generator(seed: int) -> torch.Generator:
    rng = torch.Generator()
    rng.manual_seed(seed)
    return rng


def split_train_into_train_and_eval(
    train_dataset: Dataset, eval_size: int, seed: int
) -> Tuple[Dataset, Dataset]:
    assert eval_size < len(
        train_dataset  # noqa
    ), "Requested eval_size cannot be equal/larger than original train data size."
    new_train_size = len(train_dataset) - eval_size  # noqa
    train_dataset, eval_dataset = torch.utils.data.random_split(
        train_dataset, [new_train_size, eval_size], generator=_get_generator(seed)
    )
    return train_dataset, eval_dataset


@dataclass
class RewardDataCollatorWithPadding:
    r"""
    Reward DataCollator class that pads the inputs to the maximum length of the batch.
    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for encoding the data.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the tokenizer.
        max_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the sequence to be processed.
        pad_to_multiple_of (`Optional[int]`, `optional`, defaults to `None`):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`, `optional`, defaults to `"pt"`):
            The tensor type to use.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_chosen = []
        features_rejected = []
        margin = []
        # check if we have a margin. If we do, we need to batch it as well
        has_margin = "margin" in features[0]
        pixel_values_chosen = []
        pixel_values_rejected = []
        for feature in features:
            # check if the keys are named as expected
            if (
                "input_ids_chosen" not in feature
                or "input_ids_rejected" not in feature
                or "attention_mask_chosen" not in feature
                or "attention_mask_rejected" not in feature
            ):
                raise ValueError(
                    "The features should include `input_ids_chosen`, `attention_mask_chosen`, `input_ids_rejected` and `attention_mask_rejected`"
                )

            features_chosen.append(
                {
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                }
            )
            features_rejected.append(
                {
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                }
            )
            pixel_values_chosen.append(feature["pixel_values_chosen"])
            pixel_values_rejected.append(feature["pixel_values_rejected"])
            if has_margin:
                margin.append(feature["margin"])
        # import pdb; pdb.set_trace()
        batch_chosen = self.tokenizer.pad(
            features_chosen,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_rejected = self.tokenizer.pad(
            features_rejected,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        # import pdb; pdb.set_trace()
        batch = {
            "input_ids_chosen": batch_chosen["input_ids"],
            "attention_mask_chosen": batch_chosen["attention_mask"],
            "input_ids_rejected": batch_rejected["input_ids"],
            "attention_mask_rejected": batch_rejected["attention_mask"],
            "pixel_values_chosen": torch.stack(pixel_values_chosen).squeeze(),
            "pixel_values_rejected": torch.stack(pixel_values_rejected).squeeze(),
            "return_loss": True,
        }

        # import pdb
        # pdb.set_trace()
        if has_margin:
            margin = torch.tensor(margin, dtype=torch.float)
            batch["margin"] = margin
        return batch


class MultiModalRewardTrainer(RewardTrainer):
    # def __init__(
    #     self,
    #     model: Union[PreTrainedModel, nn.Module] = None,
    #     args: Optional[RewardConfig] = None,
    #     data_collator: Optional[DataCollator] = None,
    #     train_dataset: Optional[Dataset] = None,
    #     eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
    #     processor: Optional[LlavaProcessor] = None,
    #     model_init: Optional[Callable[[], PreTrainedModel]] = None,
    #     compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
    #     callbacks: Optional[List[TrainerCallback]] = None,
    #     optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
    #         None,
    #         None,
    #     ),
    #     preprocess_logits_for_metrics: Optional[
    #         Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    #     ] = None,
    #     max_length: Optional[int] = None,
    #     peft_config: Optional[Dict] = None,
    # ):
    #     print("TRAINING ARGS", args)
    #     print("DATASET START INIT", train_dataset[0].keys())
    #     if type(args) == TrainingArguments:
    #         warnings.warn(
    #             "Using `transformers.TrainingArguments` for `args` is deprecated and will be removed in a future version. Please use `RewardConfig` instead.",
    #             FutureWarning,
    #         )
    #         if max_length is not None:
    #             warnings.warn(
    #                 "The `max_length` argument is deprecated and will be removed in a future version. Please use the `RewardConfig` to set `max_length` instead.",
    #                 FutureWarning,
    #             )
    #     else:
    #         if max_length is not None and args.max_length is not None:
    #             raise ValueError(
    #                 "You cannot specify both `max_length` and `args.max_length`. Please use the `RewardConfig` to set `max_length` once."
    #             )
    #         if max_length is not None and args.max_length is None:
    #             warnings.warn(
    #                 "The `max_length` argument is deprecated and will be removed in a future version. Please use the `RewardConfig` to set `max_length` instead.",
    #                 FutureWarning,
    #             )
    #     if not is_peft_available() and peft_config is not None:
    #         raise ValueError(
    #             "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
    #         )
    #     elif is_peft_available() and peft_config is not None:
    #         if not isinstance(model, PeftModel):
    #             if getattr(model, "is_loaded_in_8bit", False) or getattr(
    #                 model, "is_quantized", False
    #             ):
    #                 _supports_gc_kwargs = "gradient_checkpointing_kwargs" in list(
    #                     inspect.signature(prepare_model_for_kbit_training).parameters
    #                 )

    #                 preprare_model_kwargs = {
    #                     "use_gradient_checkpointing": args.gradient_checkpointing
    #                 }

    #                 if (
    #                     not _supports_gc_kwargs
    #                     and args.gradient_checkpointing_kwargs is not None
    #                 ):
    #                     warnings.warn(
    #                         "You passed `gradient_checkpointing_kwargs` in the trainer's kwargs, but your peft version does not support it. "
    #                         "please update to the latest version of peft to use `gradient_checkpointing_kwargs`."
    #                     )
    #                 elif (
    #                     _supports_gc_kwargs
    #                     and args.gradient_checkpointing_kwargs is not None
    #                 ):
    #                     preprare_model_kwargs["gradient_checkpointing_kwargs"] = (
    #                         args.gradient_checkpointing_kwargs
    #                     )

    #                 model = prepare_model_for_kbit_training(
    #                     model, **preprare_model_kwargs
    #                 )

    #             model = get_peft_model(model, peft_config)

    #     if compute_metrics is None:
    #         compute_metrics = compute_accuracy

    #     if data_collator is None:
    #         if processor is None:
    #             raise ValueError(
    #                 "max_length or a tokenizer must be specified when using the default RewardDataCollatorWithPadding"
    #             )
    #         if type(args) == TrainingArguments:
    #             if max_length is None:
    #                 warnings.warn(
    #                     "When using RewardDataCollatorWithPadding, you should set `max_length` in RewardConfig."
    #                     " It will be set to `512` by default, but you should do it yourself in the future.",
    #                     UserWarning,
    #                 )
    #                 max_length = 512
    #         else:
    #             if max_length is None and args.max_length is None:
    #                 warnings.warn(
    #                     "When using RewardDataCollatorWithPadding, you should set `max_length` in RewardConfig."
    #                     " It will be set to `512` by default, but you should do it yourself in the future.",
    #                     UserWarning,
    #                 )
    #                 max_length = 512
    #             if max_length is None and args.max_length is not None:
    #                 max_length = args.max_length

    #         data_collator = RewardDataCollatorWithPadding(
    #             processor.tokenizer, max_length=max_length
    #         )

    #         if args.remove_unused_columns:
    #             try:  # for bc before https://github.com/huggingface/transformers/pull/25435
    #                 args.remove_unused_columns = False
    #             except FrozenInstanceError:
    #                 args = replace(args, remove_unused_columns=False)
    #             # warn users
    #             warnings.warn(
    #                 "When using RewardDataCollatorWithPadding, you should set `remove_unused_columns=False` in your RewardConfig"
    #                 " we have set it for you, but you should do it yourself in the future.",
    #                 UserWarning,
    #             )

    #         self.use_reward_data_collator = True
    #     else:
    #         self.use_reward_data_collator = False
    #     print("DATASET END INIT", train_dataset[0].keys())
    #     print("***ARGS", args)

    #     super().__init__(
    #         model,
    #         args,
    #         data_collator,
    #         train_dataset,
    #         eval_dataset,
    #         processor,
    #         model_init,
    #         compute_metrics,
    #         callbacks,
    #         optimizers,
    #         preprocess_logits_for_metrics,
    #     )

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_reward_data_collator:
            warnings.warn(
                "The current compute_loss is implemented for RewardDataCollatorWithPadding,"
                " if you are using a custom data collator make sure you know what you are doing or"
                " implement your own compute_loss method."
            )


        # import pdb
        # pdb.set_trace()

        # import pdb; pdb.set_trace()
        pixel_outputs = model.vision_tower(inputs["pixel_values_chosen"])
        import pdb; pdb.set_trace()
        rewards_chosen = model(
            input_ids=inputs["input_ids_chosen"].squeeze(),
            attention_mask=inputs["attention_mask_chosen"].squeeze(),
            pixel_values=inputs["pixel_values_chosen"].squeeze(),
            return_dict=True,
        )["logits"]
        rewards_rejected = model(
            input_ids=inputs["input_ids_rejected"].squeeze(),
            attention_mask=inputs["attention_mask_rejected"].squeeze(),
            pixel_values=inputs["pixel_values_rejected"].squeeze(),
            return_dict=True,
        )["logits"]
        # calculate loss, optionally modulate with margin
        if "margin" in inputs:
            loss = -nn.functional.logsigmoid(
                rewards_chosen - rewards_rejected - inputs["margin"]
            ).mean()
        else:
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()

        if return_outputs:
            return loss, {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected,
            }
        return loss

class PreprocessDataset(Dataset):
    def __init__(self, data, processor, image_path, caption_map, starting_prompt):
        self.data = data
        self.processor = processor
        self.image_path = image_path
        self.caption_map = caption_map
        self.starting_prompt = starting_prompt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        new_example = {}
        first = example["output_1"]
        second = example["output_2"]
        choice = example["preference"]
        image = example["image"]
        conversations = example["conversations"]
        if choice == 1:
            chosen = first
            rejected = second
        elif choice == 2:
            chosen = second
            rejected = first
        else:
            raise ValueError("Choice must be 1 or 2")
        raw_image = Image.open(os.path.join(self.image_path, image))
        if conversations[0]["from"] == "human":
            starting_index = 0
        else:
            starting_index = 1
        prompt = ""

        assert conversations[-1]["from"] == "gpt"
        conversations = conversations[:-1]
        for conversation in conversations[starting_index:]:
            if conversation["from"] == "human":
                role_string = "USER"
            elif conversation["from"] == "gpt":
                role_string = "ASSISTANT"
            else:
                role_string = "ASSISTANT"
            if prompt == "":
                prompt += f"{self.starting_prompt}\n{role_string}:\n"
            else:
                prompt += f"{role_string}:"
            prompt += f"{conversation['value']}\n"

        prompt_ending = value_prompt(self.caption_map[image])
        prompt_chosen = prompt + f"ASSISTANT: {chosen}\n{prompt_ending}"
        prompt_reject = prompt + f"ASSISTANT: {rejected}\n{prompt_ending}"

        processed_chosen = self.processor(
            prompt_chosen,
            raw_image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )#.to(0, torch.bfloat16)
        processed_rejected = self.processor(
            prompt_reject,
            raw_image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )#.to(0, torch.bfloat16)

        new_example["input_ids_chosen"] = processed_chosen["input_ids"]
        new_example["attention_mask_chosen"] = processed_chosen["attention_mask"]
        new_example["pixel_values_chosen"] = processed_chosen["pixel_values"]

        new_example["input_ids_rejected"] = processed_rejected["input_ids"]
        new_example["attention_mask_rejected"] = processed_rejected["attention_mask"]
        new_example["pixel_values_rejected"] = processed_rejected["pixel_values"]
        new_example["length"] = processed_chosen["input_ids"].shape

        return new_example


class LlavaVisionText2TextModelTester:
    def __init__(
        self,
        parent,
        ignore_index=-100,
        image_token_index=0,
        projector_hidden_act="gelu",
        seq_length=7,
        vision_feature_select_strategy="default",
        vision_feature_layer=-1,
        text_config={
            "model_type": "llama",
            "seq_length": 7,
            "is_training": True,
            "use_input_mask": True,
            "use_token_type_ids": False,
            "use_labels": True,
            "vocab_size": 99,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 37,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 16,
            "type_sequence_label_size": 2,
            "initializer_range": 0.02,
            "num_labels": 3,
            "num_choices": 4,
            "pad_token_id": 0,
        },
        is_training=True,
        vision_config={
            "batch_size": 12,
            "image_size": 30,
            "patch_size": 2,
            "num_channels": 3,
            "is_training": True,
            "hidden_size": 32,
            "projection_dim": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 37,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "initializer_range": 0.02,
        },
    ):
        self.parent = parent
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
        self.text_config = text_config
        self.vision_config = vision_config
        self.seq_length = seq_length

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.is_training = is_training

        self.batch_size = 3
        self.num_channels = 3
        self.image_size = 336
        self.encoder_seq_length = 231

    def get_config(self):
        return LlavaConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            ignore_index=self.ignore_index,
            image_token_index=self.image_token_index,
            projector_hidden_act=self.projector_hidden_act,
            vision_feature_select_strategy=self.vision_feature_select_strategy,
            vision_feature_layer=self.vision_feature_layer,
            torch_dtype=torch.bfloat16,
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor(
            [
                self.batch_size,
                self.vision_config["num_channels"],
                self.vision_config["image_size"],
                self.vision_config["image_size"],
            ]
        )
        config = self.get_config()

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 1) + 1
        attention_mask = input_ids.ne(1).to(torch_device)
        # we are giving 3 images let's make sure we pass in 3 image tokens
        input_ids[:, 1] = config.image_token_index
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict

    def create_and_check_llava_model_fp16_forward_simple(self, config, input_ids, pixel_values, attention_mask):
        model = LlavaForConditionalGeneration(config=config)
        model.to(torch_device)
        # model.half()
        model.eval()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            vision_output = model.vision_tower(pixel_values)
            import pdb; pdb.set_trace()
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                return_dict=True,
            )
        import pdb; pdb.set_trace()
        self.parent.assertFalse(torch.isnan(output).any().item())


    def create_and_check_llava_model_fp16_forward(self, config, input_ids, pixel_values, attention_mask):
        bits = 4
        bits_and_bytes_config = BitsAndBytesConfig(
            load_in_4bit=bits == 4,
            load_in_8bit=bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_skip_modules=["mm_projector", "lm_head"],
        )
        peft_config = LoraConfig(
            r=16,
            lora_alpha=16,
            bias="none",
            task_type="CAUSAL_LM",
            lora_dropout=0.0,
            target_modules={"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"},
            # modules_to_save=["scores"],
        )
        model = LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/bakLlava-v1-hf",
            # low_cpu_mem_usage=True,
            # load_in_4bit=bits == 4,
            # load_in_8bit=bits == 8,
            # device_map={"": current_device},
            quantization_config=bits_and_bytes_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=False,

        )
        adapter_name = "lora_default"
        model = PeftModelForCausalLM(model, peft_config, adapter_name=adapter_name)
        prompt = "<image>\nUSER: What are the things I should be cautious about when I visit this place?\nASSISTANT:"
        image_file = "https://llava-vl.github.io/static/images/view.jpg"
        raw_image = Image.open(requests.get(image_file, stream=True).raw)

        hfparser = transformers.HfArgumentParser(
            (TrainingArguments)
        )
        (
            training_args,
            extra_args,
        ) = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)

        processor = AutoProcessor.from_pretrained(
            "llava-hf/bakLlava-v1-hf",
            model_max_length=training_args.model_max_length,
            padding_side="left",
            truncation_side="right"
        )

        inputs = processor(prompt, raw_image, return_tensors="pt")
        input_ids_stack = []
        attention_mask_stack = []
        pixel_values_stack = []
        train_dataset = load_dataset("zhiqings/LLaVA-Human-Preference-10K")["train"]
        print("TRAIN LENGTH", len(train_dataset))
        data_dir = "../LLaVA-RLHF/data"
        image_path = os.path.join(data_dir, "coco/train2017")
        with open(os.path.join(data_dir, "image_to_caption.json")) as f:
            caption_map = json.load(f)
        train_dataset = PreprocessDataset(train_dataset, processor, image_path, caption_map, starting_prompt)
        train_dataset, eval_dataset = split_train_into_train_and_eval(
            train_dataset=train_dataset,
            eval_size=500,
            seed=42,
        )

        # output = model(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     pixel_values=pixel_values,
        # ).logits
        data_collator = RewardDataCollatorWithPadding(
            processor.tokenizer, max_length=512
        )
        
        trainer = MultiModalRewardTrainer(
            model=model,
            tokenizer=processor,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
        )
        trainer.train()
        self.parent.assertFalse(torch.isnan(output).any().item())

    def create_and_check_llava_model_fp16_generate(self, config, input_ids, pixel_values, attention_mask):
        bits = 4
        bits_and_bytes_config = BitsAndBytesConfig(
            load_in_4bit=bits == 4,
            load_in_8bit=bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_skip_modules=["mm_projector", "lm_head"],
        )
        model = LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/bakLlava-v1-hf",
            # low_cpu_mem_usage=True,
            # load_in_4bit=bits == 4,
            # load_in_8bit=bits == 8,
            # device_map={"": current_device},
            quantization_config=bits_and_bytes_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=False,

        )
        # model.to(torch_device)
        # model.half()
        model.eval()

        output = model.generate(pixel_values=pixel_values.to(torch.bfloat16),
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
            max_new_tokens=20
        )
        self.parent.assertFalse(torch.isnan(output).any().item())




@require_torch
class LlavaForConditionalGenerationModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Model tester for `LlavaForConditionalGeneration`.
    """

    all_model_classes = (LlavaForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = {"image-to-text": LlavaForConditionalGeneration} if is_torch_available() else {}
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = True
    test_head_masking = False

    def setUp(self):
        self.model_tester = LlavaVisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LlavaConfig, has_text_modality=False)

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @require_torch_fp16
    def test_llava_model_fp16_forward(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.create_and_check_llava_model_fp16_forward_simple(config, **inputs)

    @require_torch_fp16
    def test_llava_model_fp16_generate(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.create_and_check_llava_model_fp16_generate(config, **inputs)


@require_torch
class LlavaForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = AutoProcessor.from_pretrained("llava-hf/bakLlava-v1-hf")

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model = LlavaForConditionalGeneration.from_pretrained("llava-hf/bakLlava-v1-hf", load_in_4bit=True)

        prompt = "<image>\nUSER: What are the things I should be cautious about when I visit this place?\nASSISTANT:"
        image_file = "https://llava-vl.github.io/static/images/view.jpg"
        raw_image = Image.open(requests.get(image_file, stream=True).raw)
        inputs = self.processor(prompt, raw_image, return_tensors="pt")

        EXPECTED_INPUT_IDS = torch.tensor([[1, 32000, 28705, 13, 11123, 28747, 1824, 460, 272, 1722,315, 1023, 347, 13831, 925, 684, 739, 315, 3251, 456,1633, 28804, 13, 4816, 8048, 12738, 28747]])  # fmt: skip
        self.assertTrue(torch.equal(inputs["input_ids"], EXPECTED_INPUT_IDS))

        output = model.generate(**inputs, max_new_tokens=20)
        EXPECTED_DECODED_TEXT = "\nUSER: What are the things I should be cautious about when I visit this place?\nASSISTANT: When visiting this place, there are a few things one should be cautious about. Firstly,"  # fmt: skip

        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test_llama(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model_id = "llava-hf/llava-1.5-7b-hf"

        model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", load_in_4bit=True)
        processor = AutoProcessor.from_pretrained(model_id)

        prompt = "USER: <image>\nWhat are the things I should be cautious about when I visit this place?\nASSISTANT:"
        image_file = "https://llava-vl.github.io/static/images/view.jpg"
        raw_image = Image.open(requests.get(image_file, stream=True).raw)
        inputs = processor(prompt, raw_image, return_tensors="pt").to(torch_device, torch.float16)

        output = model.generate(**inputs, max_new_tokens=900, do_sample=False)
        EXPECTED_DECODED_TEXT = "USER:  \nWhat are the things I should be cautious about when I visit this place?\nASSISTANT: When visiting this place, which is a pier or dock extending over a body of water, there are a few things to be cautious about. First, be aware of the weather conditions, as sudden changes in weather can make the pier unsafe to walk on. Second, be mindful of the water depth and any potential hazards, such as submerged rocks or debris, that could cause accidents or injuries. Additionally, be cautious of the presence of wildlife, such as birds or fish, and avoid disturbing their natural habitats. Lastly, be aware of any local regulations or guidelines for the use of the pier, as some areas may be restricted or prohibited for certain activities."  # fmt: skip

        self.assertEqual(
            processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test_llama_batched(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model_id = "llava-hf/llava-1.5-7b-hf"

        model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", load_in_4bit=True)
        processor = AutoProcessor.from_pretrained(model_id)

        prompts = [
            "USER: <image>\nWhat are the things I should be cautious about when I visit this place? What should I bring with me?\nASSISTANT:",
            "USER: <image>\nWhat is this?\nASSISTANT:",
        ]
        image1 = Image.open(requests.get("https://llava-vl.github.io/static/images/view.jpg", stream=True).raw)
        image2 = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

        inputs = processor(prompts, images=[image1, image2], return_tensors="pt", padding=True)

        output = model.generate(**inputs, max_new_tokens=20)

        EXPECTED_DECODED_TEXT = ['USER:  \nWhat are the things I should be cautious about when I visit this place? What should I bring with me?\nASSISTANT: When visiting this place, which appears to be a dock or pier extending over a body of water', 'USER:  \nWhat is this?\nASSISTANT: The image features two cats lying down on a pink couch. One cat is located on']  # fmt: skip

        self.assertEqual(processor.batch_decode(output, skip_special_tokens=True), EXPECTED_DECODED_TEXT)

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test_batch(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model = LlavaForConditionalGeneration.from_pretrained("llava-hf/bakLlava-v1-hf", load_in_4bit=True)
        # The first batch is longer in terms of text, but only has 1 image. The second batch will be padded in text, but the first will be padded because images take more space!.
        prompts = [
            "USER: <image>\nWhat are the things I should be cautious about when I visit this place? What should I bring with me?\nASSISTANT:",
            "USER: <image>\nWhat is this?\nASSISTANT:",
        ]
        image1 = Image.open(requests.get("https://llava-vl.github.io/static/images/view.jpg", stream=True).raw)
        image2 = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

        inputs = self.processor(prompts, images=[image1, image2], return_tensors="pt", padding=True)

        output = model.generate(**inputs, max_new_tokens=20)

        EXPECTED_DECODED_TEXT = ['USER:  \nWhat are the things I should be cautious about when I visit this place? What should I bring with me?\nASSISTANT: When visiting this place, there are a few things to be cautious about and items to bring along', 'USER:  \nWhat is this?\nASSISTANT: Cats']  # fmt: skip
        self.assertEqual(self.processor.batch_decode(output, skip_special_tokens=True), EXPECTED_DECODED_TEXT)

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test_llama_batched_regression(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model_id = "llava-hf/llava-1.5-7b-hf"

        # Multi-image & multi-prompt (e.g. 3 images and 2 prompts now fails with SDPA, this tests if "eager" works as before)
        model = LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-7b-hf", load_in_4bit=True, attn_implementation="eager"
        )
        processor = AutoProcessor.from_pretrained(model_id, pad_token="<pad>")

        prompts = [
            "USER: <image>\nWhat are the things I should be cautious about when I visit this place? What should I bring with me?\nASSISTANT:",
            "USER: <image>\nWhat is this?\nASSISTANT: Two cats lying on a bed!\nUSER: <image>\nAnd this?\nASSISTANT:",
        ]
        image1 = Image.open(requests.get("https://llava-vl.github.io/static/images/view.jpg", stream=True).raw)
        image2 = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

        inputs = processor(prompts, images=[image1, image2, image1], return_tensors="pt", padding=True)

        output = model.generate(**inputs, max_new_tokens=20)

        EXPECTED_DECODED_TEXT = ['USER:  \nWhat are the things I should be cautious about when I visit this place? What should I bring with me?\nASSISTANT: When visiting this serene location, one should be cautious about the weather conditions and potential', 'USER:  \nWhat is this?\nASSISTANT: Two cats lying on a bed!\nUSER:  \nAnd this?\nASSISTANT: A cat sleeping on a bed.']  # fmt: skip

        self.assertEqual(processor.batch_decode(output, skip_special_tokens=True), EXPECTED_DECODED_TEXT)

    @slow
    @require_bitsandbytes
    def test_llava_index_error_bug(self):
        # This is a reproducer of https://github.com/huggingface/transformers/pull/28032 and makes sure it does not happen anymore
        # Please refer to that PR, or specifically https://github.com/huggingface/transformers/pull/28032#issuecomment-1860650043 for
        # more details
        model_id = "llava-hf/llava-1.5-7b-hf"
        model = LlavaForConditionalGeneration.from_pretrained(model_id, load_in_4bit=True)

        processor = AutoProcessor.from_pretrained(model_id)

        # Simulate a super long prompt
        user_prompt = "Describe the image:?\n" * 200
        prompt = f"USER: <image>\n{user_prompt}ASSISTANT:"
        image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"

        raw_image = Image.open(requests.get(image_file, stream=True).raw)
        inputs = processor(prompt, raw_image, return_tensors="pt").to(torch_device, torch.float16)

        # Make sure that `generate` works
        _ = model.generate(**inputs, max_new_tokens=20)

    @slow
    @require_torch_gpu
    def test_llava_merge_inputs_error_bug(self):
        # This is a reproducer of https://github.com/huggingface/transformers/pull/28333 and makes sure it does not happen anymore
        model_id = "llava-hf/llava-1.5-7b-hf"
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).to(torch_device)

        # Simulate some user inputs
        pixel_values = torch.randn(
            (2, 3, 336, 336),
            dtype=torch.float,
            device=torch_device,
        )
        input_ids = torch.tensor(
            [
                [32001, 32001, 1, 15043, 7084, 32000, 29871, 13, 7900],
                [1, 15043, 7084, 29901, 29871, 32000, 29871, 13, 7900],
            ],
            dtype=torch.long,
            device=torch_device,
        )
        attention_mask = torch.tensor(
            [[0, 0, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]],
            dtype=torch.long,
            device=torch_device,
        )

        # Make sure that the loss is properly computed
        loss = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        ).loss
        loss.backward()

