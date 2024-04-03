import json
import copy
import gc
import os
from pathlib import Path

import torch
import numpy as np
from transformers.tokenization_utils import PreTrainedTokenizerBase
from transformers.models.deberta_v2 import DebertaV2ForTokenClassification, DebertaV2TokenizerFast
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.data.data_collator import DataCollatorForTokenClassification
from datasets import Dataset, DatasetDict, concatenate_datasets

os.environ["WANDB_PROJECT"] = "pii-detect-deberta"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

TRAINING_MODEL_PATH = "microsoft/deberta-v3-large"
TRAINING_MAX_LENGTH = 3072
EVAL_MAX_LENGTH = 3072
CONF_THRESH = 0.9
LR = 2.5e-5
LR_SCHEDULER_TYPE = "cosine"
NUM_EPOCHS = 50
BATCH_SIZE = 4
EVAL_BATCH_SIZE = 8
GRAD_ACCUMULATION_STEPS = 16 // BATCH_SIZE
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
AMP = True
FREEZE_EMBEDDING = False
FREEZE_LAYERS = 6
N_SPLITS = 4
NEGATIVE_RATIO = 0.3  # down sample ratio of negative samples in the training set
OUTPUT_DIR = "output"
Path(OUTPUT_DIR).mkdir(exist_ok=True)


# In[37]:


args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    bf16=True,
    learning_rate=LR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
    report_to="wandb",
    save_strategy="epoch",
    save_total_limit=1,
    greater_is_better=True,
    load_best_model_at_end=True,
    overwrite_output_dir=True,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,
    save_total_limit = 1,
)

with Path("/workspace/test-dl/train.json").open("r") as f:
    original_data = json.load(f)

with Path("/workspace/test-dl/mpware_mixtral8x7b_v1.1-no-i-username.json").open("r") as f:
    extra_data = json.load(f)
print("MPWARE's datapoints: ", len(extra_data))

all_labels = [
    'B-EMAIL', 'B-ID_NUM', 'B-NAME_STUDENT', 'B-PHONE_NUM', 'B-STREET_ADDRESS', 'B-URL_PERSONAL', 'B-USERNAME', 'I-ID_NUM', 'I-NAME_STUDENT', 'I-PHONE_NUM', 'I-STREET_ADDRESS', 'I-URL_PERSONAL', 'O'
]
id2label = {i: l for i, l in enumerate(all_labels)}
label2id = {v: k for k, v in id2label.items()}
target = [l for l in all_labels if l != "O"]

class CustomTokenizer:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, label2id: dict, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __call__(self, example: dict) -> dict:
        # rebuild text from tokens
        text, labels, token_map = [], [], []

        for idx, (t, l, ws) in enumerate(
            zip(example["tokens"], example["provided_labels"], example["trailing_whitespace"])
        ):
            text.append(t)
            labels.extend([l] * len(t))
            token_map.extend([idx]*len(t))

            if ws:
                text.append(" ")
                labels.append("O")
                token_map.append(-1)

        text = "".join(text)
        labels = np.array(labels)

        # actual tokenization
        tokenized = self.tokenizer(
            "".join(text),
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length
        )

        token_labels = []

        for start_idx, end_idx in tokenized.offset_mapping:
            # CLS token
            if start_idx == 0 and end_idx == 0:
                token_labels.append(self.label2id["O"])
                continue

            # case when token starts with whitespace
            if text[start_idx].isspace():
                start_idx += 1

            token_labels.append(self.label2id[labels[start_idx]])

        length = len(tokenized.input_ids)

        return {**tokenized, "labels": token_labels, "length": length, "token_map": token_map}


tokenizer = DebertaV2TokenizerFast.from_pretrained(TRAINING_MODEL_PATH)
train_encoder = CustomTokenizer(tokenizer=tokenizer, label2id=label2id, max_length=TRAINING_MAX_LENGTH)
eval_encoder = CustomTokenizer(tokenizer=tokenizer, label2id=label2id, max_length=EVAL_MAX_LENGTH)

ds = DatasetDict()

for key, data in zip(["original", "extra"], [original_data, extra_data]):
    ds[key] = Dataset.from_dict({
        "full_text": [x["full_text"] for x in data],
        "document": [str(x["document"]) for x in data],
        "tokens": [x["tokens"] for x in data],
        "trailing_whitespace": [x["trailing_whitespace"] for x in data],
        "provided_labels": [x["labels"] for x in data],
    })

class ModelInit:
    def __init__(
        self,
        checkpoint: str,
        id2label: dict,
        label2id: dict,
        freeze_embedding: bool,
        freeze_layers: int,
    ) -> None:
        self.model = DebertaV2ForTokenClassification.from_pretrained(
            checkpoint,
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
        for param in self.model.deberta.embeddings.parameters():
            param.requires_grad = False if freeze_embedding else True
        for layer in self.model.deberta.encoder.layer[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False
        self.weight = copy.deepcopy(self.model.state_dict())

    def __call__(self) -> DebertaV2ForTokenClassification:
        self.model.load_state_dict(self.weight)
        return self.model

model_init = ModelInit(
    TRAINING_MODEL_PATH,
    id2label=id2label,
    label2id=label2id,
    freeze_embedding=FREEZE_EMBEDDING,
    freeze_layers=FREEZE_LAYERS,
)

negative_idxs = [i for i, labels in enumerate(ds["original"]["provided_labels"]) if not any(np.array(labels) != "O")]
exclude_indices = negative_idxs[int(len(negative_idxs) * NEGATIVE_RATIO):]

args.run_name = f"deberta-v3-run"
args.output_dir = os.path.join(OUTPUT_DIR, f"{args.run_name}")
original_ds = ds["original"].select([i for i in range(len(ds["original"])) if i not in exclude_indices])
train_ds = concatenate_datasets([original_ds, ds["extra"]])
train_ds = train_ds.map(train_encoder, num_proc=2)
trainer = Trainer(
    args=args,
    model_init=model_init,
    train_dataset=train_ds,
    eval_dataset=None,
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16),
)
trainer.train()

del trainer
gc.collect()
torch.cuda.empty_cache()