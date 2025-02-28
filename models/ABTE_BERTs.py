from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from seqeval.metrics import accuracy_score, f1_score
import numpy as np
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
# fmt: off
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataa.processing import tokenize_and_align_labels
# fmt: on


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = f1_score(true_predictions, true_labels)
    return {"F1-score": results}


ds = load_dataset("thainq107/abte-restaurants")
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
preprocessed_ds = ds.map(tokenize_and_align_labels, batched=True)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
id2label = {
    0: "O",
    1: "B-Term",
    2: "I-Term"
}
label2id = {
    "O": 0,
    "B-Term": 1,
    "I-Term": 2
}

model = AutoModelForTokenClassification.from_pretrained(
    "distilbert/distilbert-base-uncased",
    num_labels=3, id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir="abte-restaurants-distilbert-base-uncased",
    logging_dir="logs",
    learning_rate=2e-5,
    per_device_train_batch_size=256,
    per_device_eval_batch_size=256,
    num_train_epochs=100,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="F1-score",
    # report_to="wandb",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=preprocessed_ds["train"],
    eval_dataset=preprocessed_ds["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

model_path = "./saved_model"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
