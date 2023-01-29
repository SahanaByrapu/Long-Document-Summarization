import datasets
import transformers
import pandas as pd
from datasets import Dataset

# Tokenizer
from transformers import RobertaTokenizerFast

# Encoder-Decoder Model
from transformers import EncoderDecoderModel

# Training
from seq2seq_trainer import Seq2SeqTrainer
from transformers import Seq2SeqTrainer
from transformers import TrainingArguments
from dataclasses import dataclass, field
from typing import Optional

from transformers import EncoderDecoderModel
from transformers import RobertaTokenizerFast

from torchmetrics.text.rouge import ROUGEScore
from pprint import pprint

import os
os.environ["WANDB_DISABLED"] = "true"

def process_data_to_model_inputs(batch):
  # tokenize the inputs and labels
  inputs = tokenizer(batch["Text"], padding="max_length", truncation=True, max_length=encoder_max_length)
  outputs = tokenizer(batch["Summary"], padding="max_length", truncation=True, max_length=decoder_max_length)

  batch["input_ids"] = inputs.input_ids
  batch["attention_mask"] = inputs.attention_mask
  batch["decoder_input_ids"] = outputs.input_ids
  batch["decoder_attention_mask"] = outputs.attention_mask
  batch["labels"] = outputs.input_ids.copy()

  # because RoBERTa automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`. 
  # We have to make sure that the PAD token is ignored
  batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

  return batch

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }
	
# map data correctly
def generate_summary(batch):
    # Tokenizer will automatically set [BOS] <text> [EOS]
    inputs = tokenizer(batch["Text"], padding="max_length", truncation=True, max_length=40, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")
    outputs = model.generate(input_ids, attention_mask=attention_mask)
    # all special tokens including will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    batch["pred"] = output_str

    return batch

#Main code starts from here
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token
#parameter setting
batch_size=64  #
encoder_max_length=80
decoder_max_length=40

file1 = open('test.source', 'r')
file2 = open('test.target', 'r')
lines1 = file1.readlines()

df1 = pd.DataFrame (lines1, columns = ['report'])
df1['id'] = range(1, len(df1) + 1)
#df1.index.name = "title"
#df1.info()

lines2 = file2.readlines()

df2 = pd.DataFrame (lines2, columns = ['summary'])
df2['id'] = range(1, len(df2) + 1)
#df2.index.name = "title"
#df2.info()

df = pd.DataFrame (lines2, columns = ['Summary'])
df['Text'] = df1['report']
df = df.dropna()
df.head()

train_data=Dataset.from_pandas(df[:800])
val_data=Dataset.from_pandas(df[800:900])
test_data=Dataset.from_pandas(df[900:])

#processing training data
train_data = train_data.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=["Text", "Summary"]
)
train_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

#processing validation data
val_data = val_data.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=["Text", "Summary"]
)
val_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

roberta_shared = EncoderDecoderModel.from_encoder_decoder_pretrained("roberta-base", "roberta-base", tie_encoder_decoder=True)

# set special tokens
roberta_shared.config.decoder_start_token_id = tokenizer.bos_token_id                                             
roberta_shared.config.eos_token_id = tokenizer.eos_token_id

# sensible parameters for beam search
# set decoding params                               
roberta_shared.config.max_length = 40
roberta_shared.config.early_stopping = True
roberta_shared.config.no_repeat_ngram_size = 3
roberta_shared.config.length_penalty = 2.0
roberta_shared.config.num_beams = 4
roberta_shared.config.vocab_size = roberta_shared.config.encoder.vocab_size

# load rouge for validation
rouge = datasets.load_metric("rouge")



training_args = transformers.Seq2SeqTrainingArguments(
    output_dir="./",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    predict_with_generate=True,
    #evaluate_during_training=True,
    do_train=True,
    do_eval=True,
    logging_steps=2, 
    save_steps=16, 
    eval_steps=500, 
    warmup_steps=500, 
    overwrite_output_dir=True,
    save_total_limit=1
    #fp16=True 
)

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=roberta_shared,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=val_data,
)
trainer.train()

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
#model = EncoderDecoderModel.from_pretrained("./checkpoint-6432")
model = EncoderDecoderModel.from_encoder_decoder_pretrained("roberta-base", "roberta-base", tie_encoder_decoder=True)
model.to("cuda")
batch_size = 1024

results = test_data.map(generate_summary, batched=True, batch_size=batch_size, remove_columns=["Text"])
predictions = results["pred"]
references = results["Summary"]
print(len(results))
print(results[2])

rouge = ROUGEScore()
results = rouge(predictions,references)
pprint(results)