from transformers import PegasusForConditionalGeneration, PegasusTokenizer, Trainer, TrainingArguments
import torch
from torchmetrics.text.rouge import ROUGEScore
from pprint import pprint

file1 = open('test.source', 'r')
file2 = open('test.target', 'r')
lines1 = file1.readlines()
lines2 = file2.readlines()

tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-large')
model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-large')#("./checkpoint-400")
#model = EncoderDecoderModel.from_encoder_decoder_pretrained("roberta-base", "roberta-base", tie_encoder_decoder=True)
#model.to("cuda")
batch_size = 128

batch = tokenizer(lines1, truncation=True, padding="longest", return_tensors="pt")#.to(device)
translated = model.generate(**batch)
results = tokenizer.batch_decode(translated, skip_special_tokens=True)

#results = test_data.map(generate_summary, batched=True, batch_size=batch_size, remove_columns=["Text"])
predictions = results#["pred"]
references = lines2#results["Summary"]
print(len(results))
print(results[2])

rouge = ROUGEScore()
results = rouge(predictions,references)
pprint(results)
