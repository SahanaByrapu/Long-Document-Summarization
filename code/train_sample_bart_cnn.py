import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np
from datasets import load_dataset,load_metric
import nltk
import torchmetrics
from torchmetrics.text.rouge import ROUGEScore
from sklearn.model_selection import train_test_split
nltk.download('punkt')

max_input = 1024
max_target = 128
batch_size = 2
model_checkpoints = "facebook/bart-large-cnn"
metric =load_metric('rouge')

import pandas as pd

def process_file(file1,file2):

    file1 = open(file1, 'r')
    
    lines1= file1.readlines()

    df1 = pd.DataFrame (lines1, columns = ['report'])
    df1['id'] = range(1, len(df1) + 1)
    #df1.index.name = "title"
    df1.info()

    file2 = open(file2, 'r')
    lines2 = file2.readlines()

    df2 = pd.DataFrame (lines2, columns = ['summary'])
    df2['id'] = range(1, len(df2) + 1)
    #df2.index.name = "title"
    df2.info()

    #df1.head()

    df = pd.merge(df1, df2, on='id', how='left').dropna()
    #df['id'] = range(1, len(df) + 1)
    #df.head()
    return df

t=process_file("test.source","test.target")
#test=process_file("test.source","test.target")
#val=process_file("val.source","val.target")
tr, test = train_test_split(t, test_size=0.1)
train, val = train_test_split(tr, test_size=0.2)

train.to_json('train.json', orient='records')
test.to_json('test.json', orient='records')
val.to_json('val.json', orient='records')

#train.head()
#test.head()
#val.shape()

print("before tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoints)

#Check this with BERT Tokenizer later

def preprocess_data(data_to_process):
  #get all the reports
  inputs = [report for report in data_to_process['report']]
  
  #tokenize the report
  model_inputs = tokenizer(inputs,  max_length=max_input, padding='max_length', truncation=True) 
  
  #tokenize the summaries
  with tokenizer.as_target_tokenizer():
    targets = tokenizer(data_to_process['summary'], max_length=max_target, padding='max_length', truncation=True)
    
  #set labels
  model_inputs['labels'] = targets['input_ids']
  #return the tokenized data
  #input_ids, attention_mask and labels
  return model_inputs

#Load the data

#data = load_dataset("json", data_files="train.json")
#test_data = load_dataset("json", data_files="test.json")

#Running the data through the preprocess function
print("loading data")
data_files = {"train": "train.json", "val": "val.json"}
data = load_dataset("./",data_files=data_files)

tokenize_data = data.map(preprocess_data, batched = True)

#tokenize_data['test']

tokenize_data['train']

print("after tokenizer")
#Load the model

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoints)

#####################
# metrics
# compute rouge for evaluation 
#####################

def compute_rouge(pred):
  predictions, labels = pred
  #decode the predictions
  decode_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
  #decode labels
  decode_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

  #compute results
  res = metric.compute(predictions=decode_predictions, references=decode_labels, use_stemmer=True)
  #get %
  res = {key: value.mid.fmeasure * 100 for key, value in res.items()}

  pred_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
  res['gen_len'] = np.mean(pred_lens)

  return {k: round(v, 4) for k, v in res.items()}

#collator to create batches. It preprocess data with the given tokenizer
collator = transformers.DataCollatorForSeq2Seq(tokenizer, model=model)

#Train on new data

args = Seq2SeqTrainingArguments(
    'report-summ', #save directory
    evaluation_strategy='epoch',
    learning_rate=2e-5, # Need to update the learning rate to the one mentioned in the base paper
    per_device_train_batch_size=3,
    per_device_eval_batch_size= 3,
    gradient_accumulation_steps=2, #Need to update to 32 gradient steps as per the base paper
    weight_decay=0.01, #Need to update weight_decay as per the base paper
    save_total_limit=2,
    num_train_epochs=3,
    predict_with_generate=True,
    eval_accumulation_steps=3,
    fp16=True #available only with CUDA
    )

trainer = Seq2SeqTrainer(
    model, 
    args,
    train_dataset=tokenize_data['train'],
    eval_dataset=tokenize_data['val'],
    data_collator=collator,
    tokenizer=tokenizer,
    compute_metrics=compute_rouge
)

#Running the Train function
print("training")
trainer.train()
print("training done")


test_data = load_dataset("json",data_files="./test.json")
print("type of test:",type(test_data))
#test_tokenizer=tokenizer(test_data,max_length=max_input,padding='max_length',truncation=True)
#print("type of test-tokenize:",type(test_tokenizer))
tokenize_data_test = test_data.map(preprocess_data, batched = True)
#one_test = tokenize_data_test['train'][0]
#print("type of test:",type(tokenize_data))
print("id of first test data report:",tokenize_data_test['train']['id'])
index = tokenize_data_test['train']['id']

#Predicting the Test data
raw_pred, _, _ = trainer.predict(tokenize_data_test['train'])

print("type of raw-pred:",type(raw_pred))
print("prediction of first test data:",tokenizer.decode(raw_pred[0]))

#Predicting the Test data
#raw_pred, _, _ = trainer.predict(tokenize_data['test'])

#raw_pred, _, _ = trainer.predict([test_tokenizer])

#raw_pred, _, _ = trainer.predict(model_inputs)

#print("type of raw-pred:",type(raw_pred))
#print(tokenizer.decode(raw_pred[0]))

j = 0;
with open("output2.txt", "w") as txt_file:
    for line in raw_pred:
        i = index[j]
        txt_file.write(tokenizer.decode(line) + "," + str(i) + "\n")
        j = j+1

#f = open("out2.txt", "a")
#for x in raw_pred:
#    f.write(tokenizer.decode(x));
#f.close()

#print(test_data)
#print(Object.keys(test_data)[0])