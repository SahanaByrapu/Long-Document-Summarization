# models.py


from sentiment_data import *
from evaluator import *

from collections import Counter
import os
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from sklearn.feature_extraction.text import CountVectorizer
from numpy import argmax

######################################
# IMPLEMENT THE SENTIMENT CLASSIFIER #
######################################



class FeedForwardNeuralNetClassifier(nn.Module):
    """
    The Feed-Forward Neural Net sentiment classifier.
    """
    def __init__(self, n_classes, vocab_size, emb_dim, n_hidden_units):
        """
        In the __init__ function, you will define modules in FFNN.
        :param n_classes: number of classes in this classification problem
        :param vocab_size: size of vocabulary
        :param emb_dim: dimension of the embedding vectors
        :param n_hidden_units: dimension of the hidden units
        """
        super(FeedForwardNeuralNetClassifier, self).__init__()
        self.n_classes = n_classes
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.n_hidden_units = n_hidden_units
        # TODO: create a randomly initialized embedding matrix, and set padding_idx as 0
        # PAD's embedding will not be trained and by default is initialized as zer0
        self.word_embeddings = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.emb_dim,padding_idx=0)
        self.linear1 =nn.Linear(self.emb_dim, self.n_hidden_units) #Linear function
        self.linear2 =nn.Linear(self.n_hidden_units, self.n_classes)# Linear function (readout)


    # TODO: implement the FFNN architecture
    # when you build the FFNN model, you will need specify the embedding size using self.emb_dim, the hidden size using self.n_hidden_units,
    # and the output class size using self.n_classes

    def forward(self, batch_inputs: torch.Tensor, batch_lengths: torch.Tensor) -> torch.Tensor:
        """
        The forward function, which defines how FFNN should work when given a batch of inputs and their actual sent lengths (i.e., before PAD)
        :param batch_inputs: a torch.Tensor object of size (n_examples, max_sent_length_in_this_batch), which is the *indexed* inputs
        :param batch_lengths: a torch.Tensor object of size (n_examples), which describes the actual sentence length of each example (i.e., before PAD)
        :return the logits outputs of FFNN (i.e., the unnormalized hidden units before softmax)
        """
        # TODO: implement the forward function, which returns the logits
        y=batch_inputs.to(torch.int64)
        x = self.word_embeddings(y)
        x = x.mean(dim=1) ## Averaging embeddings
        x = F.relu(self.linear1(x))# Non-linearity
        logits = self.linear2(x)
      
        return logits

    
    def batch_predict(self, batch_inputs: torch.Tensor, batch_lengths: torch.Tensor) -> List[int]:
        """
        Make predictions for a batch of inputs. This function may directly invoke `forward` (which passes the input through FFNN and returns the output logits)
        :param batch_inputs: a torch.Tensor object of size (n_examples, max_sent_length_in_this_batch), which is the *indexed* inputs
        :param batch_lengths: a torch.Tensor object of size (n_examples), which describes the actual sentence length of each example (i.e., before PAD)
        :return: a list of predicted classes for this batch of data, either 0 for negative class or 1 for positive class
        """
        # TODO: implement the prediction function, which could reuse the forward function 
        # but should return a list of predicted labels
        logits=self.forward(batch_inputs,batch_lengths)
        preds_proba = F.softmax(logits,dim=1)
        preds = preds_proba.argmax(dim=1)
        preds=preds.tolist()
        return preds

##################################
# IMPLEMENT THE TRAINING METHODS #
##################################

def train_feedforward_neural_net(
    args,
    train_exs: List[SentimentExample], 
    dev_exs: List[SentimentExample]) -> FeedForwardNeuralNetClassifier:
    """
    Main entry point for your modifications. Trains and returns a FFNN model (whose architecture is configured based on args)
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """

    # TODO: read in all training examples and create a vocabulary (a List-type object called `vocab`)
    vocab = [] # replace None with the correct implementation
    train_new=[]
    classes=[]
    for i in range(len(train_exs)):
        train_new+=train_exs[i].words
        classes.append(train_exs[i].label)

    #Create a Vectorizer Object
    vectorizer = CountVectorizer()
    vectorizer.fit(train_new)


    #print("Vocabulary: ", vectorizer.vocabulary_)
    voc = vectorizer.vocabulary_
    vocab= list(voc.keys())

    # add PAD and UNK as the first two tokens
    # DO NOT CHANGE, PAD must go first and UNK next 
    #(as their indices have been hard-coded in several  places)
    vocab = ["PAD", "UNK"] + vocab
    print("Vocab size:", len(vocab))
    # write vocab to an external file, so the vocab can be reloaded to index the test set
    with open("data/vocab.txt", "w") as f:
        for word in vocab:
            f.write(word + "\n")

    # indexing the training/dev examples
    indexing_sentiment_examples(train_exs, vocabulary=vocab, UNK_idx=1)
    indexing_sentiment_examples(dev_exs, vocabulary=vocab, UNK_idx=1)

    #TODO: read the command line arguments values
    epochs= args.n_epochs
    batch_size = args.batch_size
    emb_dim = args.emb_dim
    hidden_units= args.n_hidden_units
    n_classes=len(set(classes))
    num_embeddings=len(vocab)

    #TODO: define the pre-trained GloVe_path embedding
    if args.glove_path!=None:
        glove_file = args.glove_path
        embedding_dict = {}
        with open(glove_file, 'r') as f:
            for line in f:
                values = line.split()
                # get the word
                word = values[0]
                if word in vocab:
                    # get the vector
                    vector = np.asarray(values[1:], 'float32')
                    embedding_dict[word] = vector

        num_words = len(vocab) + 1
        # initialize it to 0
        embedding_matrix = np.zeros((num_words, emb_dim))

        for i in range(len(vocab)):
            if i < num_words:
                vect = embedding_dict.get(vocab[i], [])
                if len(vect) > 0:
                    embedding_matrix[i] = vect[:emb_dim]

        num_embeddings, emb_dim = embedding_matrix.shape

    # TODO: create the FFNN classifier
    model = FeedForwardNeuralNetClassifier(n_classes, num_embeddings, emb_dim, hidden_units)

   # TODO: create the word_embeddings.weight
    if(args.glove_path!= None):
        model.word_embeddings.weight.data=torch.Tensor(embedding_matrix)

    # TODO: define an Adam optimizer, using default config
    rate_learning=0.00287
    optimizer = torch.optim.Adam(model.parameters(), lr=rate_learning)
    #optimizer=optim.SGD(model.parameters(), lr=rate_learning,momentum=0.9)
    #optimizer = optim.Adagrad(model.parameters(), lr=rate_learning)
    #optimizer = optim.Adamax(model.parameters(), lr=rate_learning)
    # TODO: define CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()

    # create a batch iterator for the training data
    batch_iterator = SentimentExampleBatchIterator(train_exs, batch_size=args.batch_size, PAD_idx=0, shuffle=True)
    # training
    best_epoch = -1
    best_acc = -1
    for epoch in range(args.n_epochs):
        print("Epoch %i" % epoch)

        batch_iterator.refresh()# initiate a new iterator for this epoch

        model.train()# turn on the "training mode"
        batch_loss = 0.0
        batch_example_count = 0
        batch_data = batch_iterator.get_next_batch
        while batch_data is not None:
            batch_inputs, batch_lengths, batch_labels = batch_data
            # TODO: clean up the gradients for this batch
            optimizer.zero_grad()

            # TODO: call the model to get the logits
            logits = model.forward(batch_inputs,batch_lengths)
            #logits = logits[:, -1]
            target= torch.tensor(batch_labels,dtype=torch.long)
            # TODO: calculate the loss (let's name it `loss`, so the follow-up lines could collect the stats)
            loss = criterion(logits,target)

            # record the loss and number of examples, so we could report some stats
            batch_example_count += len(batch_labels)
            batch_loss += loss.item() * len(batch_labels)
            # TODO: backpropagation (backward and step)
            loss.backward()
            optimizer.step()   
            # get another batch
            batch_data = batch_iterator.get_next_batch

        print("Avg loss: %.5f" % (batch_loss / batch_example_count))

        # evaluate on dev set
        model.eval() # turn on the "evaluation mode"
        acc, _, _, _ = evaluate(model, dev_exs, return_metrics=True)
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            print("Secure a new best accuracy %.3f in epoch %d!" % (best_acc, best_epoch))
            # save the current best model parameters
            print("Save the best model checkpoint as `best_model.ckpt`!")
            torch.save(model.state_dict(), "best_model.ckpt")
        print("-" * 10)

    # load back the best checkpoint on dev set
    model.load_state_dict(torch.load("best_model.ckpt"))
    
    model.eval() # switch to the evaluation mode
    return model
