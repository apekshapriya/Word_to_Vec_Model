## Aim
To create Word2vec model(create embeddings for each word in the dataset taken)

## Dataset
Taken the Wikipedia Dataset.

## Approach

Using Skip-Gram approach keeping window size 2(how it is used is shown in creation of training data)
Training samples are created as described below:
Example:
  "The quick brown fox jumps over the lazy dog."
  Here the training data will be in format:
  Input: The   Label: quick
  Input: The   Label: brown
  Input: quick Label: the
  Input: quick Label: brown
  Input: quick Label: fox
  and similarly for other words...

Here, as we are using window size of 2 and 0 skip, for each word, we take its context words(ie two words before it and two words after it). Those neighbouring words are chosen as context words as they will be used frequently with the given word. Hence in a similar context. Those context words wil decide the vector prediction of the given middle word.


# Word_to_Vec_Model
Created a model that generates an embedding word vector using skip gram approach using wikipedia dataset.

## Created 1-layer neural network model.
The model uses skip gram approach(neighbour words(context words) are predicted given a current word).
The weights between the input and hidden layer will be learnt while training and will be the required embedding vector for the given corpus.
