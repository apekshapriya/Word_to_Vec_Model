## Aim
To create Word2vec model(create embeddings for each word in the dataset taken)

## Dataset
Wikipedia Dataset.

## Approach

Using Skip-Gram approach keeping window size 2(how it is used is shown in creation of training data)

Training samples are created as described below

Example--
Suppose the source text taken is--
"The quick brown fox jumps over the lazy dog."

Here the training data will be in format-

1) Input- The                  
   Label- quick
2) Input- The                  
   Label- brown
3) Input- quick                
   Label- the
4) Input- quick                
   Label- brown
5) Input- quick
   Label- fox

And similarly for other words...

Here, as we are using window size of 2 and 0 skip, for each word, we take its context words(ie two words before it and two words after it). Those neighbouring words are chosen as context words as they will be used frequently with the given word. Hence in a similar context. Those context words wil decide the vector prediction of the given middle word.

 # Model Details:
 
 The model architecture is a single layer neural network with input input vector feeded to input layer, a hidden layer  of 300 units with liniear activation function and an output layer with softmax function. The weights between the input and hidden layer is learnt while training and will be the required embedding vector for the corpus.

## Conclusion

The implementation of this word2vec model from scratch helps in having a clear understanding of roots of natural language processing.
