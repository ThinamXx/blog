---
title: "Word Vectors"
author: "Thinam Tamang"
categories: [machine learning, word vectors, word embeddings, transformers, deep learning]
date: "2022-10-15"
---

![](./Word.png)

## **Word Vectors**  
Word vectors are also called *word embeddings* or neural word representations because these whole bunch of words are represented in a high dimensional vector space and they are embedded into that space. They are also called as a distibuted representation.  
Word vectors means having a vector for each word type i.e both for context and outside, which are initialized randomly and those vectors are progressively updated by using iterative algorithms so that they can do better job at predicting which words appear in the context of other words. 


### **Distributional Semantics**  
It states that a word's meaning is given by the words that frequently appear close-by. When a word *w* appears in a text, it's context is the set of words that appear nearby within a fixed-size window. 


## **Word2Vec**  
Word2Vec is a framework for learning word vectors developed by Mikolov et al. 2013. The idea behind Word2Vec is that we have a large corpus or body of text and every word in a fixed-vocabulary is represented by a vector. We have to go through each position *t* in the text, which has a center word *c* and context words *o* and we have to use the *similarity of the word vectors* for *c* and *o* to calculate the probability of *o* given *c* or vice versa. We have to keep adjusting the word vectors to maximize this probability. Word2Vec model maximizes the objective function by putting similar words nearby in high dimensional vector space.  
Two model variants: 
- *Skip Grams:* Predict context words given center word.
- *Continuous Bag of Words:* Predict center word from context words. 

**Main Idea of Word2Vec**
- Start with random word vectors. 
- Iterate through each word in the whole corpus. 
- Try to predict the surrounding words using word vectors. Try and predict what words surrounds the center word by using the probability distribution that is defined in terms of the dot product between the word vectors for the center word and the context words. 
- Updating the vectors so that they can predict the actual surrounding words better and better. 

**Key Points:**
- Word2Vec model actually ignores the position of words.
- Taking a *log likelihood* turns all of the products into sums which decreases the computational complexity. 
- A dot product is a natural measure for similarity between words. If two words have a larger dot product, that means they are more similar. 
- The simple way to avoid negative probabilities is to apply exponential function. 

**Training Methods:**
- To train a model, we gradually adjust parameters to minimize a loss. 
- Theta represents all the model parameters, in one long vector. We optimize these parameters by walking down the gradient. 

**Bag of Words Model:**
- Bag of words models are the models that don't pay attention to words order or position, it doesn't matter if the word is near to the center word or a bit further away on the left or right. The probability estimation will be the same at each position.  
In bag of words, we have outside word vector and center word vector, which undergoes dot product followed by softmax activation function. 

### **Optimization: Gradient Descent**  
- To learn good word vectors, we have a cost function *J($\theta$)*.
- Gradient descent is an algorithm to minimize the *J($\theta$)* by changing *$\theta$*. 
- Gradient Descent is an iterative learning algorithm that learns to maximize the *J($\theta$)* by changing the *$\theta$*.
- From the current value of *$\theta$*, calculate the gradient of *J($\theta$)*, then take small step in the direction of negative gradient to gradually move down towards the minimum. 

**Problems of Gradient Descent**
- *J($\theta$)* is a function of all windows in the corpus which is often billions. So, actually working out *J($\theta$)* or the gradient of *J($\theta$)* would be extremely expensive because we have to iterate over the entire corpus. 
- We would wait a very long before making a single update. 

### **Stochastic Gradient Descent**  
Stochastic gradient descent is a very simple modification of the gradient descent algorithm. Rather than working out an estimate of the gradient based on the entire corpus, we simply take one center word or a small batch like 32 center words, and we work out the estimate of the gradient based on them.  
Stochastic gradient descent is kind of noisy and bounces around as it makes progress, it actually means that in complex networks it learns better solution. So, it can do much more quickly and much better. 

### **Softmax Function**  
The softmax function will take any *R* in vector and turns it into the range *0* and *1*. The name of the *softmax function* comes from the fact that it's sort of like a *max* because the exponential function gives more emphasis to the big contents in different dimensions of calculating the similarity. The softmax function takes some numbers and returns the whole probability distribution. 

### **Co-occurence Vector**
- Vectors increase in size with the vocabulary. 
- Very high dimensional and requires a lot of storage though sparse. 
- Subsequent classification models have sparsity issues which makes models less robust. 

## **GloVe**
- Fast training. 
- Scalable to huge corpus. 
- Good performance even with small corpus and small vectors. 

GloVe model unify the thinking between the co-occurence matrix models and the neural models by being someway similar to the neural models but actually calculated on top of a co-occurence matrix count. 