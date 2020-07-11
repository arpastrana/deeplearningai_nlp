
# Week 2: Naive Bayes

#nlp #coursera #deeplearningai #mooc

Course 1: Natural Language Processing with Classification and Vector Spaces

## Probability and Bayes Rule

- Probabilities
- Bayes rule
- Build a Naive-Bayes classifier

Corpus of tweets
- "happy" sometimes can be labelled positive and other negative

### Probabilities
$A \to$ Positive tweet in a corpus

$$P(A) = N_{pos} / N = 13 / 20 = 0.65$$
$$P(Negative) = 1 - P(Positive) = 0.35$$

Probability in a corpus of a tweet to be positive and contain "happy" is:

$$P(A \cap B) = P(A, B) = 3/20 = 0.15$$

### Bayes Rule
- Conditional Probabilities
	
	- A tweet has 75% probability of being positive if it contains the word "happy".

	$P(A | B) = P(Positive | "happy") = 3 / 4 = 0.75$

	- Goals
		- Probability of B, given A happened.
		- Looking at the elements of A, the change that one also belongs to B

	- Bayes Rule

		- Based on conditional probabilities
		- The probability of a tweet being positive given that it contains the word happy, in terms of the probability of a tweet containing the word happy given that it is positive, is expressed as:

$$P(Positive | "happy") = P("happy" | Positive) \times \frac{P(Positive)}{P("happy")}$$

- In general form:

$$P(X | Y) = P(Y | X) \times \frac{P(X)}{P(Y)}$$

- Exercise:
	- 0.25 - positive tweets contain "happy"
	- 0.12 - all tweets contain the word "happy"
	- 0.40 - all tweets are positive
	- What is the probability of "happy to learn nlp" to be positive?


$$P(Positive|"happy") = 0.25 * \ƒrac{0.4}{0.13} = 0.77 $$

> You can calculate the probability of X given Y if you know the probability of Y given X, the probability of X, and that of Y.

## Naive Bayes Intro
- It is supervised learning.
- It is naive, because it assumes that data is independent.

- Steps
	1. Get a set of positive and negative tweets
	2. Build bag of words and count number of positive and negative instances, $N_{class}$.
	3. Find conditional probabilities of words to belong to a class, $P(w_i | class$.

	For example, this:
	
	| word | pos | neg |
	|---|---|---|
	| I | 3 | 3 | 
	| am | 3 | 3 | 
	| happy | 2 | 1 |
	| sad | 1 | 2 | 
	| $N_{class}$ | 9 | 9 | 
	
	Becomes this:
	
	| word | pos | neg |
	|---|---|---|
	| I | 0.33 | 0.33 | 
	| am | 0.33 | 0.33 | 
	| happy | 0.22 | 0.11 |
	| sad | 0.11 | 0.22 | 
	| $N_{class}$ | 1.0 | 1.0 |
	
	Words that are equally probable don't add anything to the sentiment. Those who are not are **power** words. If you get words with a 0.0 probability, this will lead to numerical errors. You would smooth the distributions to solve it.
	
- **Naive Bayes inference condition rule for binary classification**

$$\prod_{i=1}^m \frac{P(w_i | pos)}{P(w_i | neg)}$$

If result > 1, it is positive. Negative otherwise. This is called the "likelihood".

## Laplacian Smoothing
Technique to avoid probabilities to be zero for a particular word.

Usual:
$$P(w_i | class) = \frac{freq(w_i, class)}{N_{class}}$$

Becomes:
$$P(w_i | class) = \frac{freq(w_i, class) + 1}{N_{class} + V_{class}}$$

where $V_{class}$ is the number of unique words and is a term added to compensate for the $+1$ in the numerator.


## Log Likelihood

Logarithms of probabilities we have discussed. Just more convenient to work with.

### Naive Bayes Inference

For a positive-sentiment tweet, likelihood is:

$$\frac{P(pos)}{P(neg)} \prod_{i=1}^m \frac{P(w_i | pos)}{P(w_i | neg)} > 1$$

The term $\frac{P(pos)}{P(neg)}$ is called **prior ratio** and is useful when you have unbalanced dataset.

Products have the risk of bringing numerical underflow (too small values!). Thus, we resort to some logarithmic magic 🤖,

$$log(a * b) = log(a) +log(b)$$
$$log(\frac{P(pos)}{P(neg)} \prod_{i=1}^m \frac{P(w_i | pos)}{P(w_i | neg)}) = log(\frac{P(pos)}{P(neg)}) + \sum_{i=1}^m log(\frac{P(w_i | pos)}{P(w_i | neg)})$$

Or, the **log prior** + the **log likelihood**

Where, for convenience:

$$\lambda(w) = log(\frac{P(w_i | pos)}{P(w_i | neg)})$$

Originally, a value >1 would mean a positive tweet, 1 neutral, and 0 negative. With the log likelihood, a positive tweet will have a positive sign (and grow towards infinity); similarly with negative sentiments with but with a negative sign. A neutral tweet will return a value of zero.

## Training Naive Bayes
Train here means something different than gradient descent in logistic regression. 

**Steps**:
1. Collect and annotate corpus: positive and negative tweets
2. Preprocessing
	- Lowercasing
	- Removing punctuation, url, names
	- Removing stopwords
	- Stemming
	- **Tokenize** sentences (split into a list of individual words)
3. Word Count $freq(w, class)$.
4. Find conditional probabilities $P(w|class)$. Produces a table of conditional probabilities.
5. Get lambda scores, $\lambda(w)$
6. Get estimation of log prior. This value is zero if number of positive and negative tweets is equal (balanced corpus).
	 $$log(\frac{N_{pos}}{N_{neg}})$$

## Visualizing likelihoods and confidence ellipses
Jupiter notebook file (insert hyperlink later here).

Visual inspection of the tweets dataset using the Naïve Bayes features.

We will see how we can understand the log-likelihood ratio explained in the videos as a pair of numerical features that can be fed in a machine learning algorithm.

At the end of this lab, we will introduce the concept of confidence ellipse as a tool for representing the Naïve Bayes model visually.

## Testing Naive Bayes
- Predict using a Naive Bayes models 
- Using the validation set to compute model accuracy

If a word is unseen in the trained model, then the $\lambda(w)$ is zero, being treated as **neutral** words.

- Inputs:
	1. $X_{val}$
	2. $Y_{val}$
	3. $\lambda$
	4. logprior

- Score = $predict(X_{val}, \lambda, logprior)$
- Prediction = $score > 0$
- Accuracy = $\frac{1}{m}\sum_{i=1}^{m}(pred_i == Y_{val_i})$

## Applications of Naive Bayes
- Author identification: recognise whether a document was written by Shakespeare or Hemmingway.
- Spam filtering: is an email spam or not?
- Info retrieval: is a document relevant or not?
- Word disambiguation: contextual clarity of words

This method is simple, fast, and robust!

## Naive Bayes Assumptions
- Assumptions
	1. Independence between predictors and features. Words in text are independent from one another. For example:
	
			"It is sunny and hot in the desert"

		  The words "sunny" and "hot" are typically highly correlated and may lead to overestimating or under- conditional probabilities.

	2. Relative frequencies in corpus	
		
		A good dataset will have the same amount of positive and negative tweets. But this is synthetic. In real life, this is not the case. For example, Twitter bans certain very negative tweets.


## Source of errors

1. Punctuation and stop words

	Emojis are very important. Neutral words can backfire. For example:
	
		"My beloved grandmother :("
		
	Processed tweet `[beloved, grandmother]` hints to positive sentiment, which is wrong.
	
2. Word order

		"I am happy because I did not go"
		"I am not happy because I did go"
		
3. Adversarial attacks

	Sarcasm, irony and euphemisms. Machines are terrible at it.
	
		"This is a ridiculously powerful movie. I cried right through until the ending!"


