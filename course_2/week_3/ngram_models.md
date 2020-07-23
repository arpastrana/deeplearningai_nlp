# Autocomplete and Language Models

## Ngram Language Models

- Corpus: Large database of words. For instance, all books from one author.
- Language model:

Estimates the probabilities of word sequences
Applicable with the most likely suggestion, like with autocomplete.

- Text corpus: A language model + a sentence = next best words
- Other applications: Speech recognition and correction.
- Smoothing: A technique to account for not existing words.
- Model evaluation: The perplexity metric, for example.

## Ngrams and Probabilities

- Ngram: A sequence of words that appears together (order matters). Punctuation is kept.
- Unigram: A sequence of **unique** words.
- Bigram: A sequence of **pairs of words**.
- Trigram: A sequence of **triplets of words**.

## Sequence Probabilities

- Notation

  - A word is $w$
  - The corpus length is $m$
  - $w_i^{m}$ is the word $i$ from corpus of length $m$

  - Unigram:
    $$P(w_1) = \frac{count(w_1)}{m}$$
    Where $m$ is the length of the corpus.

  - Bigram: 
    $$P(w_2 | w_1) = \frac{count(w_1, w_2)}{count(w_1)}$$
    Note that every word must be followed by another word

  - Trigram: 
    $$P(w_3 | w_1^{2}) = \frac{count(w_1^{2}, w_3)}{count(w_1^{2})}$$
    Where $w_1^{2}$ is the number of appearances of $w_1, w_2$ pairs i the corpus.

  - Ngram:
    $$P(w_n | w_{i}^{n-i}) = \frac{count(w_{i}^{n-i}, w_n)}{count(w_{i}^{n-i})}$$
    Where $count(w_{i}^{n-i}) = count(w_{i}^{n})$

- Conditional probabilities

$$P(B|A) = \frac{P(A,B)}{P(A)} \to P(A,B) = P(A)P(B|A)$$
$$P(A,B,C,D) = P(A)P(B|A)P(C|A,B)P(D|A,B,C)$$

## Words not in corpus

A corpus cannot contain all words in the world. We have to approximate.
The more words we concatenate together, the less likely it is to find them in
sequence in the corpus (because of the law of small numbers).
Therefore, we have to adjust the probabilities in a sequence so that the probability
of the next word is only given by its single predecessor instead of the whole
sequence of preceding words.
This is a *Markovian* assumption: only the last word matters.

Equations are adjusted as follows:

- Ngram:
$$P(w_n|w_i^{n-i}) \to P(w_n|w_{n-1})$$
$$P(w_1^n) = \product_{i=1}^{n}P(w_i | w_{i-1})$$

In retrospect, Naive bayes doesn't account word history as the ngram method
does.

## Starting and ending sentences

We should introduce new symbols: $<s>$ to start a sentence, and $</s>$ to end
it.

To start a sentence:

- Bigram:
  $$P(w_1) \to P(w_1|w_{<s>})$$
- Trigram:
  $$P(w_1) \to P(w_1|w_{<s><s>})$$

To end a sentence:

- Ngram:
  $$P(w_n) \to P(</s>|w_n)$$

We account only for oune $</s>$ ending token, regarles of $n$.

## The ngram language model

- Elements in the mix:

  - A count matrix
    Rows are the set of (n-1)-grams. Columns, the set of words in the corpus.

  - A probability matrix
    Divides every cell in the count matrix by its sum row.

  - A language model
    From the probability matrix, a language model is created. A next word
    prediction is done by finding the word with the largest probability given
    the word preceding it, e.g. $argmax(P(w_{n-1}))$

  - Log probabilities
    The natural logarithm is applied to the values in the probability matrix to
    avoid underflow.

  - A generative language model, sentence-by-sentence
    1. Choose a start word $(<s>, w_1)$
    2. Select the most likely word from the previous word from the prob matrix.
    3. Continue until hitting $(w_n, </s>)$

## Language model evaluation

We typically need test and train datasets. The wsj datasets contains 40 million
words, for example. For small corpora, the split is like: 80% train, 10%
validation, and 10% testing. For large corpora, like for NLP applications, the
proportions change: 98% train, 1% validation, and 1% testing.

To process the dataset, we usually take either continuous test or random short
sequences.

To measure how well a trained language model model performs, we can use the
perplexity metric, which is a measure of complexity. This concept is close to
that of entropy.

$$PP(W) = (s_1, s_2, ..., s_m)^{\frac{-1}{m}}$$

Text written by real humans have low perplexity scores (20-60 for English).

The perplexity of a bigram model is:

$$PP(W) = \sqrt_m{\product_{i=1}^{m} \product_{j=1}^{|S|}
\frac{1}{P(w_j^i|w_{j-i}^i)}$$
Where $w_j^i$ is the j-th word of the i-th sentence.

The log perplexity of a bigram model, concatenating all sentences, is defined as:

$$log(PP(W)) = \frac{-1}{m}{\sum{i=1}^{m}}log({P(w_i|w_{i-1})})$$

Resulting values between 4.3 and 5-9 are reasonable for the latter.

Additionally, only compare perplexity of models that have been trained on the
same vocabulary.

## Out-of-vocabulary words

How to deal with unknown words? We introduce a proxy word, $<UNK>$.

1. Create a vocabulary $V$.
2. Replace unknown words with $<UNK>$.
3. Run an ngram model.

How to create a vocabulary?

- Criteria:
  - Heuristic: use a minimum word frequency filter.
  - max|V| heuristic: include only words until a maximum vocabulary size is met.

Be careful to use $<UNK>$ sparingly! Otherwise it will lead to inaccurate
autocomplete models.

## Smoothing

This technique is to account for missing ngrams in the training corpus and to
avoid $\frac{0}{0}$ division issues. Typical smoothing approaches are:

- Laplacian (add-one):
  $$P(w_n|w_{n-1}) = \frac{count(w_{n-1}, w_n) + 1}{count(w_{n-1}) + V}$$
- K-smoothing (add-k):
  $$P(w_n|w_{n-1}) = \frac{count(w_{n-1}, w_n) + k}{count(w_{n-1}) + k * V}$$
- Other: knesser-key, good-turing, etc.

## Backoff

In an ngram is missing, use a (n-1)-gram as replacement to find its probability.
You can apply this recursively. For example: I want to find "I want tea", but I
cannot. Then I try with "I want" instead. If this exists in my corpus, I stop
and take this probability as the fix. Otherwise, go and try with "I".

A *"stupid"* backoff model, has come up with an experimental constant value of
$0.4$. What for, though?

## Interpolation

Use the combined weights of different ngrams. The restriction is that they
should add up to 1.0.
