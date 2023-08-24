---
layout: post
title: paying attention to transformers
date: 2023-08-15 10:14:00-0400
description: getting to transformers through machine translation and attention
categories: attention deep-learning transformers neural-machine-translation
bibliography: 2023-08-15-attention.blog.bib
giscus_comments: false
related_posts: false
toc:
  sidebar: left
---

We explore transformers by understanding the motivation for neural machine translation and attention. Let's go!

<!-- ---
local and global attention
hard and soft attention
encoder and decoder architecture - rnn with lstm/gru
output of each decoder instance is a vocabulary sized vector
make the diagram more expressive/informative
different score functions for computing attention weights
is the alignment vector a vector of scalar value? probably a vector for matrix (hidden states) vector (attention) multiplication
bahdanau soft attention is like global attention.
--- -->

---

## Neural Machine Translation

It's 2014 and we begin with the task of Neural Machine Translation (NMT). Unlike statistical machine translation, NMT aims at building a single neural network that can be jointly learned to maximize the translation performance. These models generally have an encoder-decoder architecture where the encoder converts the input sequence to a fixed length vector, which the decoder uses as its context for generating the output sequence.

<div class="col-sm mt-3 mt-md-0">
    {% include figure.html path="assets/img/blog/b1-rnn.png" class="img-fluid rounded z-depth-1" zoomable=true %}
</div>
<div class="caption">
    Encoder-Decoder architecture for neural machine translation
</div>

Here, the encoder units are in blue, decoder units in green, input sequences in red and output sequences in purple. Input $$x_{0}, x_{1} ... x_{n}$$ is fed sequentially to the encoder with the output of the encoder at time $$t_{n}$$ depending on hidden state $$h_{n-1}$$ and input $$x_{n}$$. The output of the encoder $$h_{n}$$ or $$s_{0}$$ captures the input as a fixed length vector and is fed as input to the decoder. The decoder generates the first output token given $$s_{0}$$ and `<sos>` token. The output of the decoder at $$t_{1}$$ ie. $$y^{`}_{1}$$ is fed as input to the decoder at time step $$t_{2}$$ along with the previous hidden state. This is done till a `<eos>` token is generated. Now, during training, the generated input can be gibberish, so we need a good way to compute the loss and train the network. Instead of providing the previous time-step output, we provide the ground-truth label $$y_{i-1}$$ as input for time-step $$t_{i}$$.

The drawback of compressing all of the input information in a fixed length vector is that the network fails to perform well for long sentences. The performance deteriorates rapidly as the length of the sentence increases <d-cite key="bahdanau2014neural"></d-cite>.

## Attention-based Models

Attention-based models address the above problem by <i>looking</i> at the entire input sequence before generating a new word. As mentioned in <d-cite key="bahdanau2014neural"></d-cite>, each time the model generates a new word, it (soft-)searches for a set of positions in a source sentence where most relevant information is concentrated. The model then predicts the target word based on the context vectors and the previous generated target word. This is shown in the figure below (blue block). Let's explore this a bit.

<div class="col-sm mt-3 mt-md-0">
    {% include figure.html path="assets/img/blog/b1-rnn-attention.png" class="img-fluid rounded z-depth-1" zoomable=true %}
</div>
<div class="caption">
    Encoder-Decoder architecture with attention
</div>

### Attention block

Let's explore the attention block in a bit more detail. The inputs to this block while decoding at time $$t$$ are the encoder hidden states $$(h_1, h_2, ... h_n)$$ and the previous decoder hidden state $$s_{t-1}$$. The output is a context-vector $$c_t$$ which is input along with the input token to the decoder. This is shown in the figure below

<div class="col-sm mt-3 mt-md-0">
    {% include figure.html path="assets/img/blog/b1-attention-block.png" class="img-fluid rounded z-depth-1" zoomable=true %}
</div>
<div class="caption">
    Attention block
</div>

First, we compute scores ($$e_{ij}$$) using the alignment model.

\begin{equation}
e_{ij} = a(s_{j-1}, h_j)
\end{equation}

It tells us how well the inputs around position $$j$$ and the output at position $$i$$ match. Then we pass the scores through a softmax function to compute the weight $$\alpha_{ij}$$ of each $$h_j$$.

\begin{equation}
\alpha_{ij} = \dfrac{exp(e_{ij})}{\sum_{k=1}^{T_x} exp(e_{ik})}
\end{equation}

The context vector $$c_i$$ is then computed as the weighted sum of the annotations $$h_i$$

\begin{equation}
c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_i
\end{equation}

Although in the diagram above, we show only for one context vector, this operation is done for each step of the decoder generation phase. Section 3.1 of <d-cite key="bahdanau2014neural"></d-cite> has in detail description of what these values mean and how one can intuitively think of them.

### Global v/s Local attention

In <d-cite key="luong2015effective"></d-cite>, the authors explore two different types of attention mechanisms. When you attend to all the words in the input, its called global or soft attention. On the other hand, when the attention mechanism focuses on a small window of context, its called local attention. Hard attention is a modification to local, where the window size is one ie. we focus only on one hidden state to generate the context vector. While less expensive during inference, the hard attention model is non-differentiable and requires more complicated techniques such as variance reduction or reinforcement learning to train. We will explore this topic in detail later.

## Self-Attention

One of the most important building block leading upto the transformer is self-attention.

## Transformers