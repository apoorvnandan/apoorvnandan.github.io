---
layout: post
comments: true
title:  "Training Transformers on TPU"
excerpt: "Training transformers is now super easy thanks to HuggingFace and Tensorflow 2"
date:   2020-04-19 22:00:00
---
This post shows how clean and straightforward it is to use huggingface library and tensorflow 2 to train transformer models on GPU and TPU!

As of today (April, 2020),
this is how you fine tune a BERT model on GPU (Same code works for CPU as well).
<div style="text-align:center;"><img src="/assets/bert_gpu.png"></div>

And this is how you do the same thing on TPUs. (Few extra lines added to load the model on TPU)
<div style="text-align:center;"><img src="/assets/bert_tpu.png"></div>

Its honestly amazing how easy this is compared to the old tensorflow 1.1x flow in the original code base of BERT paper!

This abstraction and creation of clean APIs for latest deep learning techniques and models will continue to lower the barrier for using them in applications.
