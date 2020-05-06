---
layout: post
comments: false
title:  "Notes on Yoshua Bengio's Keynote in ICASSP 2020"
excerpt: "Overcoming the limitations of current deep learning"
date:   2020-05-06 22:00:00
---
This consists of parts of the talk which I understood well enough to explain with minimal jargons.

Most of the talk deals with the current limitations of deep learning when it comes to discovering abstractions and performing well on out of distribution data. Deep learning performs well only on tasks known as independent and identically distributed tasks. This means the test data can be assumed to come from the same distribution as training data. (Most of the popular supervised benchmarks)

## Contents
- **Current DL vs DL 2.0**
- **Meta Transfer Objective**
- **Recurrent Independent Mechanisms**

## Current Deep Learning vs Deep Learning 2.0
The book "Thinking fast and thinking slow", talks about two intelligent systems inside our head.

**System 1:** This system is fast. It works on intuition and you cannnot consciously control it. This is similar to the current state of deep learning.

**System 2:** This system is slow. It is logical and conscious. It is reponsible for all sorts of planning, reasoning and long term executions. This is what Deep Learning can hopefully achieve in its next phase or version (Deep Learning 2.0)

There are two research projects that Bengio et. al. talked about in the direction of system 2 like models.

## I. Meta Transfer Objective 
[Link to the paper](https://arxiv.org/pdf/1901.10912.pdf)

The objective is of this research is to learn representations of knowledge (eg, sensory inputs like images and audio) such that a model that works with these representations can quickly adapt to out-of-distribution data.

### The setup
You have a training distribution which explains your training data. And you have a change in the training distribution which causes your test data to be diifferent.

Here, data is any form of observations we take in a physical world.

**Assumption #1:** Any groud truth data (observations) is a composition of certain underlying independent mechanisms at work. For example, you see an apple fall, and determine that there is gravity at play here. Even if you have never seen your laptop fall (out of distribution), you know it will fall as the same gravitational mechanism is at play.

**Assumption #2:** Change in data distribution is due to one or very few of those underlying mechanisms changing by a little bit.

### The idea
If you learn bad representations of your data, your model will need a lot of out of distribution data to adapt to it, as it cannot identify the underlying mehanisms that generate that data, and hence cannot make changes corresponding to the few mechanisms that have changes.

If you learn to convert your observations into good represnetations, that capture these underlying mechanisms and their relationships, your model would identify the mechanisms that have changed in the new out of distribution data. Therfore, it can quickly adapt to it by changing only a few parameters (corresponding to the changed underlying mechanisms).

Therefore, the speed of adapting to out of distribution data can be an objective function to chase after in order to learn good representations from your observations. 

This means updating the parameters of an encoder network such that a model working with the representations it produces from your observations can quickly adapt to out of distribution inputs.

Doing this succesfully would mean that you have captured the underlying causal mechanisms behind your observations.

This is in contrast to current Deep Learning set ups where the model has a hard time adapting to out of distribution data as it has to basically retrain itself on a large amount of it to perform well.

## II. Recurrent Independent Mechanisms
[Link to the paper](https://arxiv.org/pdf/1909.10893.pdf)

A unique neural network architecture is proposed here. It consisting of multiple independent recurrent modules that interact sometimes depeding on the input. These independent modules are not all used simultaneously. We choose a subset of these modules through an attention mechanism depending on the observations at each timestep. Different modules learn to specialize in different mechanisms and the final output is a composition of these learnt mechanisms.

This architecture is named Recurrent Independent Mechanism (RIM).

They used this architecture for several tasks, which are tough for LSTMs, Transformers and other networks dealing with sequential data. One of the tasks is explained below.

### Bouncing ball task
This task consists of N balls. They have different masses and sizes. They all move as per Newtonian laws of motion and collision. You are being given the first few time frames and you need to predict the future time frames (positions of the balls) 

As humans, we quickly realise that each ball will be moving independently most of the time as per a certain mechanism (kinematics) and will sometimes change its behavious when it comes in contact with another ball. (collision)

An LSTM network (baseline) will try to fit in the positions of all the balls in the training data and predict the future positions. If the test data has more balls than those provided in training data (out of distribution) LSTMs fail to generalise and correctly predict the movement of the balls.

RIM networks however makes each sub module inside it learn a particular behaviour (components of kinematics, collision, etc) The parameters of one module is not corrupted by the behaviours being learnt by other modules. RIMs are therefore observed to better predict the future time frames in this set up.
