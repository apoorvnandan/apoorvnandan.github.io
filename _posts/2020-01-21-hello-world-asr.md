---
layout: post
comments: false
title:  "Minimal Example of End to End Speech Recognition"
excerpt: "Stripped down end to end ASR in Tensorflow"
date:   2020-01-21 22:00:00
---
## Contents
- **Audio preprocessing:** Converting raw audio into numerical features which are useful as inputs for a neural network
- **Neural Network:** A simple architecture for converting audio features into probability distributions over the possible characters in the transcript
- **CTC loss:** Calculating the loss without annotating each timestep of the audio with its corresponding character
- **Decoding:** Creating a transcript from the probability distributions for each timestep using prefix beam search and a language model

## Audio Preprocessing
You need to convert your audio into a feature matrix to feed it into your neural network. One simple way is to create spectrograms.  
```python
def create_spectrogram(signals):
    stfts = tf.signal.stft(signals, fft_length=256)
    spectrograms = tf.math.pow(tf.abs(stfts), 0.5)
    return spectrograms
````

This function computes the Short-time Fourier Transform of your audio signal and then computes the power spectrum. The output is a matrix called spectrogram. You can directly use this as your input. Other alternatives to this are filter banks and MFCCs. Audio preprocessing is a whole topic in itself. You can read about it in detail
 [here](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html).
 
## Neural Network

<img src="/assets/BASIC ASR.png">

You spectrogram has a time axis, and a frequency axis.
Your network needs to take this spectrogram as input and output the probabilities for each character ('a', 'b', 'c', ..) at each timestep in its time axis.
Here is a simple architecture.
```python
class ASR(tf.keras.Model):
    def __init__(self, filters, kernel_size, conv_stride, conv_border, n_lstm_units, n_dense_units):
        super(ASR, self).__init__()
        self.conv_layer = tf.keras.layers.Conv1D(filters,
                                                 kernel_size,
                                                 strides=conv_stride,
                                                 padding=conv_border,
                                                 activation='relu')
        self.lstm_layer = tf.keras.layers.LSTM(n_lstm_units,
                                               return_sequences=True,
                                               activation='tanh')
        self.lstm_layer_back = tf.keras.layers.LSTM(n_lstm_units,
                                                    return_sequences=True,
                                                    go_backwards=True,
                                                    activation='tanh')
        self.blstm_layer = tf.keras.layers.Bidirectional(self.lstm_layer, backward_layer=self.lstm_layer_back)
        self.dense_layer = tf.keras.layers.Dense(n_dense_units)

    def call(self, x):
        x = self.conv_layer(x)
        x = self.blstm_layer(x)
        x = self.dense_layer(x)
        return x
```
## CTC Loss
This section is crucial.  
**Why CTC?** This network attempts to predict the character at each timestep. Our labels, however, are not the characters at each timestep but just the transcription of the audio. Keep in mind that each character in the transcription may stretch across multiple timesteps. The word C-A-T will come across as C-C-C-A-A-T-T if you somehow label each timestep in the audio. Annotating your audio dataset at every 10ms is not feasible. CTC solves this problem as it does not require us to label every timestep. It takes as input the entire output probability matrix of the above neural network and the corresponding text, ignoring the position and actual offsets of each character in the transcript.
### Loss Calculation  
Suppose the ground truth label is CAT. Within these four timesteps, sequences like C-C-A-T, C-A-A-T, C-A-T-T, _-C-A-T, C-A-T-_ all correspond to our ground truth. We will calculate the probability of our ground truth by summing up the probabilities for all these sequences. The probability of a single sequence is calculated by multiplying the probabilities of its characters as per the output probability matrix. For the above sequences, the total probability comes out to be 0.0288 + 0.0144 + 0.0036 + 0.0576 + 0.0012 = 0.1056. The loss is the negative logarithm of this probability. The loss function is already implemented in TensorFlow. You can read the docs [here](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/nn/ctc_loss).

## Decoding
The output you get from the above neural network is the CTC matrix. CTC matrix gives the probability of each character in its set at each timestep. We use Prefix Beam Search to make meaningful text out of this matrix.

The set of characters in the CTC matrix has two special tokens apart from the alphabet and space character. These are blank token and end of string token.

**Purpose of blank token:** The timestep in the CTC matrix is usually small. (~10 ms) So each character of the spoken sentence stretches across multiple timesteps. For example, C-A-T becomes C-C-C-A-A-T-T. Therefore we collapse all repetition across the possible candidate strings that stand out in the CTC matrix. What about words like FUNNY where N is supposed to repeat? A blank token between the two Ns prevents them from collapsing into one without adding anything in the text. So, F-F-U-N-[blank]-N-N--Y collapses into FUNNY.

**Purpose of end-token:** End-of-string denotes the end of the spoken sentence. Decoding at timesteps after end-of-string token does not add anything to the candidate string.

### Procedure:

**Initialization:**

We have a list of candidates initially. It consists of a single blank string. The list also contains the probability of that candidate ending in a blank token and ending in a non-blank token at each timestep. The probability of the blank string ending in a blank token at time 0 is 1. The probability of it ending in a non-blank token is 0.

**Iterations:**

We take this string and add each character to it one by one. We take each extended string formed and calculate its probability of ending in a blank and non-blank token at time=1. Then we store these extended strings along with their probabilities on our list. We put these new candidates on our list and repeat the process for the next timestep.

**Case A:** If the added character is a blank token, we make no change in the candidate.

**Case B:** If the added character is a space, we multiply the probability with a number proportional to the probability of the candidate as per a language model. This prevents incorrect spellings from becoming the best candidate. So COOL will not be spelled as KUL in the final output.

**Case C:** If the added character is the same as the last character of the candidate. (candidate=FUN. character=N) We create two new candidates, FUNN and FUN. The probability of FUN is calculated from the probability of FUN ending in a blank token. The probability of FUNN is calculated using the probability of FUN ending in a non-blank token. So, if FUN does not end with the blank token, we discard the additional N instead of appending it.

**Output:**

The best candidate after all the timesteps is the output.
We make two modifications to make this process faster.
After each timestep, we discard all but the best K candidates. The candidates are sorted by the sum of their probability of ending in a blank and non-blank token.
We do not consider the characters which have their probability in the matrix below a certain threshold (~0.001).

Go through the code below for implementation details.
```python
def prefix_beam_search(ctc, 
                             alphabet, 
                             blank_token, 
                             end_token, 
                             space_token, 
                             lm, 
                             k=25, 
                             alpha=0.30, 
                             beta=5, 
                             prune=0.001):
    '''
    function to perform prefix beam search on output ctc matrix and return the best string
    :param ctc: output matrix
    :param alphabet: list of strings in the order their probabilties are present in ctc output
    :param blank_token: string representing blank token
    :param end_token: string representing end token
    :param space_token: string representing space token
    :param lm: function to calculate language model probability of given string
    :param k: threshold for selecting the k best prefixes at each timestep
    :param alpha: language model weight (b/w 0 and 1)
    :param beta: language model compensation (should be proportional to alpha)
    :param prune: threshold on the output matrix probability of a character. 
        If the probability of a character is less than this threshold, we do not extend the prefix with it
    :return: best string
    '''
    zero_pad = np.zeros((ctc.shape[0]+1,ctc.shape[1]))
    zero_pad[1:,:] = ctc
    ctc = zero_pad
    total_timesteps = ctc.shape[0]

    # #### Initialization ####
    null_token = ''
    Pb, Pnb = Cache(), Cache()
    Pb.add(0,null_token,1)
    Pnb.add(0,null_token,0)
    prefix_list = [null_token]
    
    # #### Iterations ####
    for timestep in range(1, total_timesteps):
        pruned_alphabet = [alphabet[i] for i in np.where(ctc[timestep] > prune)[0]]
        for prefix in prefix_list:
            if len(prefix) > 0 and prefix[-1] == end_token:
                Pb.add(timestep,prefix,Pb.get(timestep - 1,prefix)) 
                Pnb.add(timestep,prefix,Pnb.get(timestep - 1,prefix))
                continue  

            for character in pruned_alphabet:
                character_index = alphabet.index(character)

                # #### Iterations : Case A ####
                if character == blank_token:
                    value = Pb.get(timestep,prefix) + ctc[timestep][character_index] * (Pb.get(timestep - 1,prefix) + Pnb.get(timestep - 1,prefix))
                    Pb.add(timestep,prefix,value)
                else:
                    prefix_extended = prefix + character
                    # #### Iterations : Case C ####
                    if len(prefix) > 0 and character == prefix[-1]:
                        value = Pnb.get(timestep,prefix_extended) + ctc[timestep][character_index] * Pb.get(timestep-1,prefix)
                        Pnb.add(timestep,prefix_extended,value)
                        value = Pnb.get(timestep,prefix) + ctc[timestep][character_index] * Pnb.get(timestep-1,prefix)
                        Pnb.add(timestep,prefix,value)

                    # #### Iterations : Case B ####
                    elif len(prefix.replace(space_token, '')) > 0 and character in (space_token, end_token):
                        lm_prob = lm(prefix_extended.strip(space_token + end_token)) ** alpha
                        value = Pnb.get(timestep,prefix_extended) + lm_prob * ctc[timestep][character_index] * (Pb.get(timestep-1,prefix) + Pnb.get(timestep-1,prefix))
                        Pnb.add(timestep,prefix_extended,value) 
                    else:
                        value = Pnb.get(timestep,prefix_extended) + ctc[timestep][character_index] * (Pb.get(timestep-1,prefix) + Pnb.get(timestep-1,prefix))
                        Pnb.add(timestep,prefix_extended,value)

                    if prefix_extended not in prefix_list:
                        value = Pb.get(timestep,prefix_extended) + ctc[timestep][-1] * (Pb.get(timestep-1,prefix_extended) + Pnb.get(timestep-1,prefix_extended))
                        Pb.add(timestep,prefix_extended,value)
                        value = Pnb.get(timestep,prefix_extended) + ctc[timestep][character_index] * Pnb.get(timestep-1,prefix_extended)
                        Pnb.add(timestep,prefix_extended,value)

        prefix_list = get_k_most_probable_prefixes(Pb,Pnb,timestep,k,beta)

    # #### Output ####
    return prefix_list[0].strip(end_token)
```
This completes a bare-bones speech recognition system. You can introduce a bunch of complications to get better outputs. Bigger networks and audio preprocessing tricks help a lot. Here is the complete [code](https://github.com/apoorvnandan/speech-recognition-primer).
---
Notes:
1. The code above uses TensorFlow 2.0 and the sample audio file has been taken from the [LibriSpeech](http://www.openslr.org/12) dataset.
2. You will need to write your own batch generators to train over an audio dataset. These implementation details are not included in the code.
3. You will need to write your own language model function for the decoding part. One of the simplest implementations would be to create a dictionary of bigrams and their probabilities based on some text corpus.

References:

[1] A.Y. Hannun et al., [Prefix Search Decoding](https://arxiv.org/pdf/1408.2873v2.pdf) (2014), arXiv preprint arXiv:1408.2873, 2014  
[2] A. Graves et al., [CTC Loss](https://www.cs.toronto.edu/~graves/icml_2006.pdf) (2006), ICML 2006  
[3] L. Borgholt, [Prefix Beam Search](https://medium.com/corti-ai/ctc-networks-and-language-models-prefix-beam-search-explained-c11d1ee23306) (2018), Medium
