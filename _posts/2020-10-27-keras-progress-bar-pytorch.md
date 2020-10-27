---
layout: post
comments: true
title:  "Short Post: Keras progress bars in Pytorch code"
excerpt: "Using keras progress bars in custom python code"
date:   2020-10-27 22:00:00
---
I really like the progress bars in keras API. So, this is how to add those progress bars in any python code.

Example with pytorch training loop.
```python
import tensorflow as tf

n_epochs = 3
for epoch in range(n_epochs):
    n_batches = len(dataloader)
    print(f'Epoch {epoch+1}/{n_epochs}')
    pbar = tf.keras.utils.Progbar(target=n_batches)
    for idx, batch in enumerate(dataloader):
        train_loss = train_step(batch)
        pbar.update(idx, values=[("loss",train_loss)])
    val_loss = validate()
    pbar.update(n_batches, values=[('val_loss', val_loss)])
```
Output:
```text
Epoch 1/3
10/10 [==============================] - 2s 225ms/step - loss: 0.4698 - val_loss: 0.9111
Epoch 2/3
10/10 [==============================] - 2s 226ms/step - loss: 0.4626 - val_loss: 0.8433
Epoch 3/3
10/10 [==============================] - 2s 225ms/step - loss: 0.5835 - val_loss: 0.4997
```

Notes:  
- The metrics shown along with the progress bar are being averaged over each update.  
  So if the loss values for each batch over an epoch were [5,4,3,2,1], you would see the values [5, 4.5, 4, 3.5, 3] as the progress bar fills up.
  If you need the raw values, you need to add an argument to the instantiation of the progress bar.
  e.g.
  ```python
  pbar = tf.keras.utils.Progbar(target=n_batches, stateful_metrics=['loss'])
  ```
  This would make the progress bar not average the values corresponding to the key `loss` during each update.
- It's important to "finish" the progress bar by having the final `update` function contain the first argument as 
the `target` or total number of iterations set in the progress bar. e.g. The total in the progress bar was the number of batches or `n_batches`.
So, if you were skipping the validation step, you would still need to call 
  ```python
  pbar.update(n_batches, values=None)
  ```
  after all the training batches were processed, to finish the progress bar. And it would only show the training loss this time.  
  Failing to do so will result in incomplete progress bars for each iterations that look like this.
  ```text
  Epoch 1/3
  9/10 [==========================>...] - ETA: 0s - loss: 0.4153Epoch 2/3
  9/10 [==========================>...] - ETA: 0s - loss: 0.4637Epoch 3/3
  9/10 [==========================>...] - ETA: 0s - loss: 0.5152
   ```
