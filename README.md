## Pretraining-Based Natural Language Generation for Text Summarization

Implementation of a abstractive text-summarization architecture, as proposed by this [paper](https://arxiv.org/pdf/1902.09243.pdf).

The solution makes use of an pre-trained language model to get contextualized representations of words; these models were training on a huge corpus of unlabelled data, e.g. BERT. 

This extends the sphere of possible applications where labelled data is scarce; since the model learns to summarize given the contextualized BERT representations, we have a good chance of generalization for other domains, even if training was done using the CNN/DM dataset.

`tensorflowhub` to load the BERT module.


#### Environment

Python: `3.6`

Tensorflow version: `1.13.1`

`requirements.txt` exposes the library dependencies

#### Run

To run the training job:

```
python train.py
```

The arguments can be changed in `config.py`.

By default, tensorboard objects are written to `log`; training checkpoints are written at the end of each epoch to `checkpoints`.

```
tensorboard --logdir log/
```

Where the train and evaluation metrics can be tracked;

Note that `train.py` does not run the validation graph (validation set) in parallel with the training job.

To run the validation job in parallel with the training:

```
python train.py --eval
```

takes a long time to build the inference graph and start the training; at the end of each epoch, it uses the model inference mode (autoregressive) to calculate the loss and ROUGE metrics; it also shows writes a random article/summary prediction.

#### Resource Limitations

Due to GPU limits, using an AWS sagemaker notebook with a `Tesla V100 (16 GB RAM)`, the batch size must be set to 2, otherwise a OOM error will be thrown.
Furthermore, the target sequence length has a huge impact in the memory resources needed during training. 75 is the value used by default.

To compensate this resource constraint we use gradient accumulation with `N` steps, i.e. we run the foward `N` steps with a batch size of 2 and accumulate the gradient; after the `N` steps we run the backward pass, updating the weights.

This gives us an effective `2N` batch size; `N` (default=12) can be controlled by changing `GRADIENT_ACCUMULATION_N_STEPS` in `config.py`.

#### Notes to the reader

The core of the solution is implemented; there are however some missing pieces.

Implemented:

    * Encoder (BERT)
    * Draft Decoder
    * Refined Decoder
    * Autoregressive evaluation (greedy)
    * Gradient accumulation
    
Missing:

    * RL loss component
    * Beam-search mechanism for the draft-decoder
    * Copy mechanism
    
Any help to implement these is appreciated.

#### Configuration

| **Parameter**                 | **Default** | **Description**                                                  |
|-------------------------------|-------------|------------------------------------------------------------------|
| NUM_EPOCHS                    | 4           | Number of epochs to train                                        |
| BATCH_SIZE                    | 2           | Batch size for each training step                                |
| GRADIENT_ACCUMULATION_N_STEPS | 12          | Number of gradient accumulate steps before applying the gradient |
| BUFFER_SIZE                   | 1000        | Buffer size for the tf.Dataset pipeline                          |
| INITIAL_LR                    | 0.003       | Initial learning rate value                                      |
| WARMUP_STEPS                  | 4000        |                                                                  |
| INPUT_SEQ_LEN                 | 512         | Article length to truncate                                       |
| OUTPUT_SEQ_LEN                | 75          | Summary length to truncate                                       |
| MAX_EXAMPLE_LEN               | None        | Threshold to filter examples (articles)                          |
| VOCAB_SIZE                    | 30522       | Length of the vocabulary                                         |
| NUM_LAYERS                    | 8           | Number of layers of the Transformer Decoder                      |
| D_MODEL                       | 768         | Base embedding dimensionality (as BERT)                          |
| D_FF                          | 2048        | Transformer Feed Forward Layer                                   |
| NUM_HEADS                     | 8           | Number of heads of the transformer                               |
| DROPOUT_RATE                  | 0.1         | Dropout rate to use in training                                  |
| LOGDIR                        | log         | Location to write tensorboard objects                            |
| CHECKPOINTDIR                 | checkpoint  | Location to write model checkpoints                              |


#### Data

To train the model we use the [CNN/DM dataset](https://www.tensorflow.org/datasets/datasets#cnn_dailymail), directly from Tensorflow Datasets.
The first time it runs, it will push the dataset from the google source (~ 500 MB).

The details on how the data is pushed and prepared can be found at `data/load.py`


##### Debug

Track GPU memory usage with:

```
watch -n 2 nvidia-smi
```

System RAM usage with:

```
watch -n 2 cat /proc/meminfo
```

`report_tensor_allocations_upon_oom` is set to `True` so that we can see which variables
are filling up the memory.

```
run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
...
sess.run(..., options=run_options)
```




