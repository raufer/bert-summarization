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

By default, tensorboard objects are written to `log`

```
tensorboard --logdir log/
```

Where the train and evaluation metrics can be tracked;

#### Notes to the reader

The core of the solution is implemented; there are however some missing pieces.

Implemented:

    * Encoder (BERT)
    * Draft Decoder
    * Refined Decoder
    * Autoregressive evaluation (greedy)
    
Missing:

    * RL loss component
    * Beam-search mechanism for the draft-decoder
    * Copy mechanism


#### Configuration

| **Parameter**   | **Default** | **Description**                             |
|-----------------|-------------|---------------------------------------------|
| NUM_EPOCHS      | 4           | Number of epochs to train                   |
| BATCH_SIZE      | 10          | Batch size for each training step           |
| BUFFER_SIZE     | 1000        | Buffer size for the tf.Dataset pipeline     |
| INITIAL_LR      | 0.003       | Initial learning rate value                 |
| WARMUP_STEPS    | 4000        |                                             |
| INPUT_SEQ_LEN   | 512         | Article length to truncate                  |
| OUTPUT_SEQ_LEN  | 100         | Summary length to truncate                  |
| MAX_EXAMPLE_LEN | None        | Threshold to filter examples (articles)     |
| VOCAB_SIZE      | 30522       | Length of the vocabulary                    |
| NUM_LAYERS      | 8           | Number of layers of the Transformer Decoder |
| D_MODEL         | 768         | Base embedding dimensionality (as BERT)     |
| D_FF            | 2048        | Transformer Feed Forward Layer              |
| NUM_HEADS       | 8           | Number of heads of the transformer          |
| DROPOUT_RATE    | 0.1         | Dropout rate to use in training             |
| LOGDIR          | log         | Location to write tensorboard objects       |
| CHECKPOINTDIR   | checkpoint  | Location to write model checkpoints         |


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
``




