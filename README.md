## Pretraining-Based Natural Language Generation for Text Summarization

Implementation of a abstractive text-summarization architecture, as proposed by this [paper](https://arxiv.org/pdf/1902.09243.pdf).

The solution makes use of an pre-trained language model to get contextualized representations of words; these models were training on a huge corpus of unlabelled data, e.g. BERT. 

This extends the sphere of possible applications where labelled data is scarce; since the model learns to summarize given the contextualized BERT representations, we have a good chance of generalization for other domains, even if training was done using the CNN/DM dataset.


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
    * Draft Decoder (matrix-form)
    * Draft Decoder (greedy autoregressive form)
    * Refined Decoder (iterative-form)
    * Refined Decoder (matrix-form)
    * Refined Decoder (greedy autoregressive form)    
    
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
| SEQ_LEN         | 100         | Sequence length to use                      |
| MAX_EXAMPLE_LEN | 512         | Threshold to filter examples                |
| VOCAB_SIZE      | 30522       | Length of the vocabulary                    |
| NUM_LAYERS      | 8           | Number of layers of the Transformer Decoder |
| D_MODEL         | 768         | Base embedding dimensionality (as BERT)     |
| D_FF            | 2048        | Transformer Feed Forward Layer              |
| NUM_HEADS       | 8           | Number of heads of the transformer          |
| DROPOUT_RATE    | 0.1         | Dropout rate to use in training             |
| LOGDIR          | log         | Location to write tensorboard objects       |
| CHECKPOINTDIR   | checkpoint  | Location to write model checkpoints         |



