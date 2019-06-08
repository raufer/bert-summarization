from bunch import Bunch


config = {
    "SEQ_LEN": 3,
    "MAX_EXAMPLE_LEN": 3,
    "BATCH_SIZE": 2,
    "BUFFER_SIZE": 1,
    "VOCAB_LEN": 30522,
    "INITIAL_LR": 0.003,
    "WARMUP_STEPS": 4000,
    "LOGDIR": 'log'
}

config = Bunch(config)


