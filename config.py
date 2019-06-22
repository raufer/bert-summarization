from bunch import Bunch


config = {
    'NUM_EPOCHS': 4,    
    "BATCH_SIZE": 2,
    "GRADIENT_ACCUMULATION_N_STEPS": 18,
    "BUFFER_SIZE": 1,
    "INITIAL_LR": 0.0003,
    "WARMUP_STEPS": 4000,
    "INPUT_SEQ_LEN": 512,
    "OUTPUT_SEQ_LEN": 72,
    "MAX_EXAMPLE_LEN": None,
    "VOCAB_SIZE": 30522,    
    "NUM_LAYERS": 8,
    "D_MODEL": 768,
    "D_FF": 2048,    
    "NUM_HEADS": 8,
    "DROPOUT_RATE": 0.1,
    "LOGDIR": 'log',
    "CHECKPOINTDIR": 'checkpoint2'
}

config = Bunch(config)


