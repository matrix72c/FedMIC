class Config:
    """
    General configuration
    """
    train_data_path = "data/ml-100k.rating"
    test_data_path = "data/ml-100k.test.negative"
    mf_dim = 32
    mlp_layer_size = 128
    learning_rate = 0.001
    batch_size = 128
    epochs = 5
    dropout = 0
    device = 'cuda'
    rounds = 20000
    top_k = 10
    sample_size = 100
