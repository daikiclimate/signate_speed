from .lstm import LSTMClassifier


def build_model(config):

    model = LSTMClassifier(
        input_size=config.input_size,
        lstm_hidden=config.lstm_hidden,
        n_output=config.n_output,
    )

    return model
