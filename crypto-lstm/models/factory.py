from .lstm import LSTMModel

def create_model(cfg_model, input_size: int):
    return LSTMModel(
        input_size=input_size,
        hidden_size=cfg_model.hidden_size,
        num_layers=cfg_model.num_layers,
        dropout=cfg_model.dropout,
        use_attention=cfg_model.attention,
        use_classification_head=cfg_model.classification_head,
    )
