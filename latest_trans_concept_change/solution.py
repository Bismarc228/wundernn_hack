import numpy as np
import torch
import joblib
from collections import deque

try:
    from .utils import DataPoint
    from .model import TransformerModel
except ImportError:
    from utils import DataPoint
    from model import TransformerModel
from pathlib import Path
local_path = Path(__file__).parent
# --- КЛАСС ДЛЯ ПРЕДСКАЗАНИЙ ---

class PredictionModel:
    def __init__(self):
        self.device = torch.device('cpu')
        self.sequence_length = 100 
        self.n_features = 32 
        self.scaler = joblib.load(local_path / 'scaler.joblib')

        self.model = TransformerModel(
            input_dim=self.n_features,
            d_model=128,
            nhead=8,
            num_encoder_layers=4,
            dim_feedforward=512,
            dropout=0.15,
            use_rms_norm=True,
            use_rotary_pos_emb=True
        ).to(self.device)

        self.model.load_state_dict(torch.load(local_path / 'model.pth', map_location=self.device))
        self.model.eval()

        self.current_seq_ix = None
        self.history = deque(maxlen=self.sequence_length)

    def predict(self, data_point: DataPoint) -> np.ndarray | None:

        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.history.clear()

        self.history.append(data_point.state)

        if not data_point.need_prediction:
            return None

        if len(self.history) < self.sequence_length:
            return np.zeros(self.n_features)

        input_data = np.array(self.history)

        scaled_input = self.scaler.transform(input_data)

        input_tensor = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            scaled_prediction = self.model(input_tensor)

        scaled_prediction_np = scaled_prediction.cpu().numpy()
        prediction = self.scaler.inverse_transform(scaled_prediction_np)

        return prediction.squeeze(0)

if __name__ == "__main__":
    import os 
    print(os.getcwd())
    print(local_path)
    model = PredictionModel()
    print(model.predict(DataPoint(seq_ix=1, step_in_seq=0, state=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), need_prediction=True)))