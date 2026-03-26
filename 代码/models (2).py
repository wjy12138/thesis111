import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """Basic LSTM forecaster for inputs shaped as [batch, seq_len, feature_dim]."""

    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
        super().__init__()
        real_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=real_dropout,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        output = self.fc(last_time_step)
        return output


class CNNModel(nn.Module):
    """Basic 1D CNN forecaster for short-term temporal feature extraction."""

    def __init__(self, input_size, out_channels, kernel_size, pool_size, output_size):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(input_size, out_channels, kernel_size=kernel_size, padding=padding)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=pool_size)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.relu2 = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(out_channels, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        output = self.fc(x)
        return output


class CNNLSTMModel(nn.Module):
    """CNN frontend plus LSTM encoder with a direct multi-step output head."""

    def __init__(
        self,
        input_size,
        out_channels_1,
        out_channels_2,
        kernel_size_1,
        kernel_size_2,
        pool_size,
        hidden_size,
        num_layers,
        dropout,
        output_size,
    ):
        super().__init__()
        padding_1 = kernel_size_1 // 2
        padding_2 = kernel_size_2 // 2
        self.conv1 = nn.Conv1d(input_size, out_channels_1, kernel_size=kernel_size_1, padding=padding_1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels_1, out_channels_2, kernel_size=kernel_size_2, padding=padding_2)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=pool_size)

        real_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=out_channels_2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=real_dropout,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        output = self.fc(last_time_step)
        return output


class CNNLSTMSeq2SeqModel(nn.Module):
    """CNN encoder plus LSTM seq2seq decoder for autoregressive multi-step forecasts."""

    def __init__(self, input_size, out_channels, kernel_size, pool_size, hidden_size, num_layers, dropout, output_size):
        super().__init__()
        padding = kernel_size // 2
        real_dropout = dropout if num_layers > 1 else 0.0

        self.output_size = output_size
        self.conv1 = nn.Conv1d(input_size, out_channels, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=pool_size)
        self.encoder_lstm = nn.LSTM(
            input_size=out_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=real_dropout,
        )
        self.decoder_lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=real_dropout,
        )
        self.output_head = nn.Linear(hidden_size, 1)

    def _encode(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        _, hidden_state = self.encoder_lstm(x)
        return hidden_state

    def forward(self, x, decoder_start, decoder_inputs=None, teacher_forcing_ratio=0.0):
        hidden_state = self._encode(x)

        if decoder_start.dim() == 2:
            decoder_input = decoder_start.unsqueeze(-1)
        elif decoder_start.dim() == 3:
            decoder_input = decoder_start
        else:
            raise ValueError("decoder_start must have shape [batch, 1] or [batch, 1, 1].")

        use_teacher_forcing = decoder_inputs is not None and teacher_forcing_ratio > 0.0
        if decoder_inputs is not None and decoder_inputs.dim() == 3:
            teacher_targets = decoder_inputs.squeeze(-1)
        else:
            teacher_targets = decoder_inputs

        predictions = []
        hidden, cell = hidden_state

        for step_idx in range(self.output_size):
            decoder_output, (hidden, cell) = self.decoder_lstm(decoder_input, (hidden, cell))
            step_prediction = self.output_head(decoder_output[:, -1, :])
            predictions.append(step_prediction)

            if use_teacher_forcing:
                next_teacher_index = step_idx + 1
                if teacher_forcing_ratio >= 1.0:
                    if next_teacher_index < teacher_targets.size(1):
                        next_input = teacher_targets[:, next_teacher_index:next_teacher_index + 1]
                    else:
                        next_input = step_prediction
                else:
                    teacher_mask = torch.rand(step_prediction.size(0), device=step_prediction.device) < teacher_forcing_ratio
                    if next_teacher_index < teacher_targets.size(1):
                        teacher_step = teacher_targets[:, next_teacher_index:next_teacher_index + 1]
                    else:
                        teacher_step = step_prediction
                    next_input = torch.where(teacher_mask.unsqueeze(1), teacher_step, step_prediction)
            else:
                next_input = step_prediction

            decoder_input = next_input.unsqueeze(-1)

        return torch.cat(predictions, dim=1)
