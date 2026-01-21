import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from sleepecg import (
    evaluate,
    extract_features,
    load_classifier,
    prepare_data_pytorch,
    print_class_balance,
    read_mesa,
    read_shhs,
    save_classifier,
    set_nsrr_token,
)


class Torch_mesa(nn.Module):
    """

    Neural network intended to mimic the existing keras models.

    The model consists of:
        A normalization layer
        A fully connected linear layer (fc)
        Two gru layers (gru1, gru2)
        Another fully connected  linear layer as the output layer (output).
    In addition, features with the specified mask_value are omitted. During training, the
    cross-entropy loss is measured and the model is trained using a RMS propagation
    optimizer.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, mask_value=-1.0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.fc = nn.Linear(input_dim, 64)
        self.gru1 = nn.GRU(64, hidden_dim, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        self.output = nn.Linear(hidden_dim * 2, output_dim)
        self.mask_value = mask_value

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor with shape (batch_size, seq_len, input_dim

        Returns
        -------
        torch.Tensor
            Output tensor after forward pass with shape (batch_size, seq_len, output_dim)
        """
        x = torch.where(x == self.mask_value, torch.zeros_like(x), x)

        x = self.layer_norm(x)

        x = F.relu(self.fc(x))

        x, _ = self.gru1(x)
        x, _ = self.gru2(x)

        x = self.output(x)

        return x


set_nsrr_token("your-token-here")

TRAIN = True  # set to False to skip training and load classifier from disk

# silence warnings (which might pop up during feature extraction)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="HR analysis window too short"
)

if TRAIN:
    print("‣  Starting training...")
    print("‣‣ Extracting features...")
    records = list(read_mesa(offline=False, data_dir=r"D:\SleepData"))

    feature_extraction_params = {
        "lookback": 240,
        "lookforward": 270,
        "feature_selection": [
            "hrv-time",
            "hrv-frequency",
            "recording_start_time",
            "age",
            "gender",
        ],
        "min_rri": 0.3,
        "max_rri": 2,
        "max_nans": 0.5,
    }

    features_train, stages_train, feature_ids = extract_features(
        tqdm(records),
        **feature_extraction_params,
        n_jobs=-1,
    )

    print("‣‣ Preparing data for Pytorch...")
    stages_mode = "wake-sleep"
    features_train_pad, stages_train_pad = prepare_data_pytorch(
        features_train,
        stages_train,
        stages_mode,
    )
    print_class_balance(stages_train_pad, stages_mode)

    print("‣‣ Defining model...")
    model = Torch_mesa(
        input_dim=features_train_pad.shape[2],
        hidden_dim=8,
        output_dim=stages_train_pad.shape[-1],
        mask_value=-1.0,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)

    print("‣‣ Training model...")
    train_dataset = TensorDataset(features_train_pad, stages_train_pad)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    model.train()
    for epoch in range(25):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)

            N, T, C = outputs.size()
            batch_y = batch_y.argmax(dim=-1)
            loss = criterion(outputs.view(N * T, C), batch_y.view(N * T))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/25, Loss: {epoch_loss / len(train_loader):.4f}")

    print("‣‣ Saving classifier...")
    save_classifier(
        name="ws-pytorch-mesa",
        model=model,
        stages_mode=stages_mode,
        feature_extraction_params=feature_extraction_params,
        mask_value=-1,
        classifiers_dir="./classifiers",
    )

print("‣  Starting testing...")
print("‣‣ Loading classifier...")
clf = load_classifier("ws-pytorch-mesa", "./classifiers")
model = clf.model

print("‣‣ Extracting features...")
shhs = list(read_shhs(offline=False))

features_test, stages_test, feature_ids = extract_features(
    tqdm(shhs),
    **clf.feature_extraction_params,
    n_jobs=-1,
)

print("‣‣ Evaluating classifier...")
features_test_pad, stages_test_pad, sample_weight = prepare_data_pytorch(
    features_test,
    stages_test,
    clf.stages_mode,
)

model.eval()

test_dataset = TensorDataset(features_test_pad, stages_test_pad)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
y_pred = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        test_output = model(batch_x)
        predictions = torch.argmax(test_output, dim=-1)
        y_pred.append(predictions)

y_pred = torch.cat(y_pred, dim=0)
evaluate(stages_test_pad, y_pred, clf.stages_mode)
