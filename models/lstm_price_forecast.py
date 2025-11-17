import argparse
import os
import json
from datetime import datetime
import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# --------------------
# Repro & device
# --------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------
# Data loading
# --------------------
def load_bars(csv_path: str, feature_cols=None):
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    feature_cols = feature_cols or ["open", "high", "low", "close", "volume"]
    need = set(["timestamp"] + feature_cols)
    if not need.issubset(set(df.columns)):
        missing = need - set(df.columns)
        raise ValueError(f"CSV missing columns: {missing}")

    # timestamp parsing
    ts = df["timestamp"].astype(str)
    if ts.str.match(r"^\d{10}$").all():
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(None)
    elif ts.str.match(r"^\d{13}$").all():
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df = df[["timestamp"] + feature_cols].dropna().reset_index(drop=True)
    return df

def chronological_splits(df: pd.DataFrame, train_ratio=0.70, val_ratio=0.15):
    n = len(df)
    t_end = int(n * train_ratio)
    v_end = int(n * (train_ratio + val_ratio))
    return df.iloc[:t_end], df.iloc[t_end:v_end], df.iloc[v_end:]

class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.X[idx: idx + self.seq_len]
        y_tgt = self.y[idx + self.seq_len]  # predict next step after window
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor([y_tgt], dtype=torch.float32)

def fit_scale(train_df, val_df, test_df, feature_cols, target_col="close"):
    xsc = MinMaxScaler()
    ysc = MinMaxScaler()

    Xtr = xsc.fit_transform(train_df[feature_cols].values)
    Xva = xsc.transform(val_df[feature_cols].values)
    Xte = xsc.transform(test_df[feature_cols].values)

    ytr = ysc.fit_transform(train_df[[target_col]].values).ravel()
    yva = ysc.transform(val_df[[target_col]].values).ravel()
    yte = ysc.transform(test_df[[target_col]].values).ravel()
    return (Xtr, ytr), (Xva, yva), (Xte, yte), xsc, ysc

# --------------------
# Model
# --------------------
class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)      # [B, T, H]
        last = out[:, -1, :]       # [B, H]
        return self.head(last)     # [B, 1]

# --------------------
# Plotting (all figures saved to out_dir)
# --------------------
def plot_diagnostics(timestamps, y_true, y_pred, out_prefix, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    residuals = y_true - y_pred
    plt.figure()
    plt.hist(residuals, bins=40)
    plt.title("Residuals (y_true - y_pred)")
    plt.xlabel("Residual"); plt.ylabel("Count")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{out_prefix}_residuals_hist.png")); plt.close()

    ape = np.abs((y_true - y_pred) / (y_true + 1e-8)) * 100.0
    plt.figure()
    plt.hist(ape, bins=40)
    plt.title("Absolute Percentage Error (%)")
    plt.xlabel("APE %"); plt.ylabel("Count")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{out_prefix}_ape_hist.png")); plt.close()

    plt.figure()
    plt.scatter(y_true, y_pred, s=8)
    plt.title("True vs Predicted Close")
    plt.xlabel("True"); plt.ylabel("Predicted")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{out_prefix}_true_vs_pred_scatter.png")); plt.close()

    plt.figure()
    plt.plot(timestamps, y_true, label="True")
    plt.plot(timestamps, y_pred, label="Pred")
    plt.title("Test Period: True vs Predicted Close")
    plt.xlabel("Time"); plt.ylabel("Close")
    plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{out_prefix}_true_vs_pred_line.png")); plt.close()

# --------------------
# Training / Eval (per-seq_len)
# --------------------
def train_one(
    seq_len, Xtr, ytr, Xva, yva, Xte, yte, xsc, ysc, timestamps_te,
    input_size, device, args, out_dir
):
    ds_tr = SeqDataset(Xtr, ytr, seq_len)
    ds_va = SeqDataset(Xva, yva, seq_len)
    ds_te = SeqDataset(Xte, yte, seq_len)

    if len(ds_tr) < 1 or len(ds_va) < 1 or len(ds_te) < 1:
        raise ValueError(f"Not enough samples for seq_len={seq_len}.")

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=True,
                       pin_memory=(device.type == "cuda"))
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       pin_memory=(device.type == "cuda"))
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                       pin_memory=(device.type == "cuda"))

    model = LSTMForecaster(
        input_size=input_size,
        hidden_size=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    best_state = None
    wait = 0

    for ep in range(1, args.epochs + 1):
        # Train
        model.train()
        tr_loss = 0.0
        for xb, yb in dl_tr:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optim.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(dl_tr.dataset)

        # Val (scaled MSE)
        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                pred = model(xb)
                loss = criterion(pred, yb)
                vl_loss += loss.item() * xb.size(0)
        vl_loss /= len(dl_va.dataset)

        print(f"[seq={seq_len}] Epoch {ep:02d}/{args.epochs} | train {tr_loss:.6f} | val {vl_loss:.6f}")

        if vl_loss < best_val - 1e-7:
            best_val = vl_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f"[seq={seq_len}] Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test predictions (scaled)
    model.eval()
    preds_s, ys_s = [], []
    with torch.no_grad():
        for xb, yb in dl_te:
            xb = xb.to(device, non_blocking=True)
            pred = model(xb).cpu().numpy().ravel()
            preds_s.append(pred)
            ys_s.append(yb.cpu().numpy().ravel())
    preds_s = np.concatenate(preds_s)
    ys_s = np.concatenate(ys_s)

    # invert scale
    y_pred = ysc.inverse_transform(preds_s.reshape(-1, 1)).ravel()
    y_true = ysc.inverse_transform(ys_s.reshape(-1, 1)).ravel()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100.0
    metrics = {"MAE": mae, "RMSE": rmse, "MAPE%": mape, "val_mse": best_val}

    base = f"lstm_seq{seq_len}"

    # Save predictions CSV into out_dir
    pred_path = os.path.join(out_dir, f"{base}_test_predictions.csv")
    ts_out = timestamps_te.iloc[seq_len:].reset_index(drop=True)
    pd.DataFrame({"timestamp": ts_out, "y_true_close": y_true, "y_pred_close": y_pred}).to_csv(pred_path, index=False)

    # Save plots into out_dir
    plot_diagnostics(ts_out, y_true, y_pred, out_prefix=base, out_dir=out_dir)

    # Save model weights into out_dir
    model_path = os.path.join(out_dir, f"{base}_best.pt")
    torch.save(model.state_dict(), model_path)

    return metrics, model_path, pred_path, model

# --------------------
# Multi-step forecasting (naive autoregressive)
# --------------------
def autoregressive_forecast(model, last_window_X, steps, device):
    """
    Make multi-step forecasts by feeding each prediction back in place of 'close'
    and reusing other features from the last row (naive approach).
    Assumes X features are [open, high, low, close, volume] scaled.
    """
    model.eval()
    preds = []
    x_win = last_window_X.clone().to(device)  # [1, T, F]
    for _ in range(steps):
        with torch.no_grad():
            y_s = model(x_win).cpu().numpy().ravel()[0]  # scaled close
        preds.append(y_s)
        # shift window and append a new row reusing last row but with predicted close
        new_row = x_win[0, -1, :].clone()
        new_row[3] = torch.tensor(y_s)  # index 3 -> 'close' in FEATS
        x_win = torch.cat([x_win[:, 1:, :], new_row.view(1, 1, -1)], dim=1)
    return np.array(preds)

# --------------------
# Main
# --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to BTC OHLCV CSV")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[90], help="Sequence lengths to try, e.g. 30 60 90")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--horizon", type=int, default=1, help="Multi-step horizon for final forecast from last window")
    parser.add_argument("--base-out", type=str, default="lstm_price_forecast_results",
                        help="Base directory; a unique per-run folder will be created inside")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # --- per-run directory (timestamped) ---
    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_out = args.base_out
    run_dir = os.path.join(base_out, run_stamp)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run output dir: {run_dir}")

    # Save run configuration for reproducibility
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump({
            "csv": args.csv,
            "seq_lens": args.seq_lens,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "hidden": args.hidden,
            "layers": args.layers,
            "dropout": args.dropout,
            "lr": args.lr,
            "horizon": args.horizon,
            "device": str(device),
            "timestamp": run_stamp
        }, f, indent=2)

    feature_cols = ["open", "high", "low", "close", "volume"]
    df = load_bars(args.csv, feature_cols=feature_cols)

    # splits
    df_tr, df_va, df_te = chronological_splits(df, train_ratio=0.70, val_ratio=0.15)

    # Save split datasets for inspection
    split_dir = os.path.join(run_dir, "data_splits")
    os.makedirs(split_dir, exist_ok=True)

    df_tr.to_csv(os.path.join(split_dir, "train_split.csv"), index=False)
    df_va.to_csv(os.path.join(split_dir, "val_split.csv"), index=False)
    df_te.to_csv(os.path.join(split_dir, "test_split.csv"), index=False)

    print(f"\n Saved split CSVs to: {split_dir}")
    print(f"Train set: {len(df_tr)} rows | {df_tr['timestamp'].iloc[0]} → {df_tr['timestamp'].iloc[-1]}")
    print(f"Val set:   {len(df_va)} rows | {df_va['timestamp'].iloc[0]} → {df_va['timestamp'].iloc[-1]}")
    print(f"Test set:  {len(df_te)} rows | {df_te['timestamp'].iloc[0]} → {df_te['timestamp'].iloc[-1]}")
    
    (Xtr, ytr), (Xva, yva), (Xte, yte), xsc, ysc = fit_scale(df_tr, df_va, df_te, feature_cols, target_col="close")

    results = []
    best = {"seq_len": None, "val_mse": float("inf"), "model": None, "model_path": None}

    for seq_len in args.seq_lens:
        print(f"\n===== Training with sequence length = {seq_len} =====")
        metrics, model_path, pred_path, model = train_one(
            seq_len, Xtr, ytr, Xva, yva, Xte, yte, xsc, ysc, df_te["timestamp"],
            input_size=len(feature_cols), device=device, args=args, out_dir=run_dir
        )
        print(f"[seq={seq_len}] Test metrics: "
              f"MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, MAPE%={metrics['MAPE%']:.2f}, "
              f"best_val_mse={metrics['val_mse']:.6f}")
        results.append((seq_len, metrics, model_path, pred_path, model))

        if metrics["val_mse"] < best["val_mse"]:
            best.update({"seq_len": seq_len, "val_mse": metrics["val_mse"], "model": model, "model_path": model_path})

    # Summary file
    summary_lines = ["=== Summary (sorted by validation MSE) ==="]
    for seq_len, metrics, mpath, ppath, _ in sorted(results, key=lambda x: x[1]["val_mse"]):
        line = (f"seq={seq_len:3d} | val_mse={metrics['val_mse']:.6f} | MAE={metrics['MAE']:.4f} "
                f"| RMSE={metrics['RMSE']:.4f} | MAPE%={metrics['MAPE%']:.2f} | model={mpath} | preds={ppath}")
        print(line)
        summary_lines.append(line)
    with open(os.path.join(run_dir, "summary.txt"), "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    # Final forecast from full data with best seq_len
    best_seq = best["seq_len"]
    if best_seq is None:
        print("No model trained; exiting.")
        return

    X_all = xsc.transform(df[feature_cols].values)
    last_window = torch.tensor(X_all[-best_seq:], dtype=torch.float32).unsqueeze(0)

    # horizon=1 -> next-day (single-step). horizon>1 -> multi-step autoregressive
    if args.horizon <= 1:
        best["model"].eval()
        with torch.no_grad():
            next_scaled = best["model"](last_window.to(device)).cpu().numpy().ravel()[0]
        next_close = ysc.inverse_transform(np.array([[next_scaled]])).ravel()[0]
        msg = (f"\nBest seq_len = {best_seq} (val_mse={best['val_mse']:.6f})\n"
               f"Last date in data: {df['timestamp'].iloc[-1]}\n"
               f"Predicted NEXT-DAY close: {next_close:.2f}")
        print(msg)
        with open(os.path.join(run_dir, "next_day_prediction.txt"), "w") as f:
            f.write(msg + "\n")
    else:
        preds_scaled = autoregressive_forecast(best["model"], last_window, args.horizon, device)
        preds = ysc.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
        start_date = df["timestamp"].iloc[-1]
        print(f"\nBest seq_len = {best_seq} (val_mse={best['val_mse']:.6f})")
        print(f"Multi-step forecast horizon={args.horizon} days starting after {start_date}:")
        for i, p in enumerate(preds, 1):
            print(f"t+{i}: {p:.2f}")
        multi_path = os.path.join(run_dir, f"lstm_seq{best_seq}_multi_step_forecast.csv")
        pd.DataFrame({"step_ahead": np.arange(1, args.horizon + 1), "pred_close": preds}).to_csv(
            multi_path, index=False
        )
        print(f"Saved multi-step forecast to {multi_path}")

if __name__ == "__main__":
    main()

