import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from tflite_runtime.interpreter import Interpreter  # type: ignore
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter  # type: ignore


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load the CSV containing `relative_time` and `temperature` columns."""
    df = pd.read_csv(csv_path)
    expected_cols = {"relative_time", "temperature"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns {expected_cols}, found {set(df.columns)}"
        )
    df = df.sort_values("relative_time").reset_index(drop=True)
    return df
 

def init_interpreter(model_path: Path) -> Interpreter:
    """Initialise a TensorFlow Lite interpreter from a model file."""
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("Model input details:")
    for d in input_details:
        print(d)
    print("\nModel output details:")
    for d in output_details:
        print(d)

    return interpreter


def predict_trend(
    interpreter: Interpreter,
    temps_window: np.ndarray,
    high_thresh: float,
    low_thresh: float,
) -> int:
    """
        0. `temperatures` – **vector** of length `window_size`
        1. `high_thresh`  – **scalar** (upper bound)
        2. `low_thresh`   – **scalar** (lower bound)

    """

    input_details = interpreter.get_input_details()

    temps_detail = input_details[0]
    expected_shape = tuple(temps_detail["shape"])  # e.g. (8,) or (1, 8)

    if len(expected_shape) == 1:
        exp_len = expected_shape[0]
    else:
        exp_len = expected_shape[-1]

    if temps_window.size < exp_len:
        pad = np.pad(temps_window, (exp_len - temps_window.size, 0), "edge")
        temps_ready = pad.astype(np.float32)
    else:
        temps_ready = temps_window[-exp_len:].astype(np.float32)

    if len(expected_shape) == 2:
        temps_ready = temps_ready.reshape(1, -1)

    interpreter.set_tensor(temps_detail["index"], temps_ready)

    hi_detail = input_details[1]
    lo_detail = input_details[2]

    def _as_scalar(value, detail):
        arr = np.array(value, dtype=np.float32)
        if detail["shape"].size == 1 and detail["shape"].tolist() == [1]:
            arr = arr.reshape(1)
        return arr

    interpreter.set_tensor(hi_detail["index"], _as_scalar(high_thresh, hi_detail))
    interpreter.set_tensor(lo_detail["index"], _as_scalar(low_thresh, lo_detail))

    interpreter.invoke()

    out = interpreter.get_tensor(interpreter.get_output_details()[0]["index"]).squeeze()
    return out.item()  # scalar 0 (safe) or 1 (alert)


def run(csv_path: Path, model_path: Path, window: int = 8):

    prev_trends = np.zeros(window, dtype=np.int32)
    df = load_data(csv_path)
    interpreter = init_interpreter(model_path)
    if window <= 0:
        input_shape = interpreter.get_input_details()[0]["shape"]
        if len(input_shape) >= 2:
            window = int(input_shape[1])
        else:
            window = 1

    temps = df["temperature"].values.astype(np.float32)
    times = df["relative_time"].values.astype(np.float32)

    statuses: List[int] = []

    high_thresh = 10.0
    low_thresh = 0.0

    print(f"Using thresholds: {low_thresh:.2f} °C – {high_thresh:.2f} °C")

    for idx in range(len(temps)):
        start = max(0, idx - window + 1)
        temp_window = temps[start : idx + 1]
        time_window = times[start : idx + 1]

        if len(temp_window) < window:
            pad_size = window - len(temp_window)
            temp_window = np.pad(temp_window, (pad_size, 0), "edge")
            time_window = np.pad(time_window, (pad_size, 0), "edge")
            pad_prev = np.full(pad_size, prev_trends[0], dtype=np.int32)
            prev_trends = np.concatenate([pad_prev, prev_trends])
        status = predict_trend(
            interpreter,
            temp_window,
            high_thresh,
            low_thresh,
        )
        prev_trends = np.roll(prev_trends, -1)
        statuses.append(int(status)) 

    statuses = np.array(statuses)

    unique_stat, counts_stat = np.unique(statuses, return_counts=True)
    print("Status counts:", {0: "alright", 1: "alert"}, dict(zip(unique_stat, counts_stat)))

    cmap = ["red" if s == 1 else "green" for s in statuses]

    plt.figure(figsize=(14, 6))
    plt.scatter(times, temps, c=cmap, s=5)

    plt.axhline(high_thresh, color="black", linestyle="--", linewidth=4,
                label=f"High threshold ({high_thresh:.1f} °C)")
    plt.axhline(low_thresh, color="black", linestyle="--", linewidth=4,
                label=f"Low threshold ({low_thresh:.1f} °C)")
    
    plt.legend(loc="best")

    plt.title("Temperature Alert Detection – Red: ALERT • Green: OK")
    plt.xlabel("Relative Time")
    plt.ylabel("Temperature (°C)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run trend detection on temperature data and plot results.")
    parser.add_argument(
        "--csv",
        type=Path,
        default="filtered_temperature_data_2.csv",
        #default="final/sample_3/251.csv",
        help="Path to the CSV file containing temperature data.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default="ALGO/model.tflite",
        help="Path to the TFLite model file.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=8,
        help="Number of consecutive samples to feed the model (sliding window).",
    )

    args = parser.parse_args()
    run(args.csv, args.model, args.window) 