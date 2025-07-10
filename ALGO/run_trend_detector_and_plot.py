import argparse
from pathlib import Path
from typing import List, Tuple

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
) -> Tuple[int, int]:
    """Run inference using the 5-input model and return `(status, numeric_trend)`.

    Inputs expected by the updated model (all 1-D):
        0 – temperatures   (float32, shape `[window]`)
        1 – times          (float32, shape `[window]`)
        2 – last_preds     (int32,   shape `[window]`)
        3 – high_threshold (float32, shape `[1]`)
        4 – low_threshold  (float32, shape `[1]`)

    Output: vector `[status, trend]` (both int32)
        status: 0 → alright, 1 → alert
        trend:  -1 cooling, 0 constant, 1 heating
    """

    input_arrays = [
        temps_window.astype(np.float32),
        np.array([high_thresh], dtype=np.float32),
        np.array([low_thresh], dtype=np.float32),
    ]

    input_details = interpreter.get_input_details()


    # Resize if necessary, then set tensors
    for idx, arr in enumerate(input_arrays):
        detail = input_details[idx]
        req_shape = tuple(detail["shape"])
        if arr.shape != req_shape:
            interpreter.resize_tensor_input(detail["index"], arr.shape)
            interpreter.allocate_tensors()
            detail = interpreter.get_input_details()[idx]
        interpreter.set_tensor(detail["index"], arr)

    interpreter.invoke()

    out = interpreter.get_tensor(interpreter.get_output_details()[0]["index"]).squeeze()
    if out.shape != (2,):
        raise RuntimeError(f"Expected output shape (2,) but got {out.shape}")

    status, trend = map(int, out)
    return status, trend


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
    trends: List[int] = []

    # Derive default thresholds (±1 °C around mean) – tweak as desired / param.

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

        status, trend = predict_trend(
            interpreter,
            temp_window,
            high_thresh,
            low_thresh,
        )

        prev_trends = np.roll(prev_trends, -1)
        prev_trends[-1] = trend
        statuses.append(status)
        trends.append(trend)

    statuses = np.array(statuses)
    trends = np.array(trends)

    unique_stat, counts_stat = np.unique(statuses, return_counts=True)
    print("Status counts:", {0: "alright", 1: "alert"}, dict(zip(unique_stat, counts_stat)))

    # Colour by status: red for alert, green for alright
    dmap = ["red" if s == 1 else "blue" if s == -1 else "grey" for s in trends]

    plt.figure(figsize=(14, 6))
    plt.scatter(times, temps, c=dmap, s=5)
    plt.title("Temperature Trend Detection – Red: Heating • Blue: Cooling • Grey: Constant")
    plt.xlabel("Relative Time")
    plt.ylabel("Temperature (°C)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    cmap = ["red" if s == 1 else "green" for s in statuses]

    plt.figure(figsize=(14, 6))
    plt.scatter(times, temps, c=cmap, s=5)
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