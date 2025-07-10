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
    times_window: np.ndarray,
    last_trend: int,
) -> int:
    """Run inference on a triple of inputs (temps, times & last trend).

    The updated TFLite model now expects three inputs:
        0 → temperatures   – 1-D window of floats of length *window*
        1 → times          – 1-D window of floats of length *window*
        2 → last_trend     – scalar `int32` carrying the previously predicted trend

    The output is a scalar `int32` with values:
        1  – heating   | -1 – cooling   | 0 – constant / fluctuating
    """

    input_details = interpreter.get_input_details()
    if len(input_details) != 3:
        raise RuntimeError(
            f"Expected the model to have 3 inputs (temps, times, last_trend) but got {len(input_details)}"
        )

    # Prepare the three input arrays in the same order as the model definition.
    inputs = [
        temps_window.astype(np.float32),
        times_window.astype(np.float32),
        np.array([last_trend], dtype=np.int32),
    ]

    # Iterate through each tensor, resizing if necessary (dynamic shapes) and feeding data.
    for idx, arr in enumerate(inputs):
        detail = input_details[idx]
        required_shape = tuple(detail["shape"])

        # If the interpreter uses *None* as a placeholder for dynamic dimension, treat that as compatible size.
        compatible = (
            required_shape == arr.shape
            or any(dim is None for dim in required_shape)
        )

        if not compatible:
            try:
                interpreter.resize_tensor_input(detail["index"], arr.shape)
                interpreter.allocate_tensors()
                # refresh details after allocation
                detail = interpreter.get_input_details()[idx]
            except Exception:
                try:
                    arr = arr.reshape(detail["shape"])
                except ValueError as exc:
                    raise ValueError(
                        f"Input {idx} array has shape {arr.shape} which cannot be reshaped to {detail['shape']}"
                    ) from exc

        # Ensure dtype matches
        arr = arr.astype(detail["dtype"])
        interpreter.set_tensor(detail["index"], arr)

    interpreter.invoke()

    output_detail = interpreter.get_output_details()[0]
    output_val = interpreter.get_tensor(output_detail["index"]).squeeze()

    return int(output_val)


def run(csv_path: Path, model_path: Path, window: int = 8):
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

    predictions: List[int] = []
    prev_trend: int = 0  # Unknown at start
    for idx in range(len(temps)):
        start = max(0, idx - window + 1)
        temp_window = temps[start : idx + 1]
        time_window = times[start : idx + 1]

        if len(temp_window) < window:
            pad_size = window - len(temp_window)
            pad_temp = np.full(pad_size, temp_window[0], dtype=np.float32)
            pad_time = np.full(pad_size, time_window[0], dtype=np.float32)
            temp_window = np.concatenate([pad_temp, temp_window])
            time_window = np.concatenate([pad_time, time_window])

        pred = predict_trend(interpreter, temp_window, time_window, prev_trend)
        predictions.append(pred)
        prev_trend = pred  # feed the latest prediction into the next iteration

    predictions = np.array(predictions)

    unique, counts = np.unique(predictions, return_counts=True)
    print("Trend counts:", dict(zip(unique, counts)))

    color_map = {-1: "blue", 1: "red", 0: "grey"}
    cmap = [color_map.get(p, "black") for p in predictions]

    plt.figure(figsize=(14, 6))
    plt.scatter(times, temps, c=cmap, s=5)
    plt.title("Temperature Trend Detection – Red: Heating, Blue: Cooling, Grey: Constant/Fluctuating")
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
        default="trend_predictor.tflite",
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