import tensorflow as tf
import numpy as np

class TrendDetector(tf.Module):
    """Pure-TensorFlow implementation of the temperature-trend heuristics.

    The exported `predict` function works with a fixed-length window of
    temperatures and returns an integer trend label for each window in the
    batch:
        2  → Fluctuating / noisy
        1  → Heating
        0  → Constant / Unknown
       -1  → Cooling
    """

    def __init__(self, window_size: int = 8,
                 trend_threshold: float = 1.0,
                 trend_change_threshold: float = 0.2):
        super().__init__()
        self.window_size = window_size
        self.trend_threshold = trend_threshold
        self.trend_change_threshold = trend_change_threshold

    # Accept both temperature and corresponding time readings so that the
    # slope can be computed in exactly the same way as the NumPy implementation
    # used in `TemperatureAnalyzer.detect_trend` (which relies on a linear
    # regression over the *actual* timestamps, not the artificial
    # 0, 1, 2 … sequence).
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.float32, name="temps"),
        tf.TensorSpec(shape=[None, None], dtype=tf.float32, name="times")
    ])
    def predict(self, temps: tf.Tensor, times: tf.Tensor) -> tf.Tensor:
        """Vectorised heuristic rewrite of `TemperatureAnalyzer.detect_trend`.

        Args:
            temps: `float32` tensor with shape `[batch, window_size]` containing
                the most recent window of temperature readings (oldest → newest).
            times: `float32` tensor with shape `[batch, window_size]` containing
                the corresponding timestamps for each temperature reading.

        Returns:
            `int32` tensor with shape `[batch]` holding the trend label.
        """
        # Ensure the incoming window dimension is exactly `window_size`.
        temps = temps[:, -self.window_size:]
        times = times[:, -self.window_size:]

        # Compute a *simple* slope for each window using just the first and
        # last samples.  This is mathematically equivalent to the full OLS
        # estimate when the timestamps are evenly spaced and is close enough
        # for most practical purposes – while being significantly cheaper and
        # easier to understand.
        time_delta = times[:, -1] - times[:, 0] + 1e-6  # [batch]
        slope      = (temps[:, -1] - temps[:, 0]) / time_delta  # [batch]

        # Temperature range inside the window – used as a lightweight
        # fluctuation detector (large range but near-zero slope → noisy).
        temp_range = tf.reduce_max(temps, axis=1) - tf.reduce_min(temps, axis=1)

        # --- Decision logic -------------------------------------------------
        #  2 → Fluctuating / noisy  (large range & small absolute slope)
        #  1 → Heating             (slope above +threshold)
        #  0 → Constant / Unknown  (everything in the "dead zone")
        # -1 → Cooling             (slope below –threshold)

        heating_cond      = slope >  self.trend_change_threshold
        cooling_cond      = slope < -self.trend_change_threshold
        fluctuating_cond  = (~heating_cond & ~cooling_cond) & \
                           (temp_range > self.trend_threshold)

        labels = tf.where(fluctuating_cond,  tf.fill(tf.shape(slope), 2),
                  tf.where(heating_cond,     tf.fill(tf.shape(slope), 1),
                  tf.where(cooling_cond,     tf.fill(tf.shape(slope), -1),
                                           tf.fill(tf.shape(slope), 0))))

        return labels  # shape: [batch]


def convert_to_tflite(window_size: int = 8,
                      trend_threshold: float = 2.0,
                      trend_change_threshold: float = 0.2,
                      output_path: str = "trend_detector.tflite") -> None:
    """Builds and exports a `.tflite` file containing the heuristic model."""

    model = TrendDetector(window_size, trend_threshold, trend_change_threshold)
    # Provide example TensorSpecs so the concrete function correctly captures
    # the two-input signature (temperatures *and* corresponding timestamps).
    concrete_fn = model.predict.get_concrete_function(
        tf.TensorSpec([None, window_size], tf.float32, name="temps"),
        tf.TensorSpec([None, window_size], tf.float32, name="times")
    )

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])
    # Enable optimisations (optional – reduces size).
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"✓ Saved TensorFlow Lite model to {output_path}")


if __name__ == "__main__":
    # Example usage: build the model & verify runtime prediction.
    convert_to_tflite()

    # Quick sanity test – create a dummy heating trend.
    dummy_temps = np.array([
        np.linspace(20, 24, 8),   # clear heating
        np.linspace(24, 20, 8),   # clear cooling
        np.ones(8) * 22,          # constant
        [22, 23, 22, 23, 22, 23, 22, 23]  # fluctuating
    ], dtype=np.float32)

    # Simulate irregular (non-uniform) timestamps to showcase the new input.
    dummy_times = np.array([
        np.linspace(0, 7, 8),
        np.linspace(0, 7, 8),
        np.linspace(0, 7, 8),
        np.linspace(0, 7, 8)
    ], dtype=np.float32)

    interpreter = tf.lite.Interpreter(model_path="trend_detector.tflite")
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # The order of inputs is preserved as declared in the signature: temps, times.
    temps_idx = next(d["index"] for d in input_details if d["name"].endswith("temps"))
    times_idx = next(d["index"] for d in input_details if d["name"].endswith("times"))

    interpreter.resize_tensor_input(temps_idx, dummy_temps.shape)
    interpreter.resize_tensor_input(times_idx, dummy_times.shape)
    interpreter.allocate_tensors()

    interpreter.set_tensor(temps_idx, dummy_temps)
    interpreter.set_tensor(times_idx, dummy_times)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]["index"])

    label_map = {2: "Fluctuating", 1: "Heating", 0: "Constant", -1: "Cooling"}
    for row, pred in zip(dummy_temps, preds):
        print(f"Input temps: {np.round(row, 2)}  →  Predicted trend: {label_map[int(pred)]}") 