import tensorflow as tf
import numpy as np


class TrendDetectorTF(tf.Module):
    """
        •  1 → Heating             (`avg_diff` > `trend_change_threshold`)
        • -1 → Cooling             (`avg_diff` < -`trend_change_threshold`)
        •  0 → Constant / Unknown  (otherwise)
    """

    def __init__(self, window_size: int = 8,
                 trend_threshold: float = 2.0,
                 trend_change_threshold: float = 0.2):
        super().__init__()
        self.window_size = window_size
        self.trend_threshold = trend_threshold
        self.trend_change_threshold = trend_change_threshold

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None], dtype=tf.float32, name="temps"),
            tf.TensorSpec(shape=[None, None], dtype=tf.float32, name="times"),
        ]
    )
    def predict(self, temps: tf.Tensor, times: tf.Tensor) -> tf.Tensor:
        """Stateless trend prediction based on the most recent window."""
        temps = temps[:, -self.window_size:]
        times = times[:, -self.window_size:]

        diffs = temps[:, 1:] - temps[:, :-1]        # Shape: [batch, window_size-1]
        avg_change = tf.reduce_mean(diffs, axis=1)  # Shape: [batch]

        max_abs_change = tf.reduce_max(tf.abs(diffs), axis=1)
        is_fluctuating = max_abs_change > self.trend_threshold

        heating = tf.ones_like(avg_change, dtype=tf.int32)      # +1
        cooling = -tf.ones_like(avg_change, dtype=tf.int32)     # -1
        constant = tf.zeros_like(avg_change, dtype=tf.int32)    #  0

        base_label = tf.where(
            avg_change > self.trend_change_threshold,
            heating,
            tf.where(
                avg_change < -self.trend_change_threshold,
                cooling,
                constant,
            ),
        )

        fluctuating_label = tf.fill(tf.shape(base_label), 2)
        final_label = tf.where(is_fluctuating, fluctuating_label, base_label)
        return final_label


def convert_to_tflite(window_size: int = 8,
                      trend_threshold: float = 2.0,
                      trend_change_threshold: float = 0.2,
                      output_path: str = "trend_detector_times.tflite") -> None:
    """Exports the TrendDetectorTF model to a TFLite file."""

    model = TrendDetectorTF(window_size, trend_threshold, trend_change_threshold)
    concrete_fn = model.predict.get_concrete_function(
        tf.TensorSpec(shape=[None, window_size], dtype=tf.float32, name="temps"),
        tf.TensorSpec(shape=[None, window_size], dtype=tf.float32, name="times"),
    )

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Optional: size & perf.
    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"✓ Saved TensorFlow Lite model to {output_path}")


if __name__ == "__main__":
    convert_to_tflite()

    dummy_times = np.tile(np.arange(8, dtype=np.float32), (4, 1))
    dummy_temps = np.array([
        np.linspace(20, 24, 8),                     # Heating
        np.linspace(24, 20, 8),                     # Cooling
        np.ones(8) * 22,                            # Constant
        [22, 23, 22, 23, 22, 23, 22, 23],           # Fluctuating
    ], dtype=np.float32)

    interpreter = tf.lite.Interpreter(model_path="trend_detector_times.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Identify input indices by name for clarity
    temps_idx = next(d["index"] for d in input_details if d["name"].endswith("temps"))
    times_idx = next(d["index"] for d in input_details if d["name"].endswith("times"))

    interpreter.resize_tensor_input(temps_idx, dummy_temps.shape, strict=True)
    interpreter.resize_tensor_input(times_idx, dummy_times.shape, strict=True)
    interpreter.allocate_tensors()

    interpreter.set_tensor(temps_idx, dummy_temps)
    interpreter.set_tensor(times_idx, dummy_times)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]["index"])

    label_map = {2: "Fluctuating", 1: "Heating", 0: "Constant", -1: "Cooling"}
    for row, pred in zip(dummy_temps, preds):
        print(f"Input: {np.round(row, 2)}  →  Predicted trend: {label_map[int(pred)]}") 