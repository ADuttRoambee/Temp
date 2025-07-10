# Nothing beyond TensorFlow and the standard library is required for model build
import tensorflow as tf
import os



def build_trend_tflite_model(window_size=8,
                              trend_threshold=1.0,
                              trend_change_threshold=0.1,
                              model_path="ALGO/model.tflite"):
    """Build (or reuse) a small TFLite model for trend detection."""

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[window_size], dtype=tf.float32, name="temperatures"),
        tf.TensorSpec(shape=[],            dtype=tf.float32, name="high_thresh"),
        tf.TensorSpec(shape=[],            dtype=tf.float32, name="low_thresh")
    ])
    def _trend_fn(temperatures, high_thresh, low_thresh):
        """TensorFlow implementation of the enhanced trend & alert logic."""

        # Use a simple evenly-spaced time base to fit a line
        times = tf.cast(tf.range(window_size), tf.float32) * 100.0

        # Linear-regression slope (same maths as before but fewer tensors)
        mean_t   = tf.reduce_mean(times)
        mean_temp = tf.reduce_mean(temperatures)
        slope = tf.math.divide_no_nan(
            tf.reduce_sum((times - mean_t) * (temperatures - mean_temp)),
            tf.reduce_sum(tf.square(times - mean_t))
        )

        diffs      = temperatures[1:] - temperatures[:-1]
        avg_change = tf.reduce_mean(diffs)

        # Detect rapid flip-flopping → fluctuation
        # Original implementation could never reach its fluctuation threshold
        # given an 8-sample window, so we fold it into a constant to slim the graph.
        # --- Numeric trend ----------------------------------------------------
        heating_cond = tf.logical_or(slope >  trend_change_threshold, avg_change >  trend_threshold / 8)
        cooling_cond =tf.logical_or(slope < -trend_change_threshold, avg_change < -trend_threshold / 8)

        numeric_trend = tf.where(heating_cond, 1,
                            tf.where(cooling_cond, -1, 0))  # int32 after cast below
        numeric_trend = tf.cast(numeric_trend, tf.int32)

        current_temp  = temperatures[-1]
        above_thresh  = current_temp > high_thresh
        below_thresh  = current_temp < low_thresh
        inside_thresh = tf.logical_not(tf.logical_or(above_thresh, below_thresh))
        status = tf.where(
            inside_thresh,
            0,  # always safe inside band
            tf.where(tf.logical_and(above_thresh, numeric_trend == -1), 0,
                     tf.where(tf.logical_and(below_thresh, numeric_trend == 1), 0, 1))
        )
        status = tf.cast(status, tf.int32)
        return status

    converter = tf.lite.TFLiteConverter.from_concrete_functions([
        _trend_fn.get_concrete_function()
    ])
    # Enable default optimisations (e.g. dynamic-range quantisation) for a smaller file
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Use float16 where possible – no representative dataset required because
    # the graph has no trained weights, but this still shrinks some metadata.
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    # Ensure destination directory exists before saving
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)

    with open(model_path, "wb") as f:
        f.write(tflite_model)
    return model_path

if __name__ == "__main__":
    path = build_trend_tflite_model()
    print(f"TFLite model successfully built and saved to {path}")
