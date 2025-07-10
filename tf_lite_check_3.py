import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tensorflow as tf



def _build_trend_tflite_model(window_size=8,
                              trend_threshold=1.0,
                              trend_change_threshold=0.1,
                              model_path="ALGO/BSF_temp_trend_test.tflite"):
    """Build (or reuse) a small TFLite model for trend detection."""

    # ------------------------------------------------------------------
    # Edge-Impulse compatible signature:
    #   • 1 input tensor  : shape  (1, window_size)   ‑ batch dimension of 1
    #   • 1 output tensor : shape  (1, 3)            ‑ class probabilities
    # Classes: 0 = Cooling, 1 = Constant, 2 = Heating
    # ------------------------------------------------------------------

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, window_size], dtype=tf.float32, name="x"),
    ])
    def _trend_fn(x):
        """Stateless trend classification packed for Edge Impulse."""

        # Remove batch dimension -> [window_size]
        temperatures = tf.squeeze(x, axis=0)

        # Synthetic time axis (equally spaced) – use constant to avoid tf.linspace op
        times = tf.constant([i * 100.0 for i in range(window_size)], dtype=tf.float32)
        mean_t = tf.reduce_mean(times)
        mean_temp = tf.reduce_mean(temperatures)
        cov = tf.reduce_sum((times - mean_t) * (temperatures - mean_temp))
        var = tf.reduce_sum(tf.square(times - mean_t))
        slope = tf.math.divide_no_nan(cov, var)

        diffs = temperatures[1:] - temperatures[:-1]                # ΔT per sample
        avg_change = tf.reduce_mean(diffs)                          # mean ΔT

        signs = tf.less(diffs, 0.0)
        sign_flips = tf.cast(tf.not_equal(signs[1:], signs[:-1]), tf.int32)
        sign_changes = tf.reduce_sum(sign_flips)
        fluct_thresh = tf.cast(tf.shape(temperatures)[0], tf.float32) / 0.5  # window_size / 0.5
        is_fluctuating = tf.greater(tf.cast(sign_changes, tf.float32), fluct_thresh)

        heating_cond = tf.logical_or(
            tf.greater(slope, trend_change_threshold),
            tf.greater(avg_change, trend_threshold / 3.5)
        )
        cooling_cond = tf.logical_or(
            tf.less(slope, -trend_change_threshold),
            tf.less(avg_change, -trend_threshold / 3.5)
        )

        heating_cond = tf.logical_and(heating_cond, tf.logical_not(is_fluctuating))
        cooling_cond = tf.logical_and(cooling_cond, tf.logical_not(is_fluctuating))

        def _heat():
            return tf.constant(2, tf.int32)

        def _cool():
            return tf.constant(0, tf.int32)

        def _const():
            return tf.constant(1, tf.int32)

        numeric_trend = tf.case(
            [(heating_cond, _heat), (cooling_cond, _cool)],
            default=_const,
            exclusive=True,
        )  # 0=cool,1=const,2=heat

        # Manually build probability vector without using one_hot (MCU-friendly)
        cool_prob     = tf.cast(tf.equal(numeric_trend, 0), tf.float32)
        const_prob    = tf.cast(tf.equal(numeric_trend, 1), tf.float32)
        heating_prob  = tf.cast(tf.equal(numeric_trend, 2), tf.float32)
        probs = tf.stack([cool_prob, const_prob, heating_prob])  # shape (3,)
        probs = tf.reshape(probs, [1, 3])  # add batch dim

        return probs

    converter = tf.lite.TFLiteConverter.from_concrete_functions([
        _trend_fn.get_concrete_function()
    ])
    tflite_model = converter.convert()
    with open(model_path, "wb") as f:
        f.write(tflite_model)
    return model_path

class TemperatureAnalyzer:
    def __init__(self, speed_factor=1.0):

        self.df = pd.read_csv('filtered_temperature_data_2.csv')
        self.current_idx = 0
        self.speed_factor = speed_factor

        self.window_size = 8
        self.trend_threshold = 1.0    # °C overall change within window
        self.trend_change_threshold = 0.1  # °C per-sample slope
        self.prev_slope = 0
        self.trend_duration = 0
        self.last_confirmed_trend = "Unknown"
        
        # --- Thresholds & recent predictions -----------------------------
        mean_temp = self.df['temperature'].mean()
        self.high_threshold = mean_temp + 1.0  # default ±1 °C around mean
        self.low_threshold  = mean_temp - 1.0
        self.recent_preds = []  # rolling buffer of last trend predictions

        self.fig, self.ax = plt.subplots(figsize=(15, 8))

        self.confirmed_scatter = self.ax.scatter([], [], c=[], cmap='coolwarm', 
                                               vmin=-0.5, vmax=0.5, s=25, zorder=2)
        self.unconfirmed_scatter = self.ax.scatter([], [], c='lightgray', s=25, zorder=1)
        self.fluctuating_scatter = self.ax.scatter([], [], c='orange', s=25, zorder=5)
        self.current_point, = self.ax.plot([], [], 'ko', markersize=8)
  
        self.ax.set_xlim(self.df['relative_time'].min(), self.df['relative_time'].max())
        temp_range = self.df['temperature'].max() - self.df['temperature'].min()
        self.ax.set_ylim(
            self.df['temperature'].min() - temp_range * 0.1,
            self.df['temperature'].max() + temp_range * 0.1
        )
        self.ax.set_title('Temperature vs Time (Real-time Analysis)')
        self.ax.set_xlabel('Time (seconds)')
        self.ax.set_ylabel('Temperature (°C)')
        self.ax.grid(True, alpha=0.3)
    
        self.shown_times = []
        self.shown_temps = []
        self.confirmed_trends = []
        self.confirmed_indices = []
        self.current_trend = "Unknown"
        self.fluctuating_indices = []

        self.speed_text = self.ax.text(0.02, 0.98, f'Speed: {speed_factor}x',
                                     transform=self.ax.transAxes,
                                     bbox=dict(facecolor='white', alpha=0.7))
        self.temp_text = self.ax.text(0.02, 0.92, 'Temperature: --°C',
                                    transform=self.ax.transAxes,
                                    bbox=dict(facecolor='white', alpha=0.7))
        self.trend_text = self.ax.text(0.02, 0.86, 'Trend: Unknown',
                                     transform=self.ax.transAxes,
                                     bbox=dict(facecolor='white', alpha=0.7))

        try:
            model_path = _build_trend_tflite_model(
                window_size=self.window_size,
                trend_threshold=self.trend_threshold,
                trend_change_threshold=self.trend_change_threshold,
            )

            self.tflite_interpreter = tf.lite.Interpreter(model_path=model_path)
            self.tflite_interpreter.allocate_tensors()
            self._tflite_input_details = self.tflite_interpreter.get_input_details()
            self._tflite_output_details = self.tflite_interpreter.get_output_details()
        except Exception as _e:  # Fallback in case TensorFlow Lite is unavailable
            print("Warning: unable to initialise TFLite trend predictor –", _e)
            self.tflite_interpreter = None

    def trend_to_numeric(self, trend):
        if trend == "Heating":
            return 1.0
        elif trend == "Cooling":
            return -1.0
        else:
            return 0.0

    def calculate_slope(self, times, temps):
        if len(times) < 2:
            return 0
        slope, _ = np.polyfit(times, temps, 1)
        return slope

    def detect_trend(self, temperatures, times):
        if len(temperatures) < self.window_size:
            return "Unknown"
            
        current_window_temps = temperatures[-self.window_size:]
        current_window_times = times[-self.window_size:]
        current_slope = self.calculate_slope(current_window_times, current_window_temps)
        avg_change = np.mean(np.diff(current_window_temps))
        temp_range = np.max(current_window_temps) - np.min(current_window_temps)
        changes = np.diff(current_window_temps)
        sign_changes = np.sum(np.diff(np.signbit(changes)))
        is_fluctuating = sign_changes > self.window_size/0.5

        if is_fluctuating:
            self.fluctuating_indices.append(len(temperatures) - 1)
        
        if self.last_confirmed_trend == "Unknown":
            if (abs(current_slope) < self.trend_threshold and current_slope < -self.trend_threshold):
                return "Constant"
            if current_slope > self.trend_change_threshold or avg_change > self.trend_threshold/2:
                return "Heating"
            elif current_slope < -self.trend_change_threshold or avg_change < -self.trend_threshold/2:
                return "Cooling"
            return "Constant"
        
        if self.last_confirmed_trend == "Heating":
            if current_slope < -self.trend_change_threshold or avg_change < -self.trend_threshold/2:
                self.trend_duration = 0
                return "Cooling"
            elif (abs(current_slope) < self.trend_threshold/5 and temp_range < 1):
                self.trend_duration = 0
                return "Constant"
            self.trend_duration += 1
            return "Heating"
            
        elif self.last_confirmed_trend == "Cooling":
            if current_slope > self.trend_change_threshold or avg_change > self.trend_threshold/2:
                self.trend_duration = 0
                return "Heating"
            elif (abs(current_slope) < self.trend_threshold/5 and temp_range < 1):
                self.trend_duration = 0
                return "Constant"
            self.trend_duration += 1
            return "Cooling"
            
        else:
            if not is_fluctuating:
                if current_slope > self.trend_change_threshold or avg_change > self.trend_threshold/2:
                    self.trend_duration = 0
                    return "Heating"
                elif current_slope < -self.trend_change_threshold or avg_change < -self.trend_threshold/2:
                    self.trend_duration = 0
                    return "Cooling"
            return "Constant"

    def update_trend(self):
        if len(self.shown_temps) < self.window_size:
            return
            
        unconfirmed_start = 0 if not self.confirmed_indices else self.confirmed_indices[-1] + 1
        if unconfirmed_start >= len(self.shown_temps) - self.window_size:
            return
            
        temps_to_analyze = self.shown_temps[unconfirmed_start:]
        times_to_analyze = self.shown_times[unconfirmed_start:]
        new_trend = self.detect_trend(temps_to_analyze, times_to_analyze)
        
        if new_trend != "Unknown":
            points_to_confirm = max(0, len(temps_to_analyze) - self.window_size)
            if points_to_confirm > 0:
                self.confirmed_trends.extend([self.trend_to_numeric(new_trend)] * points_to_confirm)
                self.confirmed_indices.extend(range(unconfirmed_start, unconfirmed_start + points_to_confirm))
                self.last_confirmed_trend = new_trend
        
        self.current_trend = new_trend

    def init(self):
        return (self.confirmed_scatter, self.unconfirmed_scatter, self.fluctuating_scatter,
                self.current_point, self.speed_text, self.temp_text, self.trend_text)

    def update(self, frame):
        if self.current_idx < len(self.df):
            points_to_add = max(1, int(self.speed_factor * 10))
            for _ in range(points_to_add):
                if self.current_idx < len(self.df):
                    self.shown_times.append(self.df['relative_time'].iloc[self.current_idx])
                    self.shown_temps.append(self.df['temperature'].iloc[self.current_idx])
                    self.current_idx += 1

            self.update_trend()

            # Maintain rolling window of last predictions for the TFLite model
            self.recent_preds.append(int(self.trend_to_numeric(self.current_trend)))
            if len(self.recent_preds) > self.window_size:
                self.recent_preds.pop(0)

            if self.shown_times:
                if self.confirmed_indices:
                    confirmed_times = [self.shown_times[i] for i in self.confirmed_indices]
                    confirmed_temps = [self.shown_temps[i] for i in self.confirmed_indices]
                    self.confirmed_scatter.set_offsets(np.c_[confirmed_times, confirmed_temps])
                    self.confirmed_scatter.set_array(np.array(self.confirmed_trends))
                else:
                    self.confirmed_scatter.set_offsets(np.c_[[], []])
                
                if self.fluctuating_indices:
                    fluctuating_times = [self.shown_times[i] for i in self.fluctuating_indices]
                    fluctuating_temps = [self.shown_temps[i] for i in self.fluctuating_indices]
                    self.fluctuating_scatter.set_offsets(np.c_[fluctuating_times, fluctuating_temps])
                else:
                    self.fluctuating_scatter.set_offsets(np.c_[[], []])

                unconfirmed_indices = list(set(range(len(self.shown_times))) - 
                                        set(self.confirmed_indices) - 
                                        set(self.fluctuating_indices))
                if unconfirmed_indices:
                    unconfirmed_times = [self.shown_times[i] for i in unconfirmed_indices]
                    unconfirmed_temps = [self.shown_temps[i] for i in unconfirmed_indices]
                    self.unconfirmed_scatter.set_offsets(np.c_[unconfirmed_times, unconfirmed_temps])
                else:
                    self.unconfirmed_scatter.set_offsets(np.c_[[], []])
                
                self.current_point.set_data([self.shown_times[-1]], [self.shown_temps[-1]])
                
                if len(self.shown_times) >= self.window_size:
                    current_slope = self.calculate_slope(
                        self.shown_times[-self.window_size:],
                        self.shown_temps[-self.window_size:]
                    )
                    temp_range = max(self.shown_temps[-self.window_size:]) - min(self.shown_temps[-self.window_size:])
                    metrics_text = f" (Rate: {current_slope:.3f}°C/s, Range: {temp_range:.2f}°C)"
                else:
                    metrics_text = ""
                
                self.temp_text.set_text(f'Temperature: {self.shown_temps[-1]:.1f}°C')
                self.trend_text.set_text(f'Trend: {self.current_trend}{metrics_text}')

        return (self.confirmed_scatter, self.unconfirmed_scatter, self.fluctuating_scatter,
                self.current_point, self.speed_text, self.temp_text, self.trend_text)

    def adjust_speed(self, event):
        if event.key == 'up':
            self.speed_factor *= 2
        elif event.key == 'down':
            self.speed_factor = max(0.25, self.speed_factor / 2)
        self.speed_text.set_text(f'Speed: {self.speed_factor}x')

    def run(self):
        self.fig.canvas.mpl_connect('key_press_event', self.adjust_speed)
        
        anim = FuncAnimation(
            self.fig, self.update, init_func=self.init,
            interval=50,
            blit=True,
            repeat=False 
        )
        
        plt.figtext(
            0.99, 0.02,
            'Controls:\n↑: Double speed\n↓: Half speed',
            ha='right', va='bottom',
            bbox=dict(facecolor='white', alpha=0.7)
        )
        
        plt.show()
          
        print("\nTemperature Analysis:")
        print("Timestamp (s) | Temperature (°C) | Average Change (°C)")
        print("-" * 60)
        
        window_size = self.window_size
        for i in range(window_size, len(self.shown_times)):
            window_temps = self.shown_temps[i-window_size:i]
            avg_change = np.mean(np.diff(window_temps))
            print(f"{self.shown_times[i]:12.2f} | {self.shown_temps[i]:13.2f} | {avg_change:14.3f}")

    def predict_trend_tflite(self, temperatures, times):
        """Return 'Heating', 'Cooling', 'Constant' or 'Unknown' using TFLite."""

        if self.tflite_interpreter is None or len(temperatures) < self.window_size:
            return "Unknown"

        # Prepare input tensor shape (1, window_size)
        temps_arr = np.array(temperatures[-self.window_size:], dtype=np.float32).reshape(1, -1)

        self.tflite_interpreter.resize_tensor_input(
            self._tflite_input_details[0]['index'], temps_arr.shape
        )
        self.tflite_interpreter.allocate_tensors()
        self.tflite_interpreter.set_tensor(
            self._tflite_input_details[0]['index'], temps_arr
        )

        # Inference
        self.tflite_interpreter.invoke()
        probs = self.tflite_interpreter.get_tensor(
            self._tflite_output_details[0]['index']
        )[0]

        class_idx = int(np.argmax(probs))
        trend_str = {0: "Cooling", 1: "Constant", 2: "Heating"}.get(class_idx, "Unknown")

        return {"trend": trend_str, "probabilities": probs.tolist()}

if __name__ == "__main__":
    analyzer = TemperatureAnalyzer(speed_factor=1.0)
    analyzer.run()

    if hasattr(analyzer, "predict_trend_tflite") and analyzer.tflite_interpreter:
        print("\nTFLite Trend Predictions (window-wise):")
        for i in range(analyzer.window_size, len(analyzer.shown_times)):
            pred = analyzer.predict_trend_tflite(
                analyzer.shown_temps[: i], analyzer.shown_times[: i]
            )
            print(f"{analyzer.shown_times[i]:12.2f} s | {pred}")
