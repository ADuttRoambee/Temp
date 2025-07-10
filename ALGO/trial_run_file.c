#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

// TensorFlow Lite C API header – ensure the path is correct for your installation
#include "tensorflow/lite/c/c_api.h"

/*
 * trial_run_file.c – Minimal C re-write of `trial_run_file.py` for running a
 * TensorFlow-Lite model on temperature data stored in a CSV file.
 *
 * Build (example):
 *   gcc -std=c11 -O2 -I/path/to/tflite/include \
 *       trial_run_file.c -L/path/to/tflite/lib -ltensorflow-lite -o trial_run_file
 *
 * Usage:
 *   ./trial_run_file --csv filtered_temperature_data_2.csv \
 *                    --model model.tflite --window 8
 *
 * NOTE
 * ----
 * - Plotting has been removed. Instead, basic statistics are printed.
 * - This program relies on the (stable) TensorFlow-Lite C API.
 * - Error-handling is intentionally minimalistic for brevity; production code
 *   should check return values where omitted.
 */

/* ----------------------------- Utility macros ----------------------------- */
#define die(msg)                   \
  do { fprintf(stderr, "%s\n", msg); exit(EXIT_FAILURE); } while (0)

/* ----------------------------- CSV utilities ----------------------------- */

typedef struct {
  float *times;      // relative_time column
  float *temps;      // temperature column
  size_t len;        // number of rows read
} CsvData;

static CsvData load_csv(const char *path) {
  FILE *fp = fopen(path, "r");
  if (!fp) {
    perror("fopen");
    die("Unable to open CSV file");
  }

  // We expect a header line containing at least: relative_time,temperature,...
  char line[512];
  if (!fgets(line, sizeof line, fp)) die("Empty CSV file");

  // Allocate dynamic arrays – start small and grow.
  size_t cap = 1024;
  CsvData data = { .times = malloc(cap * sizeof(float)),
                   .temps = malloc(cap * sizeof(float)),
                   .len   = 0 };
  if (!data.times || !data.temps) die("malloc failure");

  while (fgets(line, sizeof line, fp)) {
    // strtok/strtof parsing; assume comma-separated and columns in order:
    // relative_time,temperature
    char *save = NULL;
    char *tok  = strtok_r(line, ",", &save);
    if (!tok) continue;
    float rel_time = strtof(tok, NULL);
    tok = strtok_r(NULL, ",", &save);
    if (!tok) continue;
    float temp = strtof(tok, NULL);

    if (data.len == cap) {
      cap *= 2;
      data.times = realloc(data.times, cap * sizeof(float));
      data.temps = realloc(data.temps, cap * sizeof(float));
      if (!data.times || !data.temps) die("realloc failure");
    }
    data.times[data.len] = rel_time;
    data.temps[data.len] = temp;
    data.len++;
  }
  fclose(fp);
  return data;
}

/* ------------------------ TensorFlow-Lite helpers ------------------------- */

static TfLiteInterpreter *create_interpreter(const char *model_path) {
  TfLiteModel *model = TfLiteModelCreateFromFile(model_path);
  if (!model) die("Failed to load TFLite model");

  TfLiteInterpreterOptions *options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsSetNumThreads(options, 1);

  TfLiteInterpreter *interpreter = TfLiteInterpreterCreate(model, options);
  if (!interpreter) die("Failed to create interpreter");

  if (TfLiteInterpreterAllocateTensors(interpreter) != kTfLiteOk)
    die("AllocateTensors failed");

  // Print model IO details (optional)
  int n_inputs  = TfLiteInterpreterGetInputTensorCount(interpreter);
  int n_outputs = TfLiteInterpreterGetOutputTensorCount(interpreter);
  printf("Model has %d input(s) and %d output(s)\n", n_inputs, n_outputs);
  for (int i = 0; i < n_inputs; ++i) {
    const TfLiteTensor *t = TfLiteInterpreterGetInputTensor(interpreter, i);
    printf("  Input %d: type=%d, dims=", i, TfLiteTensorType(t));
    const TfLiteIntArray *dims = TfLiteTensorDims(t);
    for (int d = 0; d < dims->size; ++d) printf("%d%s", dims->data[d], d+1==dims->size?"":"x");
    printf("\n");
  }
  for (int i = 0; i < n_outputs; ++i) {
    const TfLiteTensor *t = TfLiteInterpreterGetOutputTensor(interpreter, i);
    printf("  Output %d: type=%d, dims=", i, TfLiteTensorType(t));
    const TfLiteIntArray *dims = TfLiteTensorDims(t);
    for (int d = 0; d < dims->size; ++d) printf("%d%s", dims->data[d], d+1==dims->size?"":"x");
    printf("\n");
  }
  printf("\n");

  TfLiteInterpreterOptionsDelete(options);
  TfLiteModelDelete(model);  // Interpreter keeps its own reference.
  return interpreter;
}

static void copy_tensor_float(TfLiteInterpreter *intrp, int index, const float *src, size_t n) {
  TfLiteTensor *t = TfLiteInterpreterGetInputTensor(intrp, index);
  // Resize if needed
  const TfLiteIntArray *dims = TfLiteTensorDims(t);
  if (dims->size != 1 || (size_t)dims->data[0] != n) {
    TfLiteIntArray *new_dims = TfLiteIntArrayCreate(1);
    new_dims->data[0] = (int)n;
    new_dims->size    = 1;
    if (TfLiteInterpreterResizeInputTensor(intrp, index, new_dims) != kTfLiteOk)
      die("ResizeInputTensor failed");
    if (TfLiteInterpreterAllocateTensors(intrp) != kTfLiteOk)
      die("AllocateTensors (after resize) failed");
    t = TfLiteInterpreterGetInputTensor(intrp, index);
  }
  if (TfLiteTensorCopyFromBuffer(t, src, n * sizeof(float)) != kTfLiteOk)
    die("TensorCopyFromBuffer failed");
}

static void copy_tensor_scalar(TfLiteInterpreter *intrp, int index, float value) {
  TfLiteTensor *t = TfLiteInterpreterGetInputTensor(intrp, index);
  float v = value;
  if (TfLiteTensorCopyFromBuffer(t, &v, sizeof(float)) != kTfLiteOk)
    die("TensorCopyFromBuffer (scalar) failed");
}

static int32_t run_inference(TfLiteInterpreter *intrp,
                             const float *temp_window, size_t window,
                             float high_thresh, float low_thresh) {
  // Assuming model expects 3 inputs as per Python code.
  copy_tensor_float(intrp, 0, temp_window, window);
  copy_tensor_scalar(intrp, 1, high_thresh);
  copy_tensor_scalar(intrp, 2, low_thresh);

  if (TfLiteInterpreterInvoke(intrp) != kTfLiteOk)
    die("TfLiteInterpreterInvoke failed");

  const TfLiteTensor *out_t = TfLiteInterpreterGetOutputTensor(intrp, 0);
  int32_t out[2] = {0, 0};
  if (TfLiteTensorCopyToBuffer(out_t, out, sizeof out) != kTfLiteOk)
    die("TensorCopyToBuffer (output) failed");

  // status = out[0]; trend = out[1]
  return out[0];
}

/* ------------------------- Main application code ------------------------- */

int main(int argc, char **argv) {
  // Defaults
  const char *csv_path   = "filtered_temperature_data_2.csv";
  const char *model_path = "ALGO/model.tflite";
  int window             = 8;

  // Very light-weight argument parsing
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) csv_path = argv[++i];
    else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) model_path = argv[++i];
    else if (strcmp(argv[i], "--window") == 0 && i + 1 < argc) window = atoi(argv[++i]);
    else {
      fprintf(stderr, "Unknown or incomplete argument: %s\n", argv[i]);
      return EXIT_FAILURE;
    }
  }

  printf("CSV file   : %s\n", csv_path);
  printf("Model file : %s\n", model_path);
  printf("Window size: %d\n\n", window);

  CsvData data = load_csv(csv_path);
  printf("Loaded %zu rows from CSV\n", data.len);

  TfLiteInterpreter *intrp = create_interpreter(model_path);

  // Determine default window from model if user supplied <=0
  if (window <= 0) {
    const TfLiteTensor *t0 = TfLiteInterpreterGetInputTensor(intrp, 0);
    const TfLiteIntArray *dims = TfLiteTensorDims(t0);
    if (dims->size >= 1) window = dims->data[dims->size - 1];
    if (window <= 0) window = 1;
    printf("Using window inferred from model: %d\n", window);
  }

  float *temp_window = malloc(window * sizeof(float));
  if (!temp_window) die("malloc temp_window failed");

  int32_t *statuses = malloc(data.len * sizeof(int32_t));
  if (!statuses) die("malloc statuses failed");

  const float high_thresh = 10.0f;
  const float low_thresh  = 0.0f;
  printf("Using thresholds: %.2f °C – %.2f °C\n\n", low_thresh, high_thresh);

  // Sliding-window loop
  for (size_t idx = 0; idx < data.len; ++idx) {
    size_t start = (idx + 1 >= (size_t)window) ? idx + 1 - window : 0;
    size_t len_window = idx - start + 1;

    // Copy with padding if needed
    if (len_window < (size_t)window) {
      float first_val = data.temps[start];
      size_t pad = window - len_window;
      for (size_t p = 0; p < pad; ++p) temp_window[p] = first_val;
      memcpy(temp_window + pad, data.temps + start, len_window * sizeof(float));
    } else {
      memcpy(temp_window, data.temps + start, window * sizeof(float));
    }

    statuses[idx] = run_inference(intrp, temp_window, (size_t)window,
                                  high_thresh, low_thresh);
  }

  // Compute status counts
  size_t ok_cnt = 0, alert_cnt = 0;
  for (size_t i = 0; i < data.len; ++i) {
    if (statuses[i] == 0) ok_cnt++; else if (statuses[i] == 1) alert_cnt++;
  }
  printf("\nStatus counts: {0: 'alright', 1: 'alert'} -> {0: %zu, 1: %zu}\n",
         ok_cnt, alert_cnt);

  // Cleanup
  TfLiteInterpreterDelete(intrp);
  free(data.times);
  free(data.temps);
  free(temp_window);
  free(statuses);

  return 0;
} 