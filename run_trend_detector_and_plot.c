#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <errno.h>
#include <tensorflow/lite/c/c_api.h>

#define COLOR_RED   0xFF0000
#define COLOR_BLUE  0x0000FF
#define COLOR_GREY  0x808080

typedef struct {
    float time;
    float temp;
} DataPoint;

typedef struct {
    int heating;
    int cooling;
    int constant;
} TrendCounts;

static void die(const char *msg) {
    perror(msg);
    exit(EXIT_FAILURE);
}

static int cmp_time(const void *a, const void *b) {
    const DataPoint *da = (const DataPoint *)a;
    const DataPoint *db = (const DataPoint *)b;
    if (da->time < db->time) return -1;
    if (da->time > db->time) return 1;
    return 0;
}

static DataPoint *load_csv(const char *path, size_t *out_len) {
    FILE *fp = fopen(path, "r");
    if (!fp) die("fopen CSV");

    char line[256];
    if (!fgets(line, sizeof(line), fp)) die("read header");

    size_t cap = 1024, len = 0;
    DataPoint *arr = malloc(cap * sizeof(DataPoint));
    if (!arr) die("malloc");
    while (fgets(line, sizeof(line), fp)) {
        char *tok = strtok(line, ",");
        if (!tok) continue;
        float rel_time = strtof(tok, NULL);
        tok = strtok(NULL, ",");
        if (!tok) continue;
        float temperature = strtof(tok, NULL);

        if (len == cap) {
            cap *= 2;
            arr = realloc(arr, cap * sizeof(DataPoint));
            if (!arr) die("realloc");
        }
        arr[len].time = rel_time;
        arr[len].temp = temperature;
        len++;
    }

    fclose(fp);
    qsort(arr, len, sizeof(DataPoint), cmp_time);
    *out_len = len;
    return arr;
}

static TfLiteInterpreter *init_interpreter(const char *model_path) {
    TfLiteModel *model = TfLiteModelCreateFromFile(model_path);
    if (!model) {
        fprintf(stderr, "Failed to load model: %s\n", model_path);
        exit(EXIT_FAILURE);
    }
    TfLiteInterpreterOptions *options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(options, 1);

    TfLiteInterpreter *interpreter = TfLiteInterpreterCreate(model, options);
    TfLiteInterpreterAllocateTensors(interpreter);

    /* Print I/O details */
    int input_cnt = TfLiteInterpreterGetInputTensorCount(interpreter);
    int output_cnt = TfLiteInterpreterGetOutputTensorCount(interpreter);
    printf("Model input tensors: %d, output tensors: %d\n", input_cnt, output_cnt);
    for (int i = 0; i < input_cnt; ++i) {
        const TfLiteTensor *t = TfLiteInterpreterGetInputTensor(interpreter, i);
        printf("  Input %d: type=%d dims=" , i, TfLiteTensorType(t));
        int dims = TfLiteTensorNumDims(t);
        for (int d = 0; d < dims; ++d) {
            printf("%s%d", d==0?"[":" ", TfLiteTensorDim(t,d));
        }
        printf("]\n");
    }
    for (int i = 0; i < output_cnt; ++i) {
        const TfLiteTensor *t = TfLiteInterpreterGetOutputTensor(interpreter, i);
        printf("  Output %d: type=%d dims=" , i, TfLiteTensorType(t));
        int dims = TfLiteTensorNumDims(t);
        for (int d = 0; d < dims; ++d) {
            printf("%s%d", d==0?"[":" ", TfLiteTensorDim(t,d));
        }
        printf("]\n");
    }

    TfLiteModelDelete(model);   /* interpreter holds reference */
    TfLiteInterpreterOptionsDelete(options);
    return interpreter;
}

static int predict_trend(TfLiteInterpreter *interp,
                         const float *temps_window,
                         const float *times_window,
                         int window) {
    int input_cnt = TfLiteInterpreterGetInputTensorCount(interp);
    if (input_cnt != 2) {
        fprintf(stderr, "Model expected 2 input tensors but got %d\n", input_cnt);
        exit(EXIT_FAILURE);
    }

    /* Ensure input tensor dims match window length; resize if needed */
    for (int idx = 0; idx < 2; ++idx) {
        TfLiteTensor *tensor = TfLiteInterpreterGetInputTensor(interp, idx);
        int ndims = TfLiteTensorNumDims(tensor);
        bool needs_resize = true;
        if (ndims == 1 && TfLiteTensorDim(tensor,0) == window) needs_resize = false;
        if (needs_resize) {
            int new_dims[1] = {window};
            TfLiteInterpreterResizeInputTensor(interp, idx, new_dims, 1);
        }
    }
    TfLiteInterpreterAllocateTensors(interp);

    /* Copy data into tensors */
    TfLiteTensor *t0 = TfLiteInterpreterGetInputTensor(interp, 0);
    TfLiteTensorCopyFromBuffer(t0, temps_window, window*sizeof(float));
    TfLiteTensor *t1 = TfLiteInterpreterGetInputTensor(interp, 1);
    TfLiteTensorCopyFromBuffer(t1, times_window, window*sizeof(float));

    /* Invoke */
    if (TfLiteInterpreterInvoke(interp) != kTfLiteOk) {
        fprintf(stderr, "Interpreter invoke failed\n");
        exit(EXIT_FAILURE);
    }

    const TfLiteTensor *out = TfLiteInterpreterGetOutputTensor(interp, 0);
    int32_t val;
    TfLiteTensorCopyToBuffer(out, &val, sizeof(int32_t));
    return (int)val;
}

/*-------------------------------------------------------------
 * MAIN PIPELINE
 *-----------------------------------------------------------*/

static void write_plot_files(const DataPoint *dp, const int *pred, size_t n) {
    FILE *pf = fopen("points.dat", "w");
    if (!pf) die("points.dat");
    for (size_t i = 0; i < n; ++i) {
        uint32_t color = (pred[i]==1) ? COLOR_RED : (pred[i]==-1 ? COLOR_BLUE : COLOR_GREY);
        fprintf(pf, "%f %f 0x%06x\n", dp[i].time, dp[i].temp, color);
    }
    fclose(pf);

    FILE *gp = fopen("plot.gp", "w");
    if (!gp) die("plot.gp");
    fprintf(gp,
        "set terminal pngcairo size 1400,600\n"
        "set output 'plot.png'\n"
        "set title 'Temperature Trend Detection – Red: Heating, Blue: Cooling, Grey: Constant/Fluctuating'\n"
        "set xlabel 'Relative Time'\n"
        "set ylabel 'Temperature (°C)'\n"
        "set grid\n"
        "plot 'points.dat' using 1:2:3 with points pt 7 ps 0.5 lc rgb var notitle\n");
    fclose(gp);
    system("gnuplot plot.gp");
}

int main(int argc, char **argv) {
    /* Default args */
    const char *csv_path = "filtered_temperature_data_2.csv";
    const char *model_path = "trend_predictor.tflite";
    int window = 8;

    /* Simple arg parsing */
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--csv") && i+1 < argc) {
            csv_path = argv[++i];
        } else if (!strcmp(argv[i], "--model") && i+1 < argc) {
            model_path = argv[++i];
        } else if (!strcmp(argv[i], "--window") && i+1 < argc) {
            window = atoi(argv[++i]);
        } else {
            fprintf(stderr, "Usage: %s [--csv PATH] [--model PATH] [--window N]\n", argv[0]);
            return EXIT_FAILURE;
        }
    }

    size_t n_points = 0;
    DataPoint *data = load_csv(csv_path, &n_points);
    printf("Loaded %zu data points from %s\n", n_points, csv_path);

    TfLiteInterpreter *interp = init_interpreter(model_path);

    if (window <= 0) {
        const TfLiteTensor *in0 = TfLiteInterpreterGetInputTensor(interp, 0);
        if (TfLiteTensorNumDims(in0) >= 1) {
            window = TfLiteTensorDim(in0, 0);
        } else {
            window = 1;
        }
    }
    printf("Using window size: %d\n", window);

    float *temp_window = malloc(window * sizeof(float));
    float *time_window = malloc(window * sizeof(float));
    int *predictions = malloc(n_points * sizeof(int));

    TrendCounts counts = {0,0,0};

    for (size_t idx = 0; idx < n_points; ++idx) {
        int start = (int)idx - window + 1;
        if (start < 0) start = 0;
        int len = (int)idx - start + 1;
        int pad = window - len;
        /* Pad with first value in window */
        for (int i = 0; i < pad; ++i) {
            temp_window[i] = data[start].temp;
            time_window[i] = data[start].time;
        }
        for (int i = 0; i < len; ++i) {
            temp_window[pad + i] = data[start + i].temp;
            time_window[pad + i] = data[start + i].time;
        }

        int pred = predict_trend(interp, temp_window, time_window, window);
        predictions[idx] = pred;
        if (pred == 1) counts.heating++;
        else if (pred == -1) counts.cooling++;
        else counts.constant++;
    }

    printf("Trend counts: {1: %d, -1: %d, 0: %d}\n", counts.heating, counts.cooling, counts.constant);

    write_plot_files(data, predictions, n_points);

    TfLiteInterpreterDelete(interp);
    free(data);
    free(temp_window);
    free(time_window);
    free(predictions);

    return 0;
} 