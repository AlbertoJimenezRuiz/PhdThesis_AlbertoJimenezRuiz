#include "main.h"
#include <stdarg.h>

#define CSV_FORMAT_PRINTF "%.20f"

void read_matrix_csv(const char *path, unsigned int *rows,
                     unsigned int *columns, FLOATING_TYPE **data) {
  // Read comma separated CSV
  unsigned int i, j;

  char *token;
  FILE *fp;

  fp = fopen(path, "r"); // read mode
  if (fp == NULL) {
    abort_msg("Error reading %s\n", path);
  }

  fseek(fp, 0L, SEEK_END);
  size_t fsize = ftell(fp);
  rewind(fp);
  char *string_total = (char *)calloc(1, fsize + 1);
  char *string_total2 = (char *)calloc(1, fsize + 1);
  size_t read_data_size = fread(string_total, fsize, 1, fp);
  fclose(fp);

  if (read_data_size != 1) {
    abort_msg("Error reading %s. Different size. \n", path);
  }

  *rows = 0;
  *columns = 0;

  memcpy(string_total2, string_total, fsize);
  token = NULL;
  while ((token = strtok((token == NULL) ? string_total2 : NULL, "\n")) !=
         NULL) {
    (*rows)++;
  }

  memcpy(string_total2, string_total, fsize);
  strchr(string_total2, '\n')[0] = '\0';

  token = NULL;
  while ((token = strtok((token == NULL) ? string_total2 : NULL, ",")) !=
         NULL) {
    (*columns)++;
  }

  *data = (FLOATING_TYPE *)malloc(sizeof(FLOATING_TYPE) * (*rows) * (*columns));

  memcpy(string_total2, string_total, fsize);

  size_t string_size_2 = strlen(string_total2);
  for (i = 0; i < string_size_2; i++) {
    if (string_total2[i] == '\n')
      string_total2[i] = ',';
  }

  i = j = 0;
  token = NULL;
  while ((token = strtok((token == NULL) ? string_total2 : NULL, ",")) !=
         NULL) {

    (*data)[i * (*columns) + j] = (FLOATING_TYPE)strtod(token, NULL);
    j++;

    if (j == *columns) {
      j = 0;
      i++;
    }
  }

  free(string_total);
  free(string_total2);
}

void write_csv_fullpath(const char *path, const unsigned int rows,
                        const unsigned int columns, const FLOATING_TYPE *data) {
  // Write CSV with the data of a matrix

  char fullpath[MAXIMUM_STRING_SIZE];
  sprintf(fullpath, "%s/%s.csv", OUTPUT_PATH, path);

  FILE *fp;

  fp = fopen(fullpath, "w"); // write mode

  for (unsigned int i = 0; i < rows; i++) {
    for (unsigned int j = 0; j < columns; j++) {
      fprintf(fp, CSV_FORMAT_PRINTF, (double)(data[i * columns + j]));

      if (j != columns - 1)
        fprintf(fp, ",");
      else
        fprintf(fp, "\n");
    }
  }
  fclose(fp);
}

void write_all_wt_results_csv(const wt_parameters_struct &params,
                              unsigned int number_wind_turbines,
                              FLOATING_TYPE *output_data,
                              unsigned int output_variables, const char *msg) {
  // write output of all wind turbines in CSV format

  for (int i = 0; i < (int)number_wind_turbines; i++) {
    char path[MAXIMUM_STRING_SIZE];
    sprintf(path, "result_%s%d", msg, i + 1);
    write_csv_fullpath(path, params.number_datapoints, output_variables,
                       output_data +
                           i * output_variables * params.number_datapoints);
  }
}

void __attribute__((format(printf, 1, 0))) abort_msg(const char *format, ...) {
  // Aborts program and prints message
  va_list args;
  va_start(args, format);
  vfprintf(stderr, format, args);
  va_end(args);

  abort();
}

TimeMeasurer_us::TimeMeasurer_us() {
  start = std::chrono::high_resolution_clock::now();
}

long int TimeMeasurer_us::measure(const char *msg) {
  auto now = std::chrono::high_resolution_clock::now();
  auto time_diff =
      chrono::duration_cast<chrono::nanoseconds>(now - start).count();

  cout << setprecision(3) << fixed;
  cout << "Time " << msg << setw(15) << (double)time_diff / 1000 << " us"
       << endl;
  return (long int)(time_diff / 1000); // returns microseconds
}
