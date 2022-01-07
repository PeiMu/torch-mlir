#include <stdio.h>
#include <stdlib.h>

#define SIZE_0 2
#define SIZE_1 3

// manual written
// define memref descriptor
typedef struct MemRef_descriptor_ {
  float* allocated;
  float* aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
} Memref;

float array[SIZE_0 * SIZE_1];

void _mlir_ciface_external_function(Memref* output, Memref* input_1, Memref* input_2, Memref* input_3) {
  output->aligned = array;
  output->offset = 0;
  output->sizes[0] = SIZE_0;
  output->sizes[1] = SIZE_1;
  output->strides[0] = SIZE_1;
  output->strides[1] = 1;
  
  input_1->strides[0] = SIZE_1;
  input_1->strides[1] = 1;
  
  printf("\nit is external function\n");
  for (size_t i = 0; i < SIZE_0; i++) {
    for (size_t j = 0; j < SIZE_1; j++) {
      printf("input_1: %f\n", input_1->aligned[i*SIZE_1+j]);
      printf("input_2: %f\n", input_2->aligned[i*SIZE_1+j]);
      printf("input_3: %f\n", input_3->aligned[i*SIZE_1+j]);
//      printf("output: %f\n", output->aligned[i*3+j]);
    }
  }


  for (size_t i = 0; i < SIZE_0; i++) {
    for (size_t j = 0; j < SIZE_1; j++) {
      printf("output in external function: ");
      output->aligned[i*SIZE_1+j] = input_2->aligned[i*SIZE_1+j] - input_1->aligned[i*SIZE_1+j];
      printf("%f\n", output->aligned[i*SIZE_1+j]);
//      output->aligned[i*SIZE_1+j] = output->aligned[i*SIZE_1+j] + input_3->aligned[i*SIZE_1+j];
//      printf("%f\n", output->aligned[i*SIZE_1+j]);
    }
  }
  return;
}

void _mlir_ciface_forward(Memref* output, Memref* input_1, Memref* input_2, Memref* input_3);


int main() {
  Memref *arg0= (Memref *)malloc(sizeof(Memref));
  Memref *arg1= (Memref *)malloc(sizeof(Memref));
  Memref *arg2= (Memref *)malloc(sizeof(Memref));
  Memref *output= (Memref *)malloc(sizeof(Memref));

  float a[SIZE_0 * SIZE_1]={1, 2, 3, 4, 5, 6};
  float b[SIZE_0 * SIZE_1]={11, 12, 13, 14, 15, 16};
  float c[SIZE_0 * SIZE_1]={21, 22, 23, 24, 25, 26};

  float d[SIZE_0 * SIZE_1];
  arg0->aligned = a;
  arg0->offset = 0;
  arg0->sizes[0] = SIZE_0;
  arg0->sizes[1] = SIZE_1;
  arg0->strides[0] = SIZE_1;
  arg0->strides[1] = 1;
  arg1->aligned = b;
  arg1->offset = 0;
  arg1->sizes[0] = SIZE_0;
  arg1->sizes[1] = SIZE_1;
  arg1->strides[0] = SIZE_1;
  arg1->strides[1] = 1;
  arg2->aligned = c;
  arg2->offset = 0;
  arg2->sizes[0] = SIZE_0;
  arg2->sizes[1] = SIZE_1;
  arg2->strides[0] = SIZE_1;
  arg2->strides[1] = 1;
  output->aligned = d;
  output->offset = 0;
  output->sizes[0] = SIZE_0;
  output->sizes[1] = SIZE_1;
  output->strides[0] = SIZE_1;
  output->strides[1] = 1;

  printf("before forward\n");
  _mlir_ciface_forward(output, arg0, arg1, arg2);
  
  for (size_t i = 0; i < SIZE_0; i++) {
    for (size_t j = 0; j < SIZE_1; j++) {
      printf("output: %f\n", output->aligned[i*SIZE_1+j]);
    }
  }

  free(arg0);
  free(arg1);
  free(arg2);
  free(output);
  return 0;
}
