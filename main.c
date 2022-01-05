#include <stdio.h>
#include <stdlib.h>

// manual written
// define memref descriptor
typedef struct MemRef_descriptor_ {
  float* allocated;
  float* aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
} Memref;

/*
void _mlir_ciface_linalg_copy_viewsxsxf32_viewsxsxf32(Memref* input, Memref* output) {
  printf("\nit is copy view\n");
  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 3; j++) {
      printf("copy: input: %f\n", input->aligned[i*3+j]);
      printf("copy: output: %f\n", output->aligned[i*3+j]);
    }
  }
  return;
}
*/  

void _mlir_ciface_external_function(Memref* output, Memref* input_1, Memref* input_2, Memref* input_3) {
  printf("\nit is external function\n");
  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 3; j++) {
      printf("input_1: %f\n", input_1->aligned[i*3+j]);
      printf("input_2: %f\n", input_2->aligned[i*3+j]);
      printf("input_3: %f\n", input_3->aligned[i*3+j]);
//      printf("output: %f\n", output->aligned[i*3+j]);
    }
  }


  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 3; j++) {
      printf("output:\n");
      output->aligned[i*3+j] = input_2->aligned[i*3+j] - input_1->aligned[i*3+j];
      printf("%f", output->aligned[i*3+j]);
//      output->aligned[i*3+j] = output->aligned[i*3+j] + input_3->aligned[i*3+j];
    }
  }
  return;
}

void _mlir_ciface_forward(Memref* output, Memref* input_1, Memref* input_2, Memref* input_3);


int main() {
  Memref *arg0= (Memref *)malloc(sizeof(Memref));
  Memref *arg1= (Memref *)malloc(sizeof(Memref));
  Memref *arg2= (Memref *)malloc(sizeof(Memref));
//  Memref *output= (Memref *)malloc(sizeof(Memref));

  float a[2][3]={{10, 11, 12}, {13, 14, 15}};
  float b[2][3]={{16, 17, 18}, {19, 20, 21}};
  float c[2][3]={{22, 23, 24}, {25, 26, 27}};

  float** d = (float**)malloc(sizeof(float*) * 2);
  for (size_t i = 0; i < 2; i++) {
    d[i] = (float*)malloc(sizeof(float) * 3);
  }

  arg0->aligned = a;
  arg0->offset = 0;
  arg0->sizes[0] = 2;
  arg0->sizes[1] = 3;
  arg0->strides[0] = 1;
  arg0->strides[1] = 2;
  arg1->aligned = b;
  arg1->offset = 0;
  arg1->sizes[0] = 2;
  arg1->sizes[1] = 3;
  arg1->strides[0] = 1;
  arg1->strides[1] = 2;
  arg2->aligned = c;
  arg2->offset = 0;
  arg2->sizes[0] = 2;
  arg2->sizes[1] = 3;
  arg2->strides[0] = 1;
  arg2->strides[1] = 2;
/*  
  output->aligned = d;
  output->offset = 0;
  output->sizes[0] = 2;
  output->sizes[1] = 3;
  output->strides[0] = 1;
  output->strides[1] = 2;
*/

  printf("before forward\n");
  Memref output = _mlir_ciface_forward(arg0, arg1, arg2);
  
  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 3; j++) {
      printf("output: %f\n", output.aligned[i*3+j]);
    }
  }

  free(arg0);
  free(arg1);
  free(arg2);
//  free(output);
  return 0;
}
