#ifdef __APPLE__
  #include <OpenCL/cl.h>
#else
  #include <CL/cl.h>
#endif

#include "skeltrack-util.h"

typedef struct {
  /* Host data */
  gint *edge_matrix;
  gint *weight_matrix;
  guint *mask_matrix;
  gint *previous_matrix;
  guint *labels_matrix;
  gint *mD;

  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue command_queue;
  cl_program program;

  cl_mem edge_matrix_device;
  cl_mem weight_matrix_device;

  /* Labeling */
  cl_mem buffer_matrix_device;
  cl_mem labels_matrix_device;
  cl_mem mD_device;
  cl_mem close_node_device;

  cl_kernel initialize_graph_kernel;
  cl_kernel mesh_kernel;
  cl_kernel make_graph_kernel;
  cl_kernel join_to_biggest_kernel;
} oclData;

void        ocl_init                    (oclData                 *data,
                                         gint                     matrix_size);

void        ocl_ccl                     (oclData                 *data,
                                         guint16                 *buffer,
                                         gint                     width,
                                         gint                     height);

void        ocl_make_graph              (oclData                 *data,
                                         gint                     width,
                                         gint                     height,
                                         gint                     label,
                                         gint                     dimension_reduction);

gint *      ocl_join_to_biggest         (oclData                 *data,
                                         gint                     i,
                                         gint                     j,
                                         gint                     biggest,
                                         gint                     dist_x,
                                         gint                     dist_y,
                                         gint                     dist_z,
                                         gint                     width,
                                         gint                     height,
                                         gint                     dimension_reduction);

