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

  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue command_queue;
  cl_program program;

  cl_mem edge_matrix_device;
  cl_mem weight_matrix_device;
  cl_mem mask_matrix_device;
  cl_mem distance_matrix_device;
  cl_mem updating_distance_matrix_device;
  cl_mem previous_matrix_device;

  cl_kernel initialize_mask_kernel;
  cl_kernel dijkstra_kernel1;
  cl_kernel dijkstra_kernel2;
} oclDijkstraData;

gboolean    ocl_dijkstra_to             (oclDijkstraData         *data,
                                         Node                    *source,
                                         Node                    *target,
                                         guint                    width,
                                         guint                    height,
                                         gint                    *distance_matrix,
                                         Node                   **previous,
                                         Node                   **node_matrix);

void        ocl_init                    (oclDijkstraData         *data,
                                         gint                     matrix_size);

void        ocl_dijkstra_send_graph     (oclDijkstraData         *data,
                                         gint                     matrix_size);
