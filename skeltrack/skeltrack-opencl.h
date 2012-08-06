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

  /* Dijkstra */
  cl_mem mask_matrix_device;
  cl_mem distance_matrix_device;
  cl_mem updating_distance_matrix_device;
  cl_mem previous_matrix_device;

  cl_kernel initialize_mask_kernel;
  cl_kernel flush_distance_matrix_kernel;
  cl_kernel set_source_vertex_kernel;
  cl_kernel dijkstra_kernel1;
  cl_kernel dijkstra_kernel2;

  /* Labeling */
  cl_mem buffer_matrix_device;
  cl_mem labels_matrix_device;
  cl_mem mD_device;

  cl_kernel initialize_graph_kernel;
  cl_kernel mesh_kernel;
  cl_kernel make_graph_kernel;
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
                                         gint                     label);

gboolean    ocl_dijkstra_to             (oclData                 *data,
                                         Node                    *source,
                                         Node                    *target,
                                         guint                    width,
                                         guint                    height,
                                         gint                    *distance_matrix,
                                         Node                   **previous,
                                         Node                   **node_matrix);

void        ocl_flush_distance_matrix   (oclData                 *data,
                                         gint                     matrix_sixe);
