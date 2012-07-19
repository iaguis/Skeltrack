#include <stdio.h>
#include <limits.h>

#include "skeltrack-opencl.h"

void
check_error_file_line (int err_num,
                       int expected,
                       const char* file,
                       const int line_number);

#define check_error(a, b) check_error_file_line (a, b, __FILE__, __LINE__)

char *
error_desc (int err_num)
{
  switch (err_num)
    {
      case CL_SUCCESS:
        return "CL_SUCCESS";
        break;

      case CL_DEVICE_NOT_FOUND:
        return "CL_DEVICE_NOT_FOUND";
        break;

      case CL_DEVICE_NOT_AVAILABLE:
        return "CL_DEVICE_NOT_AVAILABLE";
        break;

      case CL_COMPILER_NOT_AVAILABLE:
        return "CL_COMPILER_NOT_AVAILABLE";
        break;

      case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        break;

      case CL_OUT_OF_RESOURCES:
        return "CL_OUT_OF_RESOURCES";
        break;

      case CL_OUT_OF_HOST_MEMORY:
        return "CL_OUT_OF_HOST_MEMORY";
        break;

      case CL_PROFILING_INFO_NOT_AVAILABLE:
        return "CL_PROFILING_INFO_NOT_AVAILABLE";
        break;

      case CL_MEM_COPY_OVERLAP:
        return "CL_MEM_COPY_OVERLAP";
        break;

      case CL_IMAGE_FORMAT_MISMATCH:
        return "CL_IMAGE_FORMAT_MISMATCH";
        break;

      case CL_IMAGE_FORMAT_NOT_SUPPORTED:
        return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        break;

      case CL_BUILD_PROGRAM_FAILURE:
        return "CL_BUILD_PROGRAM_FAILURE";
        break;

      case CL_MAP_FAILURE:
        return "CL_MAP_FAILURE";
        break;

      case CL_INVALID_VALUE:
        return "CL_INVALID_VALUE";
        break;

      case CL_INVALID_DEVICE_TYPE:
        return "CL_INVALID_DEVICE_TYPE";
        break;

      case CL_INVALID_PLATFORM:
        return "CL_INVALID_PLATFORM";
        break;

      case CL_INVALID_DEVICE:
        return "CL_INVALID_DEVICE";
        break;

      case CL_INVALID_CONTEXT:
        return "CL_INVALID_CONTEXT";
        break;

      case CL_INVALID_QUEUE_PROPERTIES:
        return "CL_INVALID_QUEUE_PROPERTIES";
        break;

      case CL_INVALID_COMMAND_QUEUE:
        return "CL_INVALID_COMMAND_QUEUE";
        break;

      case CL_INVALID_HOST_PTR:
        return "CL_INVALID_HOST_PTR";
        break;

      case CL_INVALID_MEM_OBJECT:
        return "CL_INVALID_MEM_OBJECT";
        break;

      case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
        return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        break;

      case CL_INVALID_IMAGE_SIZE:
        return "CL_INVALID_IMAGE_SIZE";
        break;

      case CL_INVALID_SAMPLER:
        return "CL_INVALID_SAMPLER";
        break;

      case CL_INVALID_BINARY:
        return "CL_INVALID_BINARY";
        break;

      case CL_INVALID_BUILD_OPTIONS:
        return "CL_INVALID_BUILD_OPTIONS";
        break;

      case CL_INVALID_PROGRAM:
        return "CL_INVALID_PROGRAM";
        break;

      case CL_INVALID_PROGRAM_EXECUTABLE:
        return "CL_INVALID_PROGRAM_EXECUTABLE";
        break;

      case CL_INVALID_KERNEL_NAME:
        return "CL_INVALID_KERNEL_NAME";
        break;

      case CL_INVALID_KERNEL_DEFINITION:
        return "CL_INVALID_KERNEL_DEFINITION";
        break;

      case CL_INVALID_KERNEL:
        return "CL_INVALID_KERNEL";
        break;

      case CL_INVALID_ARG_INDEX:
        return "CL_INVALID_ARG_INDEX";
        break;

      case CL_INVALID_ARG_VALUE:
        return "CL_INVALID_ARG_VALUE";
        break;

      case CL_INVALID_ARG_SIZE:
        return "CL_INVALID_ARG_SIZE";
        break;

      case CL_INVALID_KERNEL_ARGS:
        return "CL_INVALID_KERNEL_ARGS";
        break;

      case CL_INVALID_WORK_DIMENSION:
        return "CL_INVALID_WORK_DIMENSION";
        break;

      case CL_INVALID_WORK_GROUP_SIZE:
        return "CL_INVALID_WORK_GROUP_SIZE";
        break;

      case CL_INVALID_WORK_ITEM_SIZE:
        return "CL_INVALID_WORK_ITEM_SIZE";
        break;

      case CL_INVALID_GLOBAL_OFFSET:
        return "CL_INVALID_GLOBAL_OFFSET";
        break;

      case CL_INVALID_EVENT_WAIT_LIST:
        return "CL_INVALID_EVENT_WAIT_LIST";
        break;

      case CL_INVALID_EVENT:
        return "CL_INVALID_EVENT";
        break;

      case CL_INVALID_OPERATION:
        return "CL_INVALID_OPERATION";
        break;

      case CL_INVALID_GL_OBJECT:
        return "CL_INVALID_GL_OBJECT";
        break;

      case CL_INVALID_BUFFER_SIZE:
        return "CL_INVALID_BUFFER_SIZE";
        break;

      case CL_INVALID_MIP_LEVEL:
        return "CL_INVALID_MIP_LEVEL";
        break;

      case CL_INVALID_GLOBAL_WORK_SIZE:
        return "CL_INVALID_GLOBAL_WORK_SIZE";
        break;
    }
  return "";
}

cl_program
load_and_build_program (cl_context context,
                        cl_device_id device,
                        char *file_name)
{
  cl_program program;
  int program_size;
  cl_int err_num;

  FILE *program_f;

  char *program_buffer;

  program_f = fopen (file_name, "r");
  if (program_f == NULL)
    {
      fprintf (stderr, "%s not found\n", file_name);
      exit (1);
    }

  fseek (program_f, 0, SEEK_END);

  program_size = ftell (program_f);
  rewind (program_f);

  program_buffer = malloc (program_size + 1);
  program_buffer[program_size] = '\0';

  fread (program_buffer, sizeof (char), program_size, program_f);
  fclose (program_f);

  program = clCreateProgramWithSource (context, 1, (const char **)
                                       &program_buffer, NULL, &err_num);

  check_error (err_num, CL_SUCCESS);

  err_num = clBuildProgram (program, 0, NULL, NULL, NULL, NULL);
  if (err_num != CL_SUCCESS)
   {

     size_t size;

     err_num = clGetProgramBuildInfo (program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &size);
                                      check_error (err_num, CL_SUCCESS);

     char *log = malloc (size+1);

     err_num = clGetProgramBuildInfo (program, device, CL_PROGRAM_BUILD_LOG,
         size, log, NULL);

     fprintf (stderr, "%s\n", log);

     check_error (err_num, CL_SUCCESS);
   }

  return program;
}

void
check_error_file_line (int err_num,
                       int expected,
                       const char* file,
                       const int line_number)
{
  if (err_num != expected)
    {
      fprintf (stderr, "Line %d in File %s:", line_number, file);
      fprintf (stderr, "%s\n", error_desc (err_num));
      exit (1);
    }
}

cl_int
ocl_set_up_context (cl_device_type device_type,
                    cl_platform_id *platform,
                    cl_context *context,
                    cl_device_id *device,
                    cl_command_queue *command_queue)
{
  cl_int err_num;

  cl_context_properties contextProperties[] =
    {
      CL_CONTEXT_PLATFORM,
      (cl_context_properties)*platform,
      0
    };

  *context = clCreateContextFromType (contextProperties, device_type,
      NULL, NULL, &err_num);

  if (err_num != CL_SUCCESS)
    {
      /* FIXME add rest of the devices */
      if (device_type == CL_DEVICE_TYPE_CPU)
        printf ("No CPU devices found.\n");
      else if (device_type == CL_DEVICE_TYPE_GPU)
        printf ("No GPU devices found.\n");
    }

  err_num = clGetDeviceIDs (*platform, device_type, 1, device, NULL);

  if (err_num != CL_SUCCESS)
    {
      return err_num;
    }

  *command_queue = clCreateCommandQueue (*context, *device, 0, &err_num);

  if (err_num != CL_SUCCESS)
    {
      return err_num;
    }
  return 0;
}

void
ocl_init (oclDijkstraData *data,
                          gint matrix_size)
{
  if (data->platform == NULL)
    {
      cl_uint num_platforms;
      cl_int err_num;

      /* Find first OpenCL platform */
      err_num = clGetPlatformIDs (1, &(data->platform), &num_platforms);

      if (err_num != CL_SUCCESS || num_platforms <= 0)
        {
          printf ("Failed to find any OpenCL platforms.\n");
          return;
        }

      /* Set up context for GPU */
      err_num = ocl_set_up_context (CL_DEVICE_TYPE_CPU, &(data->platform), &(data->context), &(data->device), &(data->command_queue));
      check_error (err_num, CL_SUCCESS);

      /* Load an build OpenCL program */
      /* FIXME hardcoded file name */
      data->program = load_and_build_program (data->context, data->device,
          "/home/iaguis/igalia/Skeltrack/skeltrack/dijkstra.cl");

      /* Device buffers creation */
      data->edge_matrix_device = clCreateBuffer (data->context, CL_MEM_READ_ONLY, sizeof (gint)
          * matrix_size * NEIGHBOR_SIZE, NULL, &err_num);
      check_error (err_num, CL_SUCCESS);

      data->weight_matrix_device = clCreateBuffer (data->context, CL_MEM_READ_ONLY,
          sizeof (gint) * matrix_size * NEIGHBOR_SIZE, NULL, &err_num);
      check_error (err_num, CL_SUCCESS);

      data->mask_matrix_device = clCreateBuffer (data->context, CL_MEM_READ_WRITE, sizeof (gint)
          * matrix_size, NULL, &err_num);
      check_error (err_num, CL_SUCCESS);

      data->distance_matrix_device = clCreateBuffer (data->context, CL_MEM_READ_WRITE, sizeof (gint)
          * matrix_size, NULL, &err_num);
      check_error (err_num, CL_SUCCESS);

      data->updating_distance_matrix_device = clCreateBuffer (data->context, CL_MEM_READ_WRITE, sizeof (gint)
          * matrix_size, NULL, &err_num);
      check_error (err_num, CL_SUCCESS);

      data->previous_matrix_device = clCreateBuffer (data->context, CL_MEM_READ_WRITE,
          sizeof (gint) * matrix_size, NULL, &err_num);
      check_error (err_num, CL_SUCCESS);

      data->mask_matrix = g_slice_alloc (matrix_size * sizeof (gint));
      data->previous_matrix = g_slice_alloc (matrix_size * sizeof (gint));

      /* Create kernels */
      data->dijkstra_kernel1 = clCreateKernel (data->program, "dijkstra1", &err_num);
      check_error (err_num, CL_SUCCESS);

      data->dijkstra_kernel2 = clCreateKernel (data->program, "dijkstra2", &err_num);
      check_error (err_num, CL_SUCCESS);

      data->initialize_mask_kernel = clCreateKernel (data->program, "initialize_mask",
      &err_num);
      check_error (err_num, CL_SUCCESS);

      err_num |= clSetKernelArg (data->initialize_mask_kernel, 0, sizeof (cl_mem),
          &(data->mask_matrix_device));
    }
}

gint round_worksize_up(gint group_size, gint global_size)
{
  gint remainder = global_size % group_size;

  if (remainder == 0)
    {
      return global_size;
    }
  else
    {
      return global_size + group_size - remainder;
    }
}


gboolean
ocl_dijkstra_to (oclDijkstraData *data,
                 Node *source,
                 Node *target,
                 guint width,
                 guint height,
                 gint *distance_matrix,
                 Node **previous,
                 Node **node_matrix)
{
  gint source_vertex;
  gint matrix_size;
  gint i;
  /* 3 events */
  cl_event read_done[3];
  cl_uint err_num;
  size_t global_worksize, local_worksize, max_workgroup_size;

  source_vertex = source->j * width + source->i;
  matrix_size = width * height;

  /* Get maximum workgroup size */
  err_num = clGetDeviceInfo(data->device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
      sizeof(size_t), &max_workgroup_size, NULL);
  check_error(err_num, CL_SUCCESS);

  /* Set worksizes */
  local_worksize = max_workgroup_size;
  global_worksize = round_worksize_up(local_worksize, matrix_size);

  distance_matrix[source_vertex] = 0;

  err_num = CL_SUCCESS;
  /* Set kernel arguments */
  err_num |= clSetKernelArg (data->dijkstra_kernel1, 0, sizeof (cl_mem), &(data->edge_matrix_device));
  err_num |= clSetKernelArg (data->dijkstra_kernel1, 1, sizeof (cl_mem), &(data->weight_matrix_device));
  err_num |= clSetKernelArg (data->dijkstra_kernel1, 2, sizeof (cl_mem), &(data->mask_matrix_device));
  err_num |= clSetKernelArg (data->dijkstra_kernel1, 3, sizeof (cl_mem),
      &(data->distance_matrix_device));
  err_num |= clSetKernelArg (data->dijkstra_kernel1, 4, sizeof (cl_mem),
      &(data->updating_distance_matrix_device));
  err_num |= clSetKernelArg (data->dijkstra_kernel1, 5, sizeof (cl_mem), &(data->previous_matrix_device));
  err_num |= clSetKernelArg (data->dijkstra_kernel1, 6, sizeof (gint), &matrix_size);
  check_error (err_num, CL_SUCCESS);

  err_num |= clSetKernelArg (data->dijkstra_kernel2, 0, sizeof (cl_mem),
      &(data->distance_matrix_device));
  err_num |= clSetKernelArg (data->dijkstra_kernel2, 1, sizeof (cl_mem),
      &(data->updating_distance_matrix_device));
  err_num |= clSetKernelArg (data->dijkstra_kernel2, 2, sizeof (cl_mem),
      &(data->mask_matrix_device));
  err_num |= clSetKernelArg (data->dijkstra_kernel2, 3, sizeof (gint), &matrix_size);
  check_error (err_num, CL_SUCCESS);

  err_num |= clSetKernelArg (data->initialize_mask_kernel, 1, sizeof (cl_mem),
      &(data->previous_matrix_device));

  err_num |= clSetKernelArg (data->initialize_mask_kernel, 2, sizeof (gint),
      &source_vertex);
  err_num |= clSetKernelArg (data->initialize_mask_kernel, 3, sizeof (gint), &matrix_size);
  check_error (err_num, CL_SUCCESS);

  /* Copy new data to device */
  /* FIXME maybe {edge, weight}_matrix is not needed here? */
  err_num = clEnqueueWriteBuffer (data->command_queue,
      data->distance_matrix_device, CL_FALSE, 0, sizeof (gint) * matrix_size, distance_matrix,
      0, NULL, NULL);

  err_num = clEnqueueWriteBuffer (data->command_queue,
      data->updating_distance_matrix_device, CL_FALSE, 0, sizeof (gint) * matrix_size, distance_matrix,
      0, NULL, NULL);

  err_num = clEnqueueWriteBuffer (data->command_queue, data->edge_matrix_device,
      CL_FALSE, 0, sizeof (gint) * matrix_size * 8, data->edge_matrix, 0, NULL,
      NULL);

  err_num = clEnqueueWriteBuffer (data->command_queue, data->weight_matrix_device,
      CL_FALSE, 0, sizeof (gint) * matrix_size * 8, data->weight_matrix, 0,
      NULL, NULL);

  err_num = clEnqueueNDRangeKernel (data->command_queue, data->initialize_mask_kernel, 1, NULL,
  &global_worksize, &local_worksize, 0, NULL, NULL);
  check_error (err_num, CL_SUCCESS);

  err_num = clEnqueueReadBuffer (data->command_queue, data->mask_matrix_device, CL_FALSE, 0,
      sizeof (guint) * matrix_size, data->mask_matrix, 0, NULL, &read_done[0]);
  check_error (err_num, CL_SUCCESS);

  clWaitForEvents (1, &read_done[0]);

  while (!mask_array_empty (data->mask_matrix, matrix_size))
    {
      err_num = clEnqueueNDRangeKernel (data->command_queue, data->dijkstra_kernel1, 1, NULL,
          &global_worksize, &local_worksize, 0, NULL, NULL);
      check_error (err_num, CL_SUCCESS);

      err_num = clEnqueueNDRangeKernel (data->command_queue, data->dijkstra_kernel2, 1, NULL,
          &global_worksize, &local_worksize, 0, NULL, NULL);
      check_error (err_num, CL_SUCCESS);

      err_num = clEnqueueReadBuffer (data->command_queue, data->mask_matrix_device, CL_FALSE, 0,
      sizeof (guint) * matrix_size, data->mask_matrix, 0, NULL, &read_done[0]);
      check_error (err_num, CL_SUCCESS);

      clWaitForEvents (1, &read_done[0]);

    }
  err_num = clEnqueueReadBuffer (data->command_queue, data->distance_matrix_device, CL_FALSE, 0,
            sizeof (gint) * matrix_size, distance_matrix, 0, NULL, &read_done[1]);

  check_error (err_num, CL_SUCCESS);

  err_num = clEnqueueReadBuffer (data->command_queue, data->previous_matrix_device, CL_FALSE, 0,
            sizeof (gint) * matrix_size, data->previous_matrix, 0, NULL,
            &read_done[2]);

  check_error (err_num, CL_SUCCESS);

  clWaitForEvents (2, &read_done[1]);

  if (previous)
    {
      for (i=0; i<matrix_size; i++)
        {
          previous[i] = node_matrix[(data->previous_matrix)[i]];
        }
    }
  return FALSE;
}
