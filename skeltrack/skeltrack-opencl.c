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
ocl_init_labeling (oclData *data,
                   gint matrix_size)
{
  cl_int err_num;

  /* Load an build OpenCL program */
  /* FIXME hardcoded file name */
  data->program = load_and_build_program (data->context, data->device,
      "/home/iaguis/igalia/Skeltrack/skeltrack/ccl.cl");

  /* Device buffers creation */
  data->buffer_matrix_device = clCreateBuffer (data->context,
      CL_MEM_READ_ONLY, sizeof (guint16) * matrix_size, NULL, &err_num);
  check_error (err_num, CL_SUCCESS);

  data->labels_matrix_device = clCreateBuffer (data->context,
      CL_MEM_READ_WRITE, sizeof (guint) * matrix_size, NULL, &err_num);
  check_error (err_num, CL_SUCCESS);

  data->mD_device = clCreateBuffer (data->context, CL_MEM_READ_WRITE,
      sizeof (gint), NULL, &err_num);
  check_error (err_num, CL_SUCCESS);

  data->edge_matrix_device = clCreateBuffer (data->context,
      CL_MEM_READ_WRITE, sizeof (gint) * matrix_size * NEIGHBOR_SIZE, NULL, &err_num);
  check_error (err_num, CL_SUCCESS);

  data->weight_matrix_device = clCreateBuffer (data->context,
      CL_MEM_READ_WRITE, sizeof (gint) * matrix_size * NEIGHBOR_SIZE, NULL, &err_num);
  check_error (err_num, CL_SUCCESS);

  data->close_node_device = clCreateBuffer (data->context, CL_MEM_READ_WRITE,
      sizeof (gint) * 2, NULL, &err_num);
  check_error (err_num, CL_SUCCESS);

  data->equiv_list_device = clCreateBuffer (data->context, CL_MEM_READ_WRITE,
      sizeof (guint) * matrix_size, NULL, &err_num);
  check_error (err_num, CL_SUCCESS);

  /* Create kernels */
  data->initialize_graph_kernel = clCreateKernel (data->program,
      "initialize_graph", &err_num);
  check_error (err_num, CL_SUCCESS);

  data->mesh_kernel = clCreateKernel (data->program, "mesh_kernel", &err_num);
  check_error (err_num, CL_SUCCESS);

  data->make_graph_kernel = clCreateKernel (data->program, "make_graph",
      &err_num);
  check_error (err_num, CL_SUCCESS);

  data->join_to_biggest_kernel = clCreateKernel (data->program,
      "join_to_biggest", &err_num);
  check_error (err_num, CL_SUCCESS);

  data->mesh_kernel_2_init = clCreateKernel (data->program,
      "mesh_kernel_2_init", &err_num);
  check_error (err_num, CL_SUCCESS);

  data->mesh_kernel_2_scanning = clCreateKernel (data->program,
      "mesh_kernel_2_scanning", &err_num);
  check_error (err_num, CL_SUCCESS);

  data->mesh_kernel_2_analysis = clCreateKernel (data->program,
      "mesh_kernel_2_analysis", &err_num);
  check_error (err_num, CL_SUCCESS);

  data->mesh_kernel_2_labeling = clCreateKernel (data->program,
      "mesh_kernel_2_labeling", &err_num);
  check_error (err_num, CL_SUCCESS);
}

void
ocl_init (oclData *data,
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
      err_num = ocl_set_up_context (CL_DEVICE_TYPE_GPU, &(data->platform), &(data->context), &(data->device), &(data->command_queue));
      check_error (err_num, CL_SUCCESS);

      ocl_init_labeling (data, matrix_size);
    }
}

gint round_worksize_up (gint group_size, gint global_size)
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

void
ocl_ccl (oclData *data,
         guint16 *buffer,
         gint width,
         gint height,
         gint version)
{
  gint size;
  cl_int err_num;
  size_t local_worksize[2], global_worksize[2];
  size_t init_worksize;
  cl_event read_done;

  size = width * height;

  data->labels_matrix = g_slice_alloc (size * sizeof (guint));
  *(data->mD) = 0;

  local_worksize[0] = 16;
  local_worksize[1] = 16;
  global_worksize[0] = round_worksize_up (local_worksize[0], width);
  global_worksize[1] = round_worksize_up (local_worksize[1], height);

  err_num = CL_SUCCESS;

  if (version == 0)
    {
      err_num |= clSetKernelArg (data->mesh_kernel, 0, sizeof (cl_mem),
          &(data->buffer_matrix_device));
      err_num |= clSetKernelArg (data->mesh_kernel, 1, sizeof (cl_mem),
          &(data->labels_matrix_device));
      err_num |= clSetKernelArg (data->mesh_kernel, 2, sizeof (cl_mem),
          &(data->mD_device));
      err_num |= clSetKernelArg (data->mesh_kernel, 3, sizeof (gint),
          &width);
      err_num |= clSetKernelArg (data->mesh_kernel, 4, sizeof (gint),
          &height);
      err_num |= clSetKernelArg (data->mesh_kernel, 5, local_worksize[0] *
          local_worksize[1] * sizeof (guint), NULL);
      check_error (err_num, CL_SUCCESS);

      err_num |= clSetKernelArg (data->initialize_graph_kernel, 0, sizeof (cl_mem),
          &(data->labels_matrix_device));
      err_num |= clSetKernelArg (data->initialize_graph_kernel, 1, sizeof (cl_mem),
          &(data->edge_matrix_device));
      err_num |= clSetKernelArg (data->initialize_graph_kernel, 2, sizeof (cl_mem),
          &(data->weight_matrix_device));
      err_num |= clSetKernelArg (data->initialize_graph_kernel, 3, sizeof (gint), &size);
      check_error (err_num, CL_SUCCESS);

      // Copy new data to device
      err_num = clEnqueueWriteBuffer (data->command_queue,
          data->buffer_matrix_device, CL_FALSE, 0, sizeof (guint16) * size,
          buffer, 0, NULL, NULL);
      check_error (err_num, CL_SUCCESS);

      init_worksize = size;

      err_num = clEnqueueNDRangeKernel (data->command_queue,
          data->initialize_graph_kernel, 1, NULL, &init_worksize, NULL, 0, NULL, NULL);
      check_error (err_num, CL_SUCCESS);

      do
        {
          *(data->mD) = 0;
          err_num = clEnqueueWriteBuffer (data->command_queue, data->mD_device,
          CL_FALSE, 0, sizeof (gint), data->mD, 0, NULL, NULL);
          check_error (err_num, CL_SUCCESS);

          err_num = clEnqueueNDRangeKernel (data->command_queue, data->mesh_kernel,
              2, NULL, global_worksize, local_worksize, 0, NULL, NULL);
          check_error (err_num, CL_SUCCESS);

          err_num = clEnqueueReadBuffer (data->command_queue, data->mD_device, CL_FALSE, 0,
          sizeof (gint), data->mD, 0, NULL, &read_done);
          check_error (err_num, CL_SUCCESS);

          clWaitForEvents (1, &read_done);
          clReleaseEvent (read_done);
        } while (*(data->mD));
      err_num = clEnqueueReadBuffer (data->command_queue,
          data->labels_matrix_device, CL_FALSE, 0, sizeof (guint) * size,
          data->labels_matrix, 0, NULL, &read_done);
      check_error (err_num, CL_SUCCESS);

      clWaitForEvents (1, &read_done);
      clReleaseEvent (read_done);
    }
  else if (version == 1)
    {
      err_num |= clSetKernelArg (data->mesh_kernel_2_init, 0, sizeof (cl_mem),
          &(data->labels_matrix_device));
      err_num |= clSetKernelArg (data->mesh_kernel_2_init, 1, sizeof (cl_mem),
          &(data->equiv_list_device));
      err_num |= clSetKernelArg (data->mesh_kernel_2_init, 2, sizeof (cl_mem),
          &(data->edge_matrix_device));
      err_num |= clSetKernelArg (data->mesh_kernel_2_init, 3, sizeof (cl_mem),
          &(data->weight_matrix_device));
      check_error (err_num, CL_SUCCESS);
      err_num |= clSetKernelArg (data->mesh_kernel_2_init, 4, sizeof (gint),
          &size);
      check_error (err_num, CL_SUCCESS);

      err_num |= clSetKernelArg (data->mesh_kernel_2_scanning, 0, sizeof
          (cl_mem), &(data->buffer_matrix_device));
      err_num |= clSetKernelArg (data->mesh_kernel_2_scanning, 1, sizeof
          (cl_mem), &(data->labels_matrix_device));
      err_num |= clSetKernelArg (data->mesh_kernel_2_scanning, 2, sizeof
          (cl_mem), &(data->equiv_list_device));
      err_num |= clSetKernelArg (data->mesh_kernel_2_scanning, 3, sizeof
          (cl_mem), &(data->mD_device));
      err_num |= clSetKernelArg (data->mesh_kernel_2_scanning, 4, sizeof
          (gint), &width);
      err_num |= clSetKernelArg (data->mesh_kernel_2_scanning, 5, sizeof
          (gint), &height);
      check_error (err_num, CL_SUCCESS);

      err_num |= clSetKernelArg (data->mesh_kernel_2_analysis, 0, sizeof
          (cl_mem), &(data->labels_matrix_device));
      err_num |= clSetKernelArg (data->mesh_kernel_2_analysis, 1, sizeof
          (cl_mem), &(data->equiv_list_device));
      err_num |= clSetKernelArg (data->mesh_kernel_2_analysis, 2, sizeof
          (gint), &size);
      check_error (err_num, CL_SUCCESS);

      err_num |= clSetKernelArg (data->mesh_kernel_2_labeling, 0, sizeof
          (cl_mem), &(data->buffer_matrix_device));
      err_num |= clSetKernelArg (data->mesh_kernel_2_labeling, 1, sizeof
          (cl_mem), &(data->labels_matrix_device));
      err_num |= clSetKernelArg (data->mesh_kernel_2_labeling, 2, sizeof
          (cl_mem), &(data->equiv_list_device));
      err_num |= clSetKernelArg (data->mesh_kernel_2_labeling, 3, sizeof
          (gint), &size);
      check_error (err_num, CL_SUCCESS);

      // Copy new data to device
      err_num = clEnqueueWriteBuffer (data->command_queue,
          data->buffer_matrix_device, CL_FALSE, 0, sizeof (guint16) * size,
          buffer, 0, NULL, NULL);
      check_error (err_num, CL_SUCCESS);

      init_worksize = size;

      err_num = clEnqueueNDRangeKernel (data->command_queue,
          data->mesh_kernel_2_init, 1, NULL, &init_worksize, NULL, 0, NULL, NULL);
      check_error (err_num, CL_SUCCESS);

      do
        {
          *(data->mD) = 0;
          err_num = clEnqueueWriteBuffer (data->command_queue, data->mD_device,
          CL_FALSE, 0, sizeof (gint), data->mD, 0, NULL, NULL);
          check_error (err_num, CL_SUCCESS);

          err_num = clEnqueueNDRangeKernel (data->command_queue,
              data->mesh_kernel_2_scanning, 2, NULL, global_worksize,
              local_worksize, 0, NULL, NULL);
          check_error (err_num, CL_SUCCESS);

          err_num = clEnqueueNDRangeKernel (data->command_queue,
              data->mesh_kernel_2_analysis, 1, NULL, &init_worksize,
              NULL, 0, NULL, NULL);
          check_error (err_num, CL_SUCCESS);

          err_num = clEnqueueNDRangeKernel (data->command_queue,
              data->mesh_kernel_2_labeling, 1, NULL, &init_worksize,
              NULL, 0, NULL, NULL);
          check_error (err_num, CL_SUCCESS);

          err_num = clEnqueueReadBuffer (data->command_queue, data->mD_device, CL_FALSE, 0,
          sizeof (gint), data->mD, 0, NULL, &read_done);
          check_error (err_num, CL_SUCCESS);

          clWaitForEvents (1, &read_done);
          clReleaseEvent (read_done);
        } while (*(data->mD));
      err_num = clEnqueueReadBuffer (data->command_queue,
          data->labels_matrix_device, CL_FALSE, 0, sizeof (guint) * size,
          data->labels_matrix, 0, NULL, &read_done);
      check_error (err_num, CL_SUCCESS);

      clWaitForEvents (1, &read_done);
      clReleaseEvent (read_done);
    }

  return;
}

void
ocl_make_graph (oclData *data,
                gint width,
                gint height,
                gint label,
                gint dimension_reduction)
{
  cl_int err_num;
  cl_event read_done[2];
  gint size;

  size_t global_worksize[2], local_worksize[2];

  size = width * height;

  local_worksize[0] = 16;
  local_worksize[1] = 16;
  global_worksize[0] = round_worksize_up (local_worksize[0], width);
  global_worksize[1] = round_worksize_up (local_worksize[1], height);

  err_num = CL_SUCCESS;

  err_num |= clSetKernelArg (data->make_graph_kernel, 0, sizeof (cl_mem),
        &(data->buffer_matrix_device));
  err_num |= clSetKernelArg (data->make_graph_kernel, 1, sizeof (cl_mem),
        &(data->labels_matrix_device));
  err_num |= clSetKernelArg (data->make_graph_kernel, 2, sizeof (cl_mem),
        &(data->edge_matrix_device));
  err_num |= clSetKernelArg (data->make_graph_kernel, 3, sizeof (cl_mem),
        &(data->weight_matrix_device));
  err_num |= clSetKernelArg (data->make_graph_kernel, 4, sizeof (gint),
        &(width));
  err_num |= clSetKernelArg (data->make_graph_kernel, 5, sizeof (gint),
        &(height));
  err_num |= clSetKernelArg (data->make_graph_kernel, 6, sizeof (gint),
        &(label));
  err_num |= clSetKernelArg (data->make_graph_kernel, 7, sizeof (gint),
        &(dimension_reduction));
  check_error (err_num, CL_SUCCESS);

  err_num = clEnqueueNDRangeKernel (data->command_queue,
      data->make_graph_kernel, 2, NULL, global_worksize, local_worksize, 0,
      NULL, NULL);
  check_error (err_num, CL_SUCCESS);

  err_num = clEnqueueReadBuffer (data->command_queue, data->edge_matrix_device,
      CL_FALSE, 0, sizeof (gint) * size * NEIGHBOR_SIZE, data->edge_matrix, 0, NULL,
      &read_done[0]);
  check_error (err_num, CL_SUCCESS);

  err_num = clEnqueueReadBuffer (data->command_queue, data->weight_matrix_device,
      CL_FALSE, 0, sizeof (gint) * size * NEIGHBOR_SIZE, data->weight_matrix, 0, NULL,
      &read_done[1]);
  check_error (err_num, CL_SUCCESS);

  clWaitForEvents (2, read_done);
  clReleaseEvent (read_done[0]);
  clReleaseEvent (read_done[1]);
}

gint *
ocl_join_to_biggest (oclData *data,
                     gint i,
                     gint j,
                     gint biggest,
                     gint dist_x,
                     gint dist_y,
                     gint dist_z,
                     gint width,
                     gint height,
                     gint dimension_reduction)
{
  cl_int err_num;
  cl_event read_done;
  gint *close_node;

  size_t global_worksize[2];

  close_node = g_slice_alloc (sizeof(gint) * 2);

  close_node[0] = close_node[1] = -1;

  global_worksize[0] = width;
  global_worksize[1] = height;

  err_num = CL_SUCCESS;

  err_num = clEnqueueWriteBuffer (data->command_queue,
      data->close_node_device, CL_FALSE, 0, sizeof (gint) * 2, close_node,
      0, NULL, NULL);

  err_num |= clSetKernelArg (data->join_to_biggest_kernel, 0, sizeof (cl_mem),
      &(data->labels_matrix_device));
  err_num |= clSetKernelArg (data->join_to_biggest_kernel, 1, sizeof (cl_mem),
      &(data->buffer_matrix_device));
  err_num |= clSetKernelArg (data->join_to_biggest_kernel, 2, sizeof (cl_mem),
      &(data->close_node_device));

  err_num |= clSetKernelArg (data->join_to_biggest_kernel, 3, sizeof (int),
      &(i));
  err_num |= clSetKernelArg (data->join_to_biggest_kernel, 4, sizeof (int),
      &(j));
  err_num |= clSetKernelArg (data->join_to_biggest_kernel, 5, sizeof (int),
      &(biggest));
  err_num |= clSetKernelArg (data->join_to_biggest_kernel, 6, sizeof (int),
      &(dist_x));
  err_num |= clSetKernelArg (data->join_to_biggest_kernel, 7, sizeof (int),
      &(dist_y));
  err_num |= clSetKernelArg (data->join_to_biggest_kernel, 8, sizeof (int),
      &(dist_z));
  err_num |= clSetKernelArg (data->join_to_biggest_kernel, 9, sizeof (int),
      &(width));
  err_num |= clSetKernelArg (data->join_to_biggest_kernel, 10, sizeof (int),
      &(height));
  err_num |= clSetKernelArg (data->join_to_biggest_kernel, 11, sizeof (gint),
      &(dimension_reduction));
  check_error (err_num, CL_SUCCESS);

  err_num = clEnqueueNDRangeKernel (data->command_queue,
      data->join_to_biggest_kernel, 2, NULL, global_worksize, NULL, 0, NULL,
      NULL);
  check_error (err_num, CL_SUCCESS);

  err_num = clEnqueueReadBuffer (data->command_queue, data->close_node_device,
      CL_FALSE, 0, sizeof (gint) * 2, close_node, 0, NULL, &read_done);
  check_error (err_num, CL_SUCCESS);

  clWaitForEvents (1, &read_done);
  clReleaseEvent (read_done);

  return close_node;
}
