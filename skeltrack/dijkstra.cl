/* Dijkstra implementation based on Pawan Harish and P. J. Narayanan paper
   Accelerating large graph algorithms on the GPU using CUDA */

__kernel void
dijkstra1 (__global int *edge_matrix,
           __global int *weight_matrix,
           __global int *mask_matrix,
           __global int *distance_matrix,
           __global int *updating_distance_matrix,
           __global int *previous,
           int vertex_count)
{
  int tid = get_global_id (0);

  if (tid < vertex_count) {
    if (mask_matrix[tid] != 0)
      {
        mask_matrix[tid] = 0;

        int edgeStart = tid * 8;
        int edge;
        int nid;

        for (edge = edgeStart; ((edge - edgeStart) < 8) && ((nid = edge_matrix[edge]) !=
              -1); edge++)
          {
            if (updating_distance_matrix[nid] == -1 ||
                updating_distance_matrix[nid] > (distance_matrix[tid] + weight_matrix[edge]))
              {
                updating_distance_matrix[nid] = (distance_matrix[tid] + weight_matrix[edge]);
                if (previous)
                  {
                    previous[nid] = tid;
                  }
              }
          }
      }
  }
}

__kernel void
dijkstra2 (__global int *distance_matrix,
           __global int *updating_distance_matrix,
           __global int *mask_matrix,
           int vertex_count)
{
  int tid = get_global_id(0);

  if (tid < vertex_count)
    {
      if ((distance_matrix[tid] == -1 && updating_distance_matrix[tid] != -1) ||
          distance_matrix[tid] > updating_distance_matrix[tid])
        {
          distance_matrix[tid] = updating_distance_matrix[tid];
          mask_matrix[tid] = 1;
        }

      updating_distance_matrix[tid] = distance_matrix[tid];
    }
}

__kernel void
initialize_mask (__global int *mask_matrix,
                 __global int *previous,
                 int source_vertex,
                 int vertex_count)
{
  int tid = get_global_id(0);

  if (tid < vertex_count)
    {
      if (previous != 0)
        previous[tid] = -1;
      if (tid == source_vertex)
        {
          mask_matrix[tid] = 1;
        }
      else
        {
          mask_matrix[tid] = 0;
        }
    }
}

__kernel void
flush_distance_matrix (__global int *distance_matrix,
                       __global int *updating_distance_matrix,
                       int vertex_count)
{
  int tid = get_global_id (0);

  if (tid < vertex_count)
    {
      distance_matrix[tid] = -1;
      updating_distance_matrix[tid] = -1;
    }
}

__kernel void
set_source_vertex (__global int *distance_matrix,
                   int source_vertex)
{
  int tid = get_global_id (0);

  if (tid == source_vertex)
    {
      distance_matrix[tid] = 0;
    }
}
