#define SCALE_FACTOR .0021
#define MIN_DISTANCE -10
#define NEIGHBOR_SIZE 8

void
convert_screen_coords_to_mm (int width,
                             int height,
                             int dimension_reduction,
                             int i,
                             int j,
                             int z,
                             int *x,
                             int *y)
{
  /* Formula from http://openkinect.org/wiki/Imaging_Information */
  *x = round((i * dimension_reduction - width * dimension_reduction / 2.0) *
             (z + MIN_DISTANCE) * SCALE_FACTOR * (width / height));
  *y = round((j * dimension_reduction - height * dimension_reduction / 2.0) *
             (z + MIN_DISTANCE) * SCALE_FACTOR);
}

int
get_distance (int a_x,
              int a_y,
              int a_z,
              int b_x,
              int b_y,
              int b_z)
{
  int dx, dy, dz;

  dx = abs (a_x - b_x);
  dy = abs (a_y - b_y);
  dz = abs (a_z - b_z);

  return sqrt ((float) (dx * dx + dy * dy + dz * dz));
}

__kernel void
initialize_graph (__global unsigned int *labels,
                  __global int *edge_matrix,
                  __global int *weight_matrix,
                  int size)
{
  int tid = get_global_id (0);

  if (tid < size)
    {
      labels[tid] = tid;
      for (int i=0; i < NEIGHBOR_SIZE; i++)
        {
          edge_matrix[(tid * NEIGHBOR_SIZE) + i] = -1;
          weight_matrix[(tid * NEIGHBOR_SIZE) + i] = -1;
        }
    }
}

/* Based in K. A. Hawick, A. Leist, and D. P. Playne paper
   Parallel graph component labelling with GPUs and CUDA */
__kernel void
mesh_kernel (__global unsigned short *buffer,
             __global unsigned int *labels,
             __global int *mD,
             int width,
             int height,
             __local unsigned int *labels_local)
{
  int id, idL, label, workgroup_size, i, j, z, block_start, index;
  int nId[NEIGHBOR_SIZE];
  int size;

  size = width * height;

  workgroup_size = get_local_size (1);

  block_start = workgroup_size * width * get_group_id (1) + get_group_id (0) *
    workgroup_size;

  id = block_start + get_local_id (1) * width + get_local_id (0);

  i = get_global_id (1);
  j = get_global_id (0);

  index = 0;

  if (id < size)
    {
      for (int k = (i-1); k <= (i+1); k++)
        {
          for (int l = (j-1); l <= (j+1); l++)
            {
              if (k >= 0 && k < height && l >= 0 && l < width && (k != i || l != j))
                {
                  unsigned int neighbor = k * width + l;

                  if ((buffer[id] == 0) && (buffer[neighbor] == 0))
                    {
                      nId[index] = neighbor;
                      index++;
                    }
                  else 
                  if ((buffer[id] != 0) && (buffer[neighbor] != 0))
                    {
                      nId[index] = neighbor;
                      index++;
                    }
                }
            }
        }

      if (buffer[id] == 0)
        {
          label = 0;
        }
      else
        {
          label = labels[id];
        }

    for (int z = 0; z < index; z++)
      {
        /* Check this out, may be wrong */
        if (buffer[nId[z]] == 0)
          {
            label = 0;
          }
        if (labels[nId[z]] < label)
          {
            label = labels[nId[z]];
            atomic_xchg (mD, 1);
          }
      }
      labels[id] = label;
    }

  /* Label block in local memory */
  __local int mL;

  i = get_local_id (1);
  j = get_local_id (0);

  idL = i * workgroup_size + j;
  mL = 1;

  index = 0;

  if (id < size)
    {
      for (int k = (i-1);  k <= (i+1); k++)
        {
          for (int l = (j-1); l <= (j+1); l++)
            {
              if (k >= 0 && k < workgroup_size && l >= 0 && l < workgroup_size && (k != i || l != j))
                {
                  unsigned int neighbor_local = k * workgroup_size + l;
                  unsigned int neighbor = block_start + k * width + l;
                  
                  if (neighbor < size)
                    {
                      if ((buffer[id] == 0) && (buffer[neighbor] == 0))
                        {
                          nId[index] = neighbor_local;
                          index++;
                        }
                      else
                      if ((buffer[id] != 0) && (buffer[neighbor] != 0))
                        {
                          nId[index] = neighbor_local;
                          index++;
                        }
                    }
                }
            }
        }
    }

  while (mL)
    {
      labels_local[idL] = label;

      barrier (CLK_LOCAL_MEM_FENCE);
      mL = 0;
      for (int i = 0; i < index; i++)
        {
          if (labels_local[nId[i]] < label)
            {
              label = labels_local[nId[i]];
              mL = 1;
            }
        }
      barrier (CLK_LOCAL_MEM_FENCE);
    }
    
    if (id < size)
      {
        labels[id] = label;
      }
}

__kernel void
make_graph (__global unsigned short *buffer,
            __global unsigned int *labels,
            __global int *edge_matrix,
            __global int *weight_matrix,
            int width,
            int height,
            int label)
{
  int i, j, node, size, index;

  size = width * height;

  i = get_global_id (1);
  j = get_global_id (0);

  node = i * width + j;

  if (node < size)
    {
      index = 0;

      if (labels[node] == label)
        {
        for (int k=(i-1); k<=(i+1); k++)
          {
            for (int l=(j-1); l<=(j+1); l++)
              {
                if (k >= 0 && k < height && l >= 0 && l < width && (k != i || l != j))
                  {
                    unsigned int neighbor = k * width + l;

                    if (labels[neighbor] == label)
                      {
                        int x_node, y_node, z_node, x_neigh, y_neigh, z_neigh;

                        z_node = buffer[node];
                        z_neigh = buffer[neighbor];

                        convert_screen_coords_to_mm (width, height, 16, i, j, z_node, &x_node, &y_node);
                        convert_screen_coords_to_mm (width, height, 16, k, l, z_neigh, &x_neigh, &y_neigh);

                        edge_matrix[node * NEIGHBOR_SIZE + index] = neighbor;
                        weight_matrix[node * NEIGHBOR_SIZE + index] = get_distance (x_node, y_node, z_node, x_neigh, y_neigh, z_neigh);
                        index++;
                      }
                  }
              }
           }
        }
    }
}
