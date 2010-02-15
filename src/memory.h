/* ----------------------------------------------------------------------
   SPPARKS - Stochastic Parallel PARticle Kinetic Simulator
   http://www.cs.sandia.gov/~sjplimp/spparks.html
   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories

   Copyright (2008) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level SPPARKS directory.
------------------------------------------------------------------------- */

#ifndef SPK_MEMORY_H
#define SPK_MEMORY_H

#include "pointers.h"

namespace SPPARKS_NS {

class Memory : protected Pointers {
 public:
  Memory(class SPPARKS *);

  void *smalloc(long long int n, const char *);
  void sfree(void *);
  void *srealloc(void *, long long int n, const char *name);

  double *create_1d_double_array(int, int, const char *);
  void destroy_1d_double_array(double *, int);
  
  double **create_2d_double_array(int, int, const char *);
  void destroy_2d_double_array(double **);
  double **grow_2d_double_array(double **, int, int, const char *);

  int **create_2d_int_array(int, int, const char *);
  int **create_2d_int_ragged_array(int, int *, const char *);
  void destroy_2d_int_array(int **);
  int **grow_2d_int_array(int **, int, int, const char *);

  double **create_2d_double_array(int, int, int, const char *);
  void destroy_2d_double_array(double **, int);

  double ***create_3d_double_array(int, int, int, const char *);
  void destroy_3d_double_array(double ***);
  double ***grow_3d_double_array(double ***, int, int, int, const char *);

  double ***create_3d_double_array(int, int, int, int, const char *);
  void destroy_3d_double_array(double ***, int);

  double ***create_3d_double_array(int, int, int, int, int, int, const char *);
  void destroy_3d_double_array(double ***, int, int, int);

  int ***create_3d_int_array(int, int, int, const char *);
  void destroy_3d_int_array(int ***);

  double ****create_4d_double_array(int, int, int, int, const char *);
  void destroy_4d_double_array(double ****);

  // Templated versions

  template<typename T>
    void create_1d_T_array(T *&, int, int, const char *);

  template<typename T>
    void destroy_1d_T_array(T *, int);
  
  template<typename T>
    void create_2d_T_array(T **&, int, int, const char *);

  template<typename T>
    void destroy_2d_T_array(T **);

  template<typename T>
    void grow_2d_T_array(T **&, int, int, const char *);

  template<typename T>
    void create_2d_T_array(T **&, int, int, int, const char *);

  template<typename T>
    void destroy_2d_T_array(T **, int);
    
  template<typename T>
    void create_2d_T_array(T **&, int, int, int, int, const char *);

  template<typename T>
    void destroy_2d_T_array(T **, int, int);
    
  template<typename T>
    void create_3d_T_array(T ***&, int, int, int, const char *);

  template<typename T>
    void destroy_3d_T_array(T ***);

  template<typename T>
    void grow_3d_T_array(T ***&, int, int, int, const char *);
    
  template<typename T>
    void create_3d_T_array(T ***&, int, int, int, int, const char *);

  template<typename T>
    void destroy_3d_T_array(T ***, int);
    
  template<typename T>
    void create_3d_T_array(T ***&, int, int, int, int, int, int, const char *);

  template<typename T>
    void destroy_3d_T_array(T ***, int, int, int);
    
  template<typename T>
    void create_4d_T_array(T ****&, int, int, int, int, const char *);

  template<typename T>
    void destroy_4d_T_array(T ****);
};

// Non-inline function definitions still need to go in the header file

/* ----------------------------------------------------------------------
   create a 1d T array with index from nlo to nhi inclusive 
------------------------------------------------------------------------- */

template<typename T>
void Memory::create_1d_T_array(T *&array, int nlo, int nhi, const char *name)
{
  int n = nhi - nlo + 1;
  array = (T *) smalloc(n*sizeof(T),name);
  array = array-nlo;
}

/* ----------------------------------------------------------------------
   free a 1d T array with index offset 
------------------------------------------------------------------------- */

template<typename T>
void Memory::destroy_1d_T_array(T *array, int offset)
{
  if (array == NULL) return;
  sfree(array + offset);
}

/* ----------------------------------------------------------------------
   create a 2d T array 
------------------------------------------------------------------------- */

// Templating does not work on return-type.
// We achieve the desired result by moving the return value 
// into the argument list as a reference.
template<typename T>
void Memory::create_2d_T_array(T **&array, int n1, int n2, const char *name)
{
  T *data = (T *) smalloc(n1*n2*sizeof(T),name);
  array = (T **) smalloc(n1*sizeof(T *),name);

  int n = 0;
  for (int i = 0; i < n1; i++) {
    array[i] = &data[n];
    n += n2;
  }
}

/* ----------------------------------------------------------------------
   free a 2d T array 
------------------------------------------------------------------------- */

template<typename T>
void Memory::destroy_2d_T_array(T **array)
{
  if (array == NULL) return;
  sfree(array[0]);
  sfree(array);
}

/* ----------------------------------------------------------------------
   grow or shrink 1st dim of a 2d T array
   last dim must stay the same
   if either dim is 0, return NULL 
------------------------------------------------------------------------- */

template<typename T>
void Memory::grow_2d_T_array(T **&array,
			     int n1, int n2, const char *name)

{
  if (n1 == 0 || n2 == 0) {
    destroy_2d_T_array(array);
    array = NULL;
    return;
  }

  if (array == NULL) {
    create_2d_T_array(array,n1,n2,name);
    return;
  }

  T *data = (T *) srealloc(array[0],n1*n2*sizeof(T),name);
  sfree(array);
  array = (T **) smalloc(n1*sizeof(T *),name);

  int n = 0;
  for (int i = 0; i < n1; i++) {
    array[i] = &data[n];
    n += n2;
  }
}

/* ----------------------------------------------------------------------
   create a 2d T array with 2nd index from n2lo to n2hi inclusive 
------------------------------------------------------------------------- */

template<typename T>
 void Memory::create_2d_T_array(T **&array, int n1, int n2lo, int n2hi, const char *name)
{
  int n2 = n2hi - n2lo + 1;
  create_2d_T_array(array,n1,n2,name);

  for (int i = 0; i < n1; i++) array[i] -= n2lo;
}

/* ----------------------------------------------------------------------
   free a 2d T array with 2nd index offset 
------------------------------------------------------------------------- */

template<typename T>
void Memory::destroy_2d_T_array(T **array, int offset)
{
  if (array == NULL) return;
  sfree(&array[0][offset]);
  sfree(array);
}

/* ----------------------------------------------------------------------
   create a 2d T array with indexes from n1lo to n1hi and n2lo to n2hi inclusive 
------------------------------------------------------------------------- */

template<typename T>
  void Memory::create_2d_T_array(T **&array,
				 int n1lo, int n1hi, int n2lo, int n2hi,
				 const char *name)
{
  int n1 = n1hi - n1lo + 1;
  int n2 = n2hi - n2lo + 1;
  create_2d_T_array(array,n1,n2,name);

  for (int i = 0; i < n1; i++) array[i] -= n2lo;
  array -= n1lo;
}

/* ----------------------------------------------------------------------
   free a 2d T array with 1st and 2nd index offset 
------------------------------------------------------------------------- */

template<typename T>
void Memory::destroy_2d_T_array(T **array, int n1lo, int n2lo)
{
  if (array == NULL) return;
  sfree(&array[n1lo][n2lo]);
  sfree(array+n1lo);
}

/* ----------------------------------------------------------------------
   create a 3d T array 
------------------------------------------------------------------------- */

template<typename T>
void Memory::create_3d_T_array(T ***&array,
			       int n1, int n2, int n3, const char *name)
{
  int i,j;

  T *data = (T *) smalloc(n1*n2*n3*sizeof(T),name);
  T **plane = (T **) smalloc(n1*n2*sizeof(T *),name);
  array = (T ***) smalloc(n1*sizeof(T **),name);

  int n = 0;
  for (i = 0; i < n1; i++) {
    array[i] = &plane[i*n2];
    for (j = 0; j < n2; j++) {
      plane[i*n2+j] = &data[n];
      n += n3;
    }
  }

}

/* ----------------------------------------------------------------------
   free a 3d T array 
------------------------------------------------------------------------- */

template<typename T>
void Memory::destroy_3d_T_array(T ***array)
{
  if (array == NULL) return;
  sfree(array[0][0]);
  sfree(array[0]);
  sfree(array);
}

/* ----------------------------------------------------------------------
   grow or shrink 1st dim of a 3d T array
   last 2 dims must stay the same
   if any dim is 0, return NULL 
------------------------------------------------------------------------- */

template<typename T>
void Memory::grow_3d_T_array(T ***&array,
			     int n1, int n2, int n3, const char *name)
{
  int i,j;

  if (n1 == 0 || n2 == 0 || n3 == 0) {
    destroy_3d_T_array(array);
    array = NULL;
    return;
  }

  if (array == NULL) {
    create_3d_T_array(array,n1,n2,n3,name);
    return;
  }

  T *data = (T *) srealloc(array[0][0],n1*n2*n3*sizeof(T),name);
  sfree(array[0]);
  T **plane = (T **) smalloc(n1*n2*sizeof(T *),name);
  sfree(array);
  array = (T ***) smalloc(n1*sizeof(T **),name);

  int n = 0;
  for (i = 0; i < n1; i++) {
    array[i] = &plane[i*n2];
    for (j = 0; j < n2; j++) {
      plane[i*n2+j] = &data[n];
      n += n3;
    }
  }
}

/* ----------------------------------------------------------------------
   create a 3d T array with 1st index from n1lo to n1hi inclusive 
------------------------------------------------------------------------- */

template<typename T>
void Memory::create_3d_T_array(T ***&array, int n1lo, int n1hi, 
			       int n2, int n3, const char *name)
{
  int n1 = n1hi - n1lo + 1;
  create_3d_T_array(array,n1,n2,n3,name);
  array = array-n1lo;
}

/* ----------------------------------------------------------------------
   free a 3d T array with 1st index offset 
------------------------------------------------------------------------- */

template<typename T>
void Memory::destroy_3d_T_array(T ***array, int offset)
{
  if (array) destroy_3d_T_array(array + offset);
}

/* ----------------------------------------------------------------------
   create a 3d T array with
   1st index from n1lo to n1hi inclusive,
   2nd index from n2lo to n2hi inclusive,
   3rd index from n3lo to n3hi inclusive 
------------------------------------------------------------------------- */

template<typename T>
void Memory::create_3d_T_array(T ***&array, int n1lo, int n1hi,
					 int n2lo, int n2hi,
					 int n3lo, int n3hi, const char *name)
{
  int n1 = n1hi - n1lo + 1;
  int n2 = n2hi - n2lo + 1;
  int n3 = n3hi - n3lo + 1;
  create_3d_T_array(array,n1,n2,n3,name);

  for (int i = 0; i < n1*n2; i++) array[0][i] -= n3lo;
  for (int i = 0; i < n1; i++) array[i] -= n2lo;
  array = array-n1lo;
}

/* ----------------------------------------------------------------------
   free a 3d T array with all 3 indices offset 
------------------------------------------------------------------------- */

template<typename T>
void Memory::destroy_3d_T_array(T ***array, int n1_offset,
				     int n2_offset, int n3_offset)
{
  if (array == NULL) return;
  sfree(&array[n1_offset][n2_offset][n3_offset]);
  sfree(&array[n1_offset][n2_offset]);
  sfree(array + n1_offset);
}

/* ----------------------------------------------------------------------
   create a 4d T array 
------------------------------------------------------------------------- */

template<typename T>
void Memory::create_4d_T_array(T ****&array, int n1, int n2, int n3, int n4, const char *name)
{
  int i,j,k;

  T *data = (T *) smalloc(n1*n2*n3*n4*sizeof(T),name);
  T **cube = (T **) smalloc(n1*n2*n3*sizeof(T *),name);
  T ***plane = (T ***) smalloc(n1*n2*sizeof(T **),name);
  array = (T ****) smalloc(n1*sizeof(T ***),name);

  int n = 0;
  for (i = 0; i < n1; i++) {
    array[i] = &plane[i*n2];
    for (j = 0; j < n2; j++) {
      plane[i*n2+j] = &cube[i*n2*n3+j*n3];
      for (k = 0; k < n3; k++) {
	cube[i*n2*n3+j*n3+k] = &data[n];
	n += n4;
      }
    }
  }

}

/* ----------------------------------------------------------------------
   free a 4d T array 
------------------------------------------------------------------------- */

template<typename T>
void Memory::destroy_4d_T_array(T ****array)
{
  if (array == NULL) return;
  sfree(array[0][0][0]);
  sfree(array[0][0]);
  sfree(array[0]);
  sfree(array);
}

}

#endif
