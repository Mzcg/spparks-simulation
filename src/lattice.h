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

#ifndef SPK_LATTICE_H
#define SPK_LATTICE_H

#include "pointers.h"

namespace SPPARKS_NS {

class Lattice : protected Pointers {
 public:
  int style;                           // enum list of NONE,SC,FCC,etc
  double xlattice,ylattice,zlattice;   // lattice scale factors in 3 dims
  double a1[3],a2[3],a3[3];            // edge vectors of unit cell
  int nbasis;                          // # of basis atoms in unit cell
  double **basis;                      // fractional coords of each basis atom
                                       // within unit cell (0 <= coord < 1)
  int nrandom;                         // # of sites for random lattices
  double cutoff;                       // neighbor cutoff for random lattices

  Lattice(class SPPARKS *, int, char **);
  ~Lattice();
  int ncolors(int, int, int, int);
  int id2color(int, int, int, int, int);

private:
  double latconst;                     // lattice constant
  double origin[3];                    // lattice origin
  int orientx[3];                      // lattice orientation vecs
  int orienty[3];                      // orientx = what lattice dir lies
  int orientz[3];                      //           along x dim in box

  void add_basis(double, double, double);
};

}

#endif
