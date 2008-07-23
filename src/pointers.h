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

// Pointesr class contains ptrs to master copy of
//   fundamental SPPARKS class ptrs stored in spparks.h
// top-level SPPARKS classes inherit from Pointers to access spparks.h ptrs
// these variables are auto-initialized by Pointers class constructor
// *& variables are really pointers to the pointers in spparks.h
// & enables them to be accessed directly in any class, e.g. error->all()

#ifndef POINTERS_H
#define POINTERS_H

#include "mpi.h"
#include "stdio.h"
#include "spparks.h"

namespace SPPARKS_NS {

class Pointers {
public:
  Pointers(SPPARKS *ptr) :
    spk(ptr),
    universe(ptr->universe),
    input(ptr->input),
    memory(ptr->memory),
    error(ptr->error),
    app(ptr->app),
    solve(ptr->solve),
    sweep(ptr->sweep),
    timer(ptr->timer),
    output(ptr->output),
    world(ptr->world),
    infile(ptr->infile),
    screen(ptr->screen),
    logfile(ptr->logfile) {}
  virtual ~Pointers() {}
  
 protected:
  SPPARKS *spk;
  Memory *&memory;
  Error *&error;
  Universe *&universe;
  Input *&input;

  App *&app;
  Solve *&solve;
  Sweep *&sweep;
  Output *&output;
  Timer *&timer;

  MPI_Comm &world;
  FILE *&infile;
  FILE *&screen;
  FILE *&logfile;
};

}

#endif

