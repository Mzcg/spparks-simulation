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

#ifdef COMMAND_CLASS
CommandStyle(shell,Shell)

#else

#ifndef SPK_SHELL_H
#define SPK_SHELL_H

#include "pointers.h"

namespace SPPARKS_NS {

class Shell : protected Pointers {
 public:
  Shell(class SPPARKS *);
  void command(int, char **);
};

}

#endif
#endif
