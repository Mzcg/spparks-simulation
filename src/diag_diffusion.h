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

#ifndef SPK_DIAG_DIFFUSION_H
#define SPK_DIAG_DIFFUSION_H

#include "stdio.h"
#include "diag.h"

namespace SPPARKS_NS {

class DiagDiffusion : public Diag {

 public:
  DiagDiffusion(class SPPARKS *, int, char **);
  ~DiagDiffusion() {}
  void init();
  void compute();
  void stats(char *);
  void stats_header(char *);

 private:
  class AppDiffusion2 *appdiff;
  double deposit_success,deposit_failed;
  double nfirst_all,nsecond_all;
};

}

#endif