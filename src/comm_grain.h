/* ----------------------------------------------------------------------
   SPPARKS - Stochastic Parallel PARticle Kinetic Simulator
   contact info, copyright info, etc
------------------------------------------------------------------------- */

#ifndef COMM_GRAIN_H
#define COMM_GRAIN_H

#include "sysptr.h"

namespace SPPARKS {

class CommGrain : protected SysPtr  {
 public:
  explicit CommGrain(class SPK *);
  ~CommGrain();
};

}

#endif