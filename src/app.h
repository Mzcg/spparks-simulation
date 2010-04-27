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

#ifndef SPK_APP_H
#define SPK_APP_H

#include "pointers.h"

namespace SPPARKS_NS {
  
class App : protected Pointers {
 public:
  enum APP_CLASSES{GENERAL,LATTICE,OFF_LATTICE};

  int appclass;           // one of the enum values
  char *style;            // style name of app
  double time;            // current simulation time due to executed events
  double stoptime;        // time at which to stop this run
  int sites_exist;        // 1 if sites have been created

  // owned + ghost sites

  int nglobal;                 // global # of sites
  int nlocal;                  // # of sites I own
  int nghost;                  // # of ghost sites I store

  int ninteger,ndouble;        // # of ints and doubles per site
  int *id;                     // global ID of site
  double **xyz;                // coords of site
  int **iarray;                // one or more ints per site
  double **darray;             // one or more doubles per site

  App(class SPPARKS *, int, char **);
  virtual ~App();
  void run(int, char **);
  void reset_time(double);

  // pure virtual functions, must be defined in child class
  
  virtual void input(char *, int, char **) = 0;
  virtual void init() = 0;
  virtual void setup() = 0;
  virtual void iterate() = 0;

  // virtual functions, may be overridden in child class

  virtual void stats(char *strtmp) {strtmp[0] = '\0';};
  virtual void stats_header(char *strtmp) {strtmp[0] = '\0';};
  virtual void *extract(char *) {return NULL;};

 protected:
  int first_run;
  double nextoutput;

  void create_arrays();
  void recreate_arrays();
};

}

#endif
