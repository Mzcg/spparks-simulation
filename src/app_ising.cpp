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

#include "math.h"
#include "mpi.h"
#include "string.h"
#include "stdlib.h"
#include "app_ising.h"
#include "comm_lattice.h"
#include "solve.h"
#include "random_park.h"
#include "timer.h"
#include "memory.h"
#include "error.h"

#include <map>

using namespace SPPARKS_NS;

/* ---------------------------------------------------------------------- */

AppIsing::AppIsing(SPPARKS *spk, int narg, char **arg) : 
  AppLattice(spk,narg,arg)
{
  // parse arguments

  if (narg < 2) error->all("Illegal app_style command");

  int seed = atoi(arg[1]);
  random = new RandomPark(seed);

  options(narg-2,&arg[2]);

  // define lattice and partition it across processors

  create_lattice();
  sites = new int[1 + maxneigh];

  // initialize my portion of lattice
  // each site = one of 2 spins
  // loop over global list so assignment is independent of # of procs
  // use map to see if I own global site

  if (infile) read_file();

  else {
    std::map<int,int> hash;
    for (int i = 0; i < nlocal; i++)
      hash.insert(std::pair<int,int> (id[i],i));
    std::map<int,int>::iterator loc;
    
    int isite;
    for (int iglobal = 1; iglobal <= nglobal; iglobal++) {
      isite = random->irandom(2);
      loc = hash.find(iglobal);
      if (loc != hash.end()) lattice[loc->second] = isite;
    }
  }
}

/* ---------------------------------------------------------------------- */

AppIsing::~AppIsing()
{
  delete random;
  delete [] sites;
}

/* ----------------------------------------------------------------------
   compute energy of site
------------------------------------------------------------------------- */

double AppIsing::site_energy(int i)
{
  int isite = lattice[i];
  int eng = 0;
  for (int j = 0; j < numneigh[i]; j++)
    if (isite != lattice[neighbor[i][j]]) eng++;
  return (double) eng;
}

/* ----------------------------------------------------------------------
   perform a site event with null bin rejection
------------------------------------------------------------------------- */

void AppIsing::site_event_rejection(int i, RandomPark *random)
{
  int oldstate = lattice[i];
  double einitial = site_energy(i);

  // event = random spin from 1 to 2, including self

  if (random->uniform() < 0.5) lattice[i] = 1;
  else lattice[i] = 2;
  double efinal = site_energy(i);

  // event = spin flip
  // replace line below: null bin size = 1 - single event propensity

  //if (oldstate == 1) lattice[i] = 2;
  //else lattice[i] = 1;
  //double efinal = site_energy(i);

  // accept or reject via Boltzmann criterion
  // null bin size = 2 - single event propensity

  if (efinal <= einitial) {
  } else if (temperature == 0.0) {
    lattice[i] = oldstate;
  } else if (random->uniform() > exp((einitial-efinal)*t_inverse)) {
    lattice[i] = oldstate;
  }

  // set mask if site could not have changed
  // if site changed, unset mask of sites with affected propensity
  // OK to change mask of ghost sites since never used

  if (Lmask) {
    if (einitial < 0.5*numneigh[i]) mask[i] = 1;
    if (lattice[i] != oldstate)
      for (int j = 0; j < numneigh[i]; j++)
	mask[neighbor[i][j]] = 0;
  }
}

/* ----------------------------------------------------------------------
   compute total propensity of owned site summed over possible events
------------------------------------------------------------------------- */

double AppIsing::site_propensity(int i)
{
  // event = spin flip

  int oldstate = lattice[i];
  int newstate = 1;
  if (oldstate == 1) newstate = 2;

  // compute energy difference between initial and final state
  // if downhill or no energy change, propensity = 1
  // if uphill energy change, propensity = Boltzmann factor

  double einitial = site_energy(i);
  lattice[i] = newstate;
  double efinal = site_energy(i);
  lattice[i] = oldstate;

  if (efinal <= einitial) return 1.0;
  else if (temperature == 0.0) return 0.0;
  else return exp((einitial-efinal)*t_inverse);
}

/* ----------------------------------------------------------------------
   choose and perform an event for site
------------------------------------------------------------------------- */

void AppIsing::site_event(int i, RandomPark *random)
{
  int m;

  // perform event = spin flip

  if (lattice[i] == 1) lattice[i] = 2;
  else lattice[i] = 1;

  // compute propensity changes for self and neighbor sites
  // ignore update of neighbor sites with isite < 0

  int nsites = 0;
  int isite = i2site[i];
  sites[nsites++] = isite;
  propensity[isite] = site_propensity(i);

  for (int j = 0; j < numneigh[i]; j++) {
    m = neighbor[i][j];
    isite = i2site[m];
    if (isite < 0) continue;
    sites[nsites++] = isite;
    propensity[isite] = site_propensity(m);
  }

  solve->update(nsites,sites,propensity);
}
