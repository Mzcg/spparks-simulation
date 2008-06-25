/* ----------------------------------------------------------------------
   SPPARKS - Stochastic Parallel PARticle Kinetic Simulator
   contact info, copyright info, etc
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

AppIsing::AppIsing(SPPARKS *spk, int narg, char **arg) : AppLattice(spk,narg,arg)
{
  // parse arguments

  if (narg < 2) error->all("Invalid app_style ising command");

  seed = atoi(arg[1]);
  random = new RandomPark(seed);

  options(narg-2,&arg[2]);

  // define lattice and partition it across processors

  create_lattice();
  sites = new int[1 + maxneigh];

  // initialize my portion of lattice
  // each site = one of 2 spins
  // loop over global list so assignment is independent of # of procs
  // use map to see if I own global site

  std::map<int,int> hash;
  for (int i = 0; i < nlocal; i++)
    hash.insert(std::pair<int,int> (id[i],i));
  std::map<int,int>::iterator loc;

  int ilocal,isite;
  for (int iglobal = 1; iglobal <= nglobal; iglobal++) {
    isite = random->irandom(2);
    loc = hash.find(iglobal);
    if (loc != hash.end()) lattice[loc->second] = isite;
  }
}

/* ---------------------------------------------------------------------- */

AppIsing::~AppIsing()
{
  delete random;
  delete [] sites;
  delete comm;
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
   randomly pick new state for site
------------------------------------------------------------------------- */

void AppIsing::site_pick_random(int i, double ran)
{
  int iran = (int) (2*ran) + 1;
  if (iran > 2) iran = 2;
  lattice[i] = iran;
}

/* ----------------------------------------------------------------------
   randomly pick new state for site from neighbor values
------------------------------------------------------------------------- */

void AppIsing::site_pick_local(int i, double ran)
{
  int iran = (int) (numneigh[i]*ran) + 1;
  if (iran > numneigh[i]) iran = numneigh[i];
  lattice[i] = lattice[neighbor[i][iran]];
}

/* ----------------------------------------------------------------------
   compute total propensity of owned site
   based on einitial,efinal for each possible event
   if no energy change, propensity = 1
   if downhill energy change, propensity = 1
   if uphill energy change, propensity set via Boltzmann factor
   if proc owns full domain, there are no ghosts, so ignore full flag
------------------------------------------------------------------------- */

double AppIsing::site_propensity(int i, int full)
{
  // only event is a spin flip

  int oldstate = lattice[i];
  int newstate = 1;
  if (oldstate == 1) newstate = 2;

  // compute energy difference between initial and final state

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
   update propensities of all affected sites
   if proc owns full domain, there are no ghosts, so ignore full flag
   if proc owns sector, ignore neighbor sites that are ghosts
------------------------------------------------------------------------- */

void AppIsing::site_event(int i, int full)
{
  int j,m,isite;

  // only event is a spin flip

  if (lattice[i] == 1) lattice[i] = 2;
  else lattice[i] = 1;

  // compute propensity changes for self and neighbor sites

  int nsites = 0;
  isite = i2site[i];
  sites[nsites++] = isite;
  propensity[isite] = site_propensity(i,full);

  for (j = 0; j < numneigh[i]; j++) {
    m = neighbor[i][j];
    isite = i2site[m];
    if (isite < 0) continue;
    sites[nsites++] = isite;
    propensity[isite] = site_propensity(m,full);
  }

  solve->update(nsites,sites,propensity);
}

/* ----------------------------------------------------------------------
  clear mask values of site and its neighbors
  OK to clear ghost site mask values
------------------------------------------------------------------------- */

void AppIsing::site_clear_mask(char *mask, int i)
{
  mask[i] = 0;
  for (int j = 0; j < numneigh[i]; j++) mask[neighbor[i][j]] = 0;
}
