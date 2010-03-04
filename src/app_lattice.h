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

#ifndef SPK_APP_LATTICE_H
#define SPK_APP_LATTICE_H

#include "stdio.h"
#include "app.h"

#include <stack>

namespace SPPARKS_NS {

class AppLattice : public App {
  friend class CommLattice;

 public:
  int nx,ny,nz;                // size of lattice
  int delpropensity;           // # of sites away needed to compute propensity
  int delevent;                // # of sites away affected by an event

  int maxneigh;                // max neighbors of any site in entire system
  int *numneigh;               // # of neighbors of each site
  int **neighbor;              // list of neighbors of each site

  class CommLattice *comm;

  AppLattice(class SPPARKS *, int, char **);
  virtual ~AppLattice();
  void input(char *, int, char **);
  void init();
  void setup();
  void iterate();
  void *extract(char *);

  void grow(int);
  void add_site(int, double, double, double);
  void add_ghost(int, double, double, double, int, int);
  void add_neighbors(int, int, char **);
  void add_values(int, char **);
  void print_connectivity();

  // pure virtual functions, must be defined in child class

  virtual void grow_app() = 0;
  virtual double site_energy(int) = 0;
  virtual void site_event_rejection(int, class RandomPark *) = 0;
  virtual double site_propensity(int) = 0;
  virtual void site_event(int, class RandomPark *) = 0;

  // virtual functions, may be overridden by child class

  void virtual input_app(char *, int, char **);
  void virtual init_app() {}
  void virtual setup_app() {}
  virtual void *extract_app(char *) {return NULL;}

  virtual void push_new_site(int, int *, int, std::stack<int>*);
  virtual void push_connected_neighbors(int, int *, int, std::stack<int>*);
  virtual void connected_ghosts(int, int *, class Cluster *, int);

 protected:
  int me,nprocs;

  int naccept,nattempt;       // number of accepted and attempted events
  int nsweeps;                // number of sweeps performed
  double temperature,t_inverse;  // temperature settings
  double dt_sweep;            // rKMC time for nglobal attemped events
  double dt_rkmc;             // rKMC time for one pass thru all sectors
  double dt_kmc;              // KMC time for one pass thru all sectors

  int allow_kmc;               // 1 if app supports KMC
  int allow_rejection;         // 1 if app supports rejection KMC
  int allow_masking;           // 1 if app supports rKMC masking
  int numrandom;               // # of RN used by rejection routine

  int sweepflag;               // set if rejection KMC solver
  int sectorflag;              // 1 if partition my domain into sectors
  int nsector;                 // 1,2,4,8 = # of sectors
  int nsector_user;            // 0 if default, else 2,4,8
  int ncolors;                 // # of colors, depends on lattice
  int bothflag;                // 1 if both sectors and colors

  class RandomPark *ranapp;    // RN generator for KMC and rejection KMC
  class RandomPark *ranstrict; // RN generator for per-site strict rKMC
  int *siteseeds;              // per-site seeds for ransite
  int *sitelist;               // randomized list of site indices

  bool Lmask;                  // masking on/off
  char *mask;                  // size of nlocal + nghost sites

  double Ladapt;               // adaptive sector time increments for KMC
  double tstop;                // requested time increment in sector
  double nstop;                // requested events per site in sector

  int nmax;                    // max # of sites per-site arrays can store

                               // arrays for owned + ghost sites
  int *owner;                  // proc who owns the site
  int *index;                  // index of site on owning proc

  double *propensity;          // probabilities for each owned site
  int *i2site;                 // mapping of owned lattice to site index

  struct Set {                 // subset of lattice sites I own
    int nlocal;                // # of owned sites in set
    int nselect;               // # of selections from set for rKMC
    int nloop;                 // # of loops over set for rKMC
    int nborder;               // # of sites with non-set site as neighbor
    int *border;               // lattice index for each border site
    int *bsites;               // list of border sites to pass to solver
    class Solve *solve;        // KMC solver
    double *propensity;        // propensities for set sites
    int *site2i;               // map from set sites to lattice index
    int *i2site;               // map from lattice index to set sites
  };
  Set *set;                    // list of subsets
  int nset;                    // # of subsets of lattice sites

  struct Site {
    int id,proc,index;
    double x,y,z;
  };

  void iterate_kmc_global(double);
  void iterate_kmc_sector(double);
  void iterate_rejection(double);

  typedef void (AppLattice::*FnPtrSweep)(int, int *);
  FnPtrSweep sweep;                         // ptr to< sweep functions
  void sweep_nomask_nostrict(int, int *);
  void sweep_mask_nostrict(int, int *);
  void sweep_nomask_strict(int, int *);
  void sweep_mask_strict(int, int *);

  void ghosts_from_connectivity();
  void connectivity_within_cutoff();

  void create_set(int, int, int, class Solve *);
  class Solve *free_set(int);
  int id2color(int);
  int find_border_sites(int);
  void boundary_clear_mask(int);

  void stats(char *);
  void stats_header(char *);

  void set_sector(int, char **);
  void set_sweep(int, char **);
  void set_temperature(int, char **);
};

}

#endif
