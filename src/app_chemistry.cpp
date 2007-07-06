/* ----------------------------------------------------------------------
   SPPARKS - Stochastic Parallel PARticle Kinetic Simulator
   contact info, copyright info, etc
------------------------------------------------------------------------- */

#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "app_chemistry.h"
#include "solve.h"
#include "finish.h"
#include "timer.h"
#include "memory.h"
#include "error.h"

using namespace SPPARKS;

#define MAX_PRODUCT 5
#define AVOGADRO 6.023e23

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

/* ---------------------------------------------------------------------- */

AppChemistry::AppChemistry(SPK *spk, int narg, char **arg) : App(spk,narg,arg)
{
  if (narg != 1) error->all("Invalid app_style chemistry command");

  // default settings

  ntimestep = 0;
  time = 0.0;
  volume = 0.0;
  stats_delta = 0.0;

  nspecies = 0;
  sname = NULL;

  nreactions = 0;
  rname = NULL;
  nreactant = NULL;
  reactants = NULL;
  nproduct = NULL;
  products = NULL;
  rate = NULL;

  pcount = NULL;
  ndepends = NULL;
  depends = NULL;
  propensity = NULL;
  rcount = NULL;
}

/* ---------------------------------------------------------------------- */

AppChemistry::~AppChemistry()
{
  for (int i = 0; i < nspecies; i++) delete [] sname[i];
  memory->sfree(sname);

  for (int i = 0; i < nreactions; i++) delete [] rname[i];
  memory->sfree(rname);

  memory->sfree(nreactant);
  memory->destroy_2d_int_array(reactants);
  memory->sfree(nproduct);
  memory->destroy_2d_int_array(products);
  memory->sfree(rate);

  memory->sfree(pcount);
  delete [] ndepends;
  memory->destroy_2d_int_array(depends);
  delete [] propensity;
  delete [] rcount;
}

/* ---------------------------------------------------------------------- */

void AppChemistry::init()
{
  if (volume == 0.0) error->all("Invalid volume setting");
  if (nreactions == 0)
    error->all("No reactions defined for chemistry app");

  factor_zero = AVOGADRO * volume;
  factor_dual = 1.0 / (AVOGADRO * volume);

  // determine reaction dependencies

  delete [] ndepends;
  memory->destroy_2d_int_array(depends);
  ndepends = new int[nreactions];
  build_dependency_graph();

  // compute initial propensity for each reaction
  // inform Gillespie solver

  delete [] propensity;
  propensity = new double[nreactions];
  for (int m = 0; m < nreactions; m++) propensity[m] = compute_propensity(m);

  // zero reaction counts

  delete [] rcount;
  rcount = new int[nreactions];
  for (int m = 0; m < nreactions; m++) rcount[m] = 0;

  // print stats header

  if (screen) {
    fprintf(screen,"Step Time");
    for (int m = 0; m < nspecies; m++) fprintf(screen," %s",sname[m]);
    fprintf(screen,"\n");
  }
  if (logfile) {
    fprintf(logfile,"Step Time");
    for (int m = 0; m < nspecies; m++) fprintf(logfile," %s",sname[m]);
    fprintf(logfile,"\n");
  }
  stats();

  // setup future calls to stats()

  stats_time = time + stats_delta;
  if (stats_delta == 0.0) stats_time = stoptime;
}

/* ---------------------------------------------------------------------- */

void AppChemistry::input(char *command, int narg, char **arg)
{
  if (narg == 0) error->all("Invalid command");
  if (strcmp(command,"count") == 0) set_count(narg,arg);
  else if (strcmp(command,"reaction") == 0) add_reaction(narg,arg);
  else if (strcmp(command,"run") == 0) run(narg,arg);
  else if (strcmp(command,"species") == 0) add_species(narg,arg);
  else if (strcmp(command,"stats") == 0) set_stats(narg,arg);
  else if (strcmp(command,"volume") == 0) set_volume(narg,arg);
  else error->all("Invalid command");
}

/* ----------------------------------------------------------------------
   perform a run
------------------------------------------------------------------------- */

void AppChemistry::run(int narg, char **arg)
{
  if (narg != 1) error->all("Illegal run command");
  stoptime = time + atof(arg[0]);

  // error check

  if (solve == NULL) error->all("No solver class defined");

  // init classes used by this app
  
  init();
  solve->init(nreactions,propensity);
  timer->init();

  // perform the run

  iterate();

  // final statistics

  Finish finish(spk);
}

/* ----------------------------------------------------------------------
   iterate on Gillespie solver
------------------------------------------------------------------------- */

void AppChemistry::iterate()
{
  int m,ireaction;
  double dt;

  int done = 0;

  timer->barrier_start(TIME_LOOP);

  while (!done) {
    ntimestep++;

    timer->stamp();
    ireaction = solve->event(&dt);
    timer->stamp(TIME_SOLVE);

    // Check if solver failed to pick an event

    if (ireaction < 0) {

      done = 1;

    } else {

    // update particle counts due to reaction

      rcount[ireaction]++;
      for (m = 0; m < nreactant[ireaction]; m++)
	pcount[reactants[ireaction][m]]--;
      for (m = 0; m < nproduct[ireaction]; m++)
	pcount[products[ireaction][m]]++;

      // update propensities of dependent reactions
      // inform Gillespie solver of changes

      for (m = 0; m < ndepends[ireaction]; m++)
	propensity[depends[ireaction][m]] = 
	  compute_propensity(depends[ireaction][m]);
      solve->update(ndepends[ireaction],depends[ireaction],propensity);

      // update time by Gillepsie dt

      time += dt;
      if (time >= stoptime) done = 1;

    }

    if (time > stats_time || done) {
      stats();
      stats_time += stats_delta;
      timer->stamp(TIME_OUTPUT);
    }
  }

  timer->barrier_stop(TIME_LOOP);
}

/* ----------------------------------------------------------------------
   print stats
------------------------------------------------------------------------- */

void AppChemistry::stats()
{
  if (screen) {
    fprintf(screen,"%d %g",ntimestep,time);
    for (int m = 0; m < nspecies; m++) fprintf(screen," %d",pcount[m]);
    fprintf(screen,"\n");
  }
  if (logfile) {
    fprintf(logfile,"%d %g",ntimestep,time);
    for (int m = 0; m < nspecies; m++) fprintf(logfile," %d",pcount[m]);
    fprintf(logfile,"\n");
  }
}

/* ---------------------------------------------------------------------- */

void AppChemistry::set_count(int narg, char **arg)
{
  if (narg != 2) error->all("Illegal count command");
  
  int ispecies = find_species(arg[0]);
  if (ispecies < 0) {
    char *str = new char[128];
    sprintf(str,"Species ID %s does not exist",arg[0]);
    error->all(str);
  }
  pcount[ispecies] = atoi(arg[1]);
}

/* ---------------------------------------------------------------------- */

void AppChemistry::add_reaction(int narg, char **arg)
{
  if (narg < 3) error->all("Illegal reaction command");

  // store ID

  if (find_reaction(arg[0]) >= 0) {
    char *str = new char[128];
    sprintf(str,"Reaction ID %s already exists",arg[0]);
    error->all(str);
  }

  int n = nreactions + 1;
  rname = (char **) memory->srealloc(rname,n*sizeof(char *),
					  "chemistry:rname");
  int nlen = strlen(arg[0]) + 1;
  rname[nreactions] = new char[nlen];
  strcpy(rname[nreactions],arg[0]);

  // grow reaction arrays

  nreactant = (int *) memory->srealloc(nreactant,n*sizeof(int),
					    "chemistry:nreactnant");
  reactants = memory->grow_2d_int_array(reactants,n,2,
					     "chemistry:reactants");
  nproduct = (int *) memory->srealloc(nproduct,n*sizeof(int),
					   "chemistry:nproduct");
  products = memory->grow_2d_int_array(products,n,MAX_PRODUCT,
					    "chemistry:products");
  rate = (double *) memory->srealloc(rate,n*sizeof(double),
					  "chemistry:rate");

  // find which arg is numeric reaction rate

  char c;
  int iarg = 1;
  while (iarg < narg) {
    c = arg[iarg][0];
    if ((c >= '0' && c <= '9') || c == '+' || c == '-' || c == '.') break;
    iarg++;
  }

  // error checks

  if (iarg == narg) error->all("Reaction has no numeric rate");
  if (iarg < 1 || iarg > 3) 
    error->all("Reaction must have 0,1,2 reactants");
  if (narg-1 - iarg > MAX_PRODUCT) 
    error->all("Reaction cannot have more than MAX_PRODUCT products");

  // extract reactant and product species names
  // if any species does not exist, create it

  nreactant[nreactions] = 0;
  for (int i = 1; i < iarg; i++) {
    int ispecies = find_species(arg[i]);
    if (ispecies == -1) error->all("Unknown species in reaction command");
    reactants[nreactions][i-1] = ispecies;
    nreactant[nreactions]++;
  }

  rate[nreactions] = atof(arg[iarg]);

  nproduct[nreactions] = 0;
  for (int i = iarg+1; i < narg; i++) {
    int ispecies = find_species(arg[i]);
    if (ispecies == -1) error->all("Unknown species in reaction command");
    products[nreactions][i - (iarg+1)] = ispecies;
    nproduct[nreactions]++;
  }
  
  nreactions++;
}

/* ---------------------------------------------------------------------- */

void AppChemistry::add_species(int narg, char **arg)
{
  if (narg == 0) error->all("Illegal species command");

  // grow species arrays

  int n = nspecies + narg;
  sname = (char **) memory->srealloc(sname,n*sizeof(char *),
					  "chemistry:sname");
  pcount = (int *) memory->srealloc(pcount,n*sizeof(int),
					 "chemistry:pcount");

  for (int iarg = 0; iarg < narg; iarg++) {
    if (find_species(arg[iarg]) >= 0) {
      char *str = new char[128];
      sprintf(str,"Species ID %s already exists",arg[iarg]);
      error->all(str);
    }
    int nlen = strlen(arg[iarg]) + 1;
    sname[nspecies+iarg] = new char[nlen];
    strcpy(sname[nspecies+iarg],arg[iarg]);
    pcount[nspecies+iarg] = 0;
  }
  nspecies += narg;
}

/* ---------------------------------------------------------------------- */

void AppChemistry::set_stats(int narg, char **arg)
{
  if (narg != 1) error->all("Illegal stats command");
  stats_delta = atof(arg[0]);
}

/* ---------------------------------------------------------------------- */

void AppChemistry::set_volume(int narg, char **arg)
{
  if (narg != 1) error->all("Illegal volume command");
  volume = atof(arg[0]);
}

/* ----------------------------------------------------------------------
   return reaction index (0 to N-1) for a reaction ID
   return -1 if doesn't exist
------------------------------------------------------------------------- */

int AppChemistry::find_reaction(char *str)
{
  for (int i = 0; i < nreactions; i++)
    if (strcmp(str,rname[i]) == 0) return i;
  return -1;
}

/* ----------------------------------------------------------------------
   return species index (0 to N-1) for a species ID
   return -1 if doesn't exist
------------------------------------------------------------------------- */

int AppChemistry::find_species(char *str)
{
  for (int i = 0; i < nspecies; i++)
    if (strcmp(str,sname[i]) == 0) return i;
  return -1;
}

/* ----------------------------------------------------------------------
   build dependency graph for entire set of reactions
   reaction N depends on M if a reactant of N is a reactant or product of M
------------------------------------------------------------------------- */

void AppChemistry::build_dependency_graph()
{
  int i,j,k,m,n,mspecies,nspecies;

  // count the dependencies in flag array:
  // loop over reactants & products of each reaction
  // for each reaction m, mspecies = its reactants and products
  // for each species, loop over reactants of all reactions n
  // if a match, then set flag[n] since n is in dependency list of m

  int *flag = new int[nreactions];

  for (m = 0; m < nreactions; m++) {
    for (n = 0; n < nreactions; n++) flag[n] = 0;

    for (i = 0; i < nreactant[m]; i++) {
      mspecies = reactants[m][i];
      for (n = 0; n < nreactions; n++) {
	for (j = 0; j < nreactant[n]; j++) {
	  nspecies = reactants[n][j];
	  if (mspecies == nspecies) flag[n] = 1;
	}
      }
    }

    for (i = 0; i < nproduct[m]; i++) {
      mspecies = products[m][i];
      for (n = 0; n < nreactions; n++) {
	for (j = 0; j < nreactant[n]; j++) {
	  nspecies = reactants[n][j];
	  if (mspecies == nspecies) flag[n] = 1;
	}
      }
    }

    ndepends[m] = 0;
    for (n = 0; n < nreactions; n++) if (flag[n]) ndepends[m]++;
  }

  delete [] flag;

  // allocate depends array, 2nd dim is max of ndepends[]

  memory->destroy_2d_int_array(depends);
  int nmax = 0;
  for (m = 0; m < nreactions; m++) nmax = MAX(nmax,ndepends[m]);
  depends = memory->create_2d_int_array(nreactions,nmax,
					     "chemistry:depends");

  // zero the dependencies

  for (m = 0; m < nreactions; m++) ndepends[m] = 0;

  // store the dependencies via same loops as before
  // k loop insures dependency was not already stored

  for (m = 0; m < nreactions; m++) ndepends[m] = 0;

  for (m = 0; m < nreactions; m++) {
    for (i = 0; i < nreactant[m]; i++) {
      mspecies = reactants[m][i];
      for (n = 0; n < nreactions; n++) {
	for (j = 0; j < nreactant[n]; j++) {
	  nspecies = reactants[n][j];
	  if (mspecies == nspecies) {
	    for (k = 0; k < ndepends[m]; k++)
	      if (n == depends[m][k]) break;
	    if (k == ndepends[m]) depends[m][ndepends[m]++] = n;
	  }
	}
      }
    }

    for (i = 0; i < nproduct[m]; i++) {
      mspecies = products[m][i];
      for (n = 0; n < nreactions; n++) {
	for (j = 0; j < nreactant[n]; j++) {
	  nspecies = reactants[n][j];
	  if (mspecies == nspecies) {
	    for (k = 0; k < ndepends[m]; k++)
	      if (n == depends[m][k]) break;
	    if (k == ndepends[m]) depends[m][ndepends[m]++] = n;
	  }
	}
      }
    }
  }
}

/* ----------------------------------------------------------------------
   compute propensity of a single reaction
   for mono reaction: propensity = count * rate
   for dual reaction: propensity = count1 * count2 * rate / (Avogadro*Volume)
   for dual reaction: cut in half if reactants are same species
------------------------------------------------------------------------- */

double AppChemistry::compute_propensity(int m)
{
  double p;
  if (nreactant[m] == 0) p = factor_zero * rate[m];
  else if (nreactant[m] == 1) p = pcount[reactants[m][0]] * rate[m];
  else {
    if (reactants[m][0] == reactants[m][1]) 
      p = 0.5 * factor_dual * pcount[reactants[m][0]] * 
	(pcount[reactants[m][1]] - 1) * rate[m];
    else
      p = factor_dual * pcount[reactants[m][0]] * 
	pcount[reactants[m][1]] * rate[m];
  }
  return p;
}