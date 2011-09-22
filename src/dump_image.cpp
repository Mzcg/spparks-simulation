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

/* ----------------------------------------------------------------------
   Contributing author: Nathan Fabian (Sandia)
------------------------------------------------------------------------- */

#include "mpi.h"
#include "math.h"
#include "ctype.h"
#include "stdlib.h"
#include "string.h"
#include "dump_image.h"
#include "math_extra.h"
#include "app.h"
#include "app_lattice.h"
#include "domain.h"
#include "lattice.h"
#include "input.h"
#include "variable.h"
#include "random_park.h"
#include "random_mars.h"
#include "error.h"
#include "memory.h"

#ifdef SPPARKS_JPEG
#include "jpeglib.h"
#endif

using namespace SPPARKS_NS;

#define NCOLORS 140
#define BIG 1.0e20

enum{PPM,JPG};
enum{SPHERE,CUBE};
enum{NUMERIC,IATTRIBUTE,DATTRIBUTE,MINVALUE,MAXVALUE};
enum{STATIC,DYNAMIC};
enum{CONTINUOUS,DISCRETE,SEQUENTIAL};
enum{ABSOLUTE,FRACTIONAL};
enum{NO,YES};
enum{INT,DOUBLE,BIGINT};           // also in dump_text

#define MIN(A,B) ((A) < (B)) ? (A) : (B)
#define MAX(A,B) ((A) > (B)) ? (A) : (B)

/* ---------------------------------------------------------------------- */

DumpImage::DumpImage(SPPARKS *spk, int narg, char **arg) : 
  DumpText(spk, narg, arg)
{
  if (binary || multiproc) error->all("Invalid dump image filename");

  PI = 4.0*atan(1.0);

  // set filetype based on filename suffix

  int n = strlen(filename);
  if (strlen(filename) > 4 && strcmp(&filename[n-4],".jpg") == 0)
    filetype = JPG;
  else if (strlen(filename) > 5 && strcmp(&filename[n-5],".jpeg") == 0)
    filetype = JPG;
  else filetype = PPM;

#ifndef SPPARKS_JPEG
  if (filetype == JPG) error->all("Cannot dump JPG file");
#endif

  // site color,diameter settings

  if (size_one != 2) error->all("Illegal dump image command");

  if (vtype[0] == INT) scolor = IATTRIBUTE;
  else scolor = DATTRIBUTE;

  if (vtype[1] == INT) sdiam = IATTRIBUTE;
  else sdiam = DATTRIBUTE;

  // set defaults for optional args

  shape = SPHERE;
  boundflag = NO;
  crange = drange = NO;
  width = height = 512;
  theta = 60.0 * PI/180.0;
  phi = 30.0 * PI/180.0;
  thetastr = phistr = NULL;
  cflag = STATIC;
  cx = cy = cz = 0.5;
  cxstr = cystr = czstr = NULL;
  if (domain->dimension == 3) {
    up[0] = 0.0; up[1] = 0.0; up[2] = 1.0;
  } else if (domain->dimension == 2) {
    up[0] = 0.0; up[1] = 1.0; up[2] = 0.0;
  } else if (domain->dimension == 1) {
    up[0] = 1.0; up[1] = 0.0; up[2] = 0.0;
  }
  upxstr = upystr = upzstr = NULL;
  zoom = 1.0;
  zoomstr = NULL;
  persp = 0.0;
  perspstr = NULL;
  boxflag = YES;
  boxdiam = 0.02;
  axesflag = NO;
  shiny = 1.0;
  ssao = NO;

  // parse optional args

  int iarg = ioptional;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"shape") == 0) {
      if (iarg+2 > narg) error->all("Illegal dump image command");
      if (strcmp(arg[iarg+1],"sphere") == 0) shape = SPHERE;
      else if (strcmp(arg[iarg+1],"cube") == 0) shape = CUBE;
      iarg += 2;

    } else if (strcmp(arg[iarg],"boundary") == 0) {
      if (iarg+3 > narg) error->all("Illegal dump image command");
      if (strcmp(arg[iarg+1],"yes") == 0) boundflag = YES;
      else if (strcmp(arg[iarg+1],"no") == 0) boundflag = NO;
      else error->all("Illegal dump image command");
      bounddiam = atof(arg[iarg+2]);
      if (bounddiam < 0.0) error->all("Illegal dump image command");
      iarg += 3;

    } else if (strcmp(arg[iarg],"sdiam") == 0) {
      if (iarg+2 > narg) error->all("Illegal dump image command");
      sdiam = NUMERIC;
      sdiamvalue = atof(arg[iarg+1]);
      if (sdiamvalue <= 0.0) error->all("Illegal dump image command");
      iarg += 2;

    } else if (strcmp(arg[iarg],"crange") == 0) {
      if (iarg+3 > narg) error->all("Illegal dump image command");
      crange = YES;
      clo = atoi(arg[iarg+1]);
      chi = atoi(arg[iarg+2]);
      if (clo > chi) error->all("Illegal dump image command");
      iarg += 3;

    } else if (strcmp(arg[iarg],"drange") == 0) {
      if (iarg+3 > narg) error->all("Illegal dump image command");
      drange = YES;
      dlo = atoi(arg[iarg+1]);
      dhi = atoi(arg[iarg+2]);
      if (dlo > dhi) error->all("Illegal dump image command");
      iarg += 3;

    } else if (strcmp(arg[iarg],"size") == 0) {
      if (iarg+3 > narg) error->all("Illegal dump image command");
      width = atoi(arg[iarg+1]);
      height = atoi(arg[iarg+2]);
      if (width <= 0 || height <= 0) error->all("Illegal dump image command");
      iarg += 3;

    } else if (strcmp(arg[iarg],"view") == 0) {
      if (iarg+3 > narg) error->all("Illegal dump image command");
      if (strstr(arg[iarg+1],"v_") == arg[iarg+1]) {
	int n = strlen(&arg[iarg+1][2]) + 1;
	thetastr = new char[n];
	strcpy(thetastr,&arg[iarg+1][2]);
      } else {
	theta = atof(arg[iarg+1]);
	if (theta < 0.0 || theta > 180.0)
	  error->all("Invalid dump image theta value");
	theta *= PI/180.0;
      }
      if (strstr(arg[iarg+1],"v_") == arg[iarg+2]) {
	int n = strlen(&arg[iarg+2][2]) + 1;
	phistr = new char[n];
	strcpy(phistr,&arg[iarg+2][2]);
      } else {
	phi = atof(arg[iarg+2]);
	phi *= PI/180.0;
      }
      iarg += 3;

    } else if (strcmp(arg[iarg],"center") == 0) {
      if (iarg+5 > narg) error->all("Illegal dump image command");
      if (strcmp(arg[iarg+1],"s") == 0) cflag = STATIC;
      else if (strcmp(arg[iarg+1],"d") == 0) cflag = DYNAMIC;
      else error->all("Illegal dump image command");
      if (strstr(arg[iarg+2],"v_") == arg[iarg+2]) {
	int n = strlen(&arg[iarg+2][2]) + 1;
	cxstr = new char[n];
	strcpy(cxstr,&arg[iarg+2][2]);
	cflag = DYNAMIC;
      } else cx = atof(arg[iarg+2]);
      if (strstr(arg[iarg+3],"v_") == arg[iarg+3]) {
	int n = strlen(&arg[iarg+3][2]) + 1;
	cystr = new char[n];
	strcpy(cystr,&arg[iarg+3][2]);
	cflag = DYNAMIC;
      } else cy = atof(arg[iarg+3]);
      if (strstr(arg[iarg+4],"v_") == arg[iarg+4]) {
	int n = strlen(&arg[iarg+4][2]) + 1;
	czstr = new char[n];
	strcpy(czstr,&arg[iarg+4][2]);
	cflag = DYNAMIC;
      } else cz = atof(arg[iarg+4]);
      iarg += 5;

    } else if (strcmp(arg[iarg],"up") == 0) {
      if (iarg+4 > narg) error->all("Illegal dump image command");
      if (strstr(arg[iarg+1],"v_") == arg[iarg+1]) {
	int n = strlen(&arg[iarg+1][2]) + 1;
	upxstr = new char[n];
	strcpy(upxstr,&arg[iarg+1][2]);
      } else up[0] = atof(arg[iarg+1]);
      if (strstr(arg[iarg+2],"v_") == arg[iarg+2]) {
	int n = strlen(&arg[iarg+2][2]) + 1;
	upystr = new char[n];
	strcpy(upystr,&arg[iarg+2][2]);
      } else up[1] = atof(arg[iarg+1]);
      if (strstr(arg[iarg+3],"v_") == arg[iarg+3]) {
	int n = strlen(&arg[iarg+3][2]) + 1;
	upzstr = new char[n];
	strcpy(upzstr,&arg[iarg+3][2]);
      } else up[2] = atof(arg[iarg+3]);
      iarg += 4;

    } else if (strcmp(arg[iarg],"zoom") == 0) {
      if (iarg+2 > narg) error->all("Illegal dump image command");
      if (strstr(arg[iarg+1],"v_") == arg[iarg+1]) {
	int n = strlen(&arg[iarg+1][2]) + 1;
	zoomstr = new char[n];
	strcpy(zoomstr,&arg[iarg+1][2]);
      } else {
	zoom = atof(arg[iarg+1]);
	if (zoom <= 0.0) error->all("Illegal dump image command");
      }
      iarg += 2;

    } else if (strcmp(arg[iarg],"persp") == 0) {
      error->all("Dump image persp option is not yet supported");
      if (iarg+2 > narg) error->all("Illegal dump image command");
      if (strstr(arg[iarg+1],"v_") == arg[iarg+1]) {
	int n = strlen(&arg[iarg+1][2]) + 1;
	perspstr = new char[n];
	strcpy(perspstr,&arg[iarg+1][2]);
      } else {
	persp = atof(arg[iarg+1]);
	if (persp < 0.0) error->all("Illegal dump image command");
      }
      iarg += 2;

    } else if (strcmp(arg[iarg],"box") == 0) {
      if (iarg+3 > narg) error->all("Illegal dump image command");
      if (strcmp(arg[iarg+1],"yes") == 0) boxflag = YES;
      else if (strcmp(arg[iarg+1],"no") == 0) boxflag = NO;
      else error->all("Illegal dump image command");
      boxdiam = atof(arg[iarg+2]);
      if (boxdiam < 0.0) error->all("Illegal dump image command");
      iarg += 3;

    } else if (strcmp(arg[iarg],"axes") == 0) {
      if (iarg+3 > narg) error->all("Illegal dump image command");
      if (strcmp(arg[iarg+1],"yes") == 0) axesflag = YES;
      else if (strcmp(arg[iarg+1],"no") == 0) axesflag = NO;
      else error->all("Illegal dump image command");
      axeslen = atof(arg[iarg+2]);
      axesdiam = atof(arg[iarg+3]);
      if (axeslen < 0.0 || axesdiam < 0.0)
	error->all("Illegal dump image command");
      iarg += 4;

    } else if (strcmp(arg[iarg],"shiny") == 0) {
      if (iarg+2 > narg) error->all("Illegal dump image command");
      shiny = atof(arg[iarg+1]);
      if (shiny < 0.0 || shiny > 1.0)
	error->all("Illegal dump image command");
      iarg += 2;

    } else if (strcmp(arg[iarg],"ssao") == 0) {
      if (iarg+4 > narg) error->all("Illegal dump image command");
      if (strcmp(arg[iarg+1],"yes") == 0) ssao = YES;
      else if (strcmp(arg[iarg+1],"no") == 0) ssao = NO;
      else error->all("Illegal dump image command");
      seed = atoi(arg[iarg+2]);
      if (seed <= 0) error->all("Illegal dump image command");
      ssaoint = atof(arg[iarg+3]);
      if (ssaoint < 0.0 || ssaoint > 1.0)
	error->all("Illegal dump image command");
      iarg += 4;

    } else error->all("Illegal dump image command");
  }

  // error checks

  if (boundflag == YES && shape == SPHERE && me == 0)
    error->warning("Using dump image boundary with spheres");

  if (boundflag == YES) {
    if (app->appclass != App::LATTICE)
      error->all("Dump image boundary requires lattice app");
    applattice = (AppLattice *) app;
  }

  // params based on args

  npixels = width * height;

  // additional defaults for dump_modify options

  ncolors = 0;
  username = NULL;
  userrgb = NULL;

  // color and diameter settings

  diamattribute = NULL;
  colorattribute = NULL;
  color_memflag = NULL;

  if (sdiam == IATTRIBUTE) {
    if (drange == NO) error->all("Dump image drange must be set");
    memory->create(diamattribute,dhi-dlo+1,"image:diamattribute");
    for (int i = dlo; i <= dhi; i++) diamattribute[i-dlo] = 1.0;
  }

  if (scolor == IATTRIBUTE) {
    if (crange == NO) error->all("Dump image crange must be set");
    colorattribute = (double **) memory->smalloc((chi-clo+1)*sizeof(double *),
						 "image:colorattribute");
    memory->create(color_memflag,chi-clo+1,"image:color_memflag");

    for (int i = clo; i <= chi; i++) {
      int j = i-clo;
      int m = j % NCOLORS;
      colorattribute[j] = color2rgb("tmp",m+1);
      color_memflag[j] = 0;
    }
  }

  boundcolor = color2rgb("white",0);
  boxcolor = color2rgb("yellow",0);
  background[0] = background[1] = background[2] = 0;

  mlo = MINVALUE;
  mhi = MAXVALUE;
  mstyle = CONTINUOUS;
  mrange = FRACTIONAL;

  nentry = 2;
  mentry = new MapEntry[nentry];
  mentry[0].svalue = 0.0;
  mentry[0].color = color2rgb("blue",0);
  mentry[1].svalue = 1.0;
  mentry[1].color = color2rgb("red",0);

  // static parameters

  FOV = PI/6.0;              // 30 degrees
  ambientColor[0] = 0.0;
  ambientColor[1] = 0.0;
  ambientColor[2] = 0.0;

  keyLightPhi = -PI/4.0;     // -45 degrees
  keyLightTheta = PI/6.0;    // 30 degrees
  keyLightColor[0] = 0.9;
  keyLightColor[1] = 0.9;
  keyLightColor[2] = 0.9;

  fillLightPhi = PI/6.0;     // 30 degrees
  fillLightTheta = 0; 
  fillLightColor[0] = 0.9;
  fillLightColor[1] = 0.9;
  fillLightColor[2] = 0.9;

  backLightPhi = PI;         // 180 degrees
  backLightTheta = PI/12.0;  // 15 degrees
  backLightColor[0] = 0.9;
  backLightColor[1] = 0.9;
  backLightColor[2] = 0.9;

  // viewflag = DYNAMIC if any view parameter is dynamic

  viewflag = STATIC;
  if (thetastr || phistr || cflag == DYNAMIC || 
      upxstr || upystr || upzstr || zoomstr || perspstr) viewflag = DYNAMIC;

  if (cflag == STATIC) box_center();
  if (viewflag == STATIC) view_params();

  // image and depth buffers

  memory->create(depthBuffer,npixels,"dump:depthBuffer");
  memory->create(surfaceBuffer,2*npixels,"dump:surfaceBuffer");
  memory->create(imageBuffer,3*npixels,"dump:imageBuffer");
  memory->create(depthcopy,npixels,"dump:depthcopy");
  memory->create(surfacecopy,npixels*2,"dump:surfacecopy");
  memory->create(rgbcopy,3*npixels,"dump:rgbcopy");

  // RNG for SSAO depth shading

  random = NULL;
  if (ssao) {
    random = new RandomPark(ranmaster->uniform()); 
    double seed = ranmaster->uniform();
    random->reset(seed,me,100);
  }
}

/* ---------------------------------------------------------------------- */

DumpImage::~DumpImage()
{
  memory->destroy(diamattribute);
  if (colorattribute) {
    for (int i = clo; i <= chi; i++)
      if (color_memflag[i-clo]) delete [] colorattribute[i-clo];
    memory->sfree(colorattribute);
  }
  memory->destroy(color_memflag);

  for (int i = 0; i < ncolors; i++) delete [] username[i];
  memory->sfree(username);
  memory->destroy(userrgb);
  delete [] mentry;

  memory->destroy(depthBuffer);
  memory->destroy(surfaceBuffer);
  memory->destroy(imageBuffer);
  memory->destroy(depthcopy);
  memory->destroy(surfacecopy);
  memory->destroy(rgbcopy);

  delete random; 
}

/* ---------------------------------------------------------------------- */

void DumpImage::init_style()
{
  if (multifile == 0) error->all("Dump image requires one snapshot per file");

  DumpText::init_style();

  // check variables

  if (thetastr) {
    thetavar = input->variable->find(thetastr);
    if (thetavar < 0) 
      error->all("Variable name for dump image theta does not exist");
    if (!input->variable->equalstyle(thetavar))
      error->all("Variable for dump image theta is invalid style");
  }
  if (phistr) {
    phivar = input->variable->find(phistr);
    if (phivar < 0) 
      error->all("Variable name for dump image phi does not exist");
    if (!input->variable->equalstyle(phivar))
      error->all("Variable for dump image phi is invalid style");
  }
  if (cxstr) {
    cxvar = input->variable->find(cxstr);
    if (cxvar < 0) 
      error->all("Variable name for dump image center does not exist");
    if (!input->variable->equalstyle(cxvar))
      error->all("Variable for dump image center is invalid style");
  }
  if (cystr) {
    cyvar = input->variable->find(cystr);
    if (cyvar < 0) 
      error->all("Variable name for dump image center does not exist");
    if (!input->variable->equalstyle(cyvar))
      error->all("Variable for dump image center is invalid style");
  }
  if (czstr) {
    czvar = input->variable->find(czstr);
    if (czvar < 0) 
      error->all("Variable name for dump image center does not exist");
    if (!input->variable->equalstyle(czvar))
      error->all("Variable for dump image center is invalid style");
  }
  if (upxstr) {
    upxvar = input->variable->find(upxstr);
    if (upxvar < 0) 
      error->all("Variable name for dump image center does not exist");
    if (!input->variable->equalstyle(upxvar))
      error->all("Variable for dump image center is invalid style");
  }
  if (upystr) {
    upyvar = input->variable->find(upystr);
    if (upyvar < 0) 
      error->all("Variable name for dump image center does not exist");
    if (!input->variable->equalstyle(upyvar))
      error->all("Variable for dump image center is invalid style");
  }
  if (upzstr) {
    upzvar = input->variable->find(upzstr);
    if (upzvar < 0) 
      error->all("Variable name for dump image center does not exist");
    if (!input->variable->equalstyle(upzvar))
      error->all("Variable for dump image center is invalid style");
  }
  if (zoomstr) {
    zoomvar = input->variable->find(zoomstr);
    if (zoomvar < 0) 
      error->all("Variable name for dump image zoom does not exist");
    if (!input->variable->equalstyle(zoomvar))
      error->all("Variable for dump image zoom is invalid style");
  }
  if (perspstr) {
    perspvar = input->variable->find(perspstr);
    if (perspvar < 0) 
      error->all("Variable name for dump image persp does not exist");
    if (!input->variable->equalstyle(perspvar))
      error->all("Variable for dump image persp is invalid style");
  }
}

/* ---------------------------------------------------------------------- */

void DumpImage::write(double time)
{
  MPI_Request requests[3];
  MPI_Status statuses[3];

  // open new file

  openfile();
  idump++;

  // reset box center and view parameters if dynamic

  if (cflag == DYNAMIC) box_center();
  if (viewflag == DYNAMIC) view_params();

  // nme = # of atoms this proc will contribute to dump
  // pack buf with x,y,z,color,diameter
  // set minmax color range if using color map
  // create my portion of image for my particles
  
  int nme = count();

  if (nme > maxbuf) {
    maxbuf = nme;
    memory->destroy(buf);
    memory->create(buf,maxbuf*size_one,"dump:buf");
  }

  pack();
  if (scolor == DATTRIBUTE) color_minmax();
  create_image();

  // merge images across procs using depth buffer
  // hi procs send to lo procs, cascading down logarithmically

  int nhalf = 1;
  while (nhalf < nprocs) nhalf *= 2;
  nhalf /= 2;

  while (nhalf) {
    if (me < nhalf && me+nhalf < nprocs) {
      MPI_Irecv(rgbcopy,npixels*3,MPI_BYTE,me+nhalf,0,world,&requests[0]);
      MPI_Irecv(surfacecopy,npixels*2,MPI_DOUBLE,me+nhalf,0,world,&requests[1]);
      MPI_Irecv(depthcopy,npixels,MPI_DOUBLE,me+nhalf,0,world,&requests[2]);
      MPI_Waitall(3,requests,statuses);

      for (int i = 0; i < npixels; i++) {
        if (depthBuffer[i] < 0 || (depthcopy[i] >= 0 && 
				   depthcopy[i] < depthBuffer[i])) {
          depthBuffer[i] = depthcopy[i];
          imageBuffer[i*3+0] = rgbcopy[i*3+0];
          imageBuffer[i*3+1] = rgbcopy[i*3+1];
          imageBuffer[i*3+2] = rgbcopy[i*3+2];
          surfaceBuffer[i*2+0] = surfacecopy[i*2+0];
          surfaceBuffer[i*2+1] = surfacecopy[i*2+1];
        }
      }

    } else if (me >= nhalf && me < 2*nhalf) {
      MPI_Send(imageBuffer,npixels*3,MPI_BYTE,me-nhalf,0,world);
      MPI_Send(surfaceBuffer,npixels*2,MPI_DOUBLE,me-nhalf,0,world);
      MPI_Send(depthBuffer,npixels,MPI_DOUBLE,me-nhalf,0,world);
    }

    nhalf /= 2;
  }

  // extra SSAO enhancement
  // bcast full image to all procs
  // each works on subset of pixels
  // gather result back to proc 0

  if (ssao) {
    MPI_Bcast(imageBuffer,npixels*3,MPI_BYTE,0,world);
    MPI_Bcast(surfaceBuffer,npixels*2,MPI_DOUBLE,0,world);
    MPI_Bcast(depthBuffer,npixels,MPI_DOUBLE,0,world);
    compute_SSAO();
    int pixelPart = height/nprocs * width*3;
    MPI_Gather(imageBuffer+me*pixelPart,pixelPart,MPI_BYTE,
               rgbcopy,pixelPart,MPI_BYTE,0,world);
    writeBuffer = rgbcopy;
  } else {
    writeBuffer = imageBuffer;
  }

  // write image file

  if (me == 0) {
    if (filetype == JPG) write_JPG();
    else write_PPM();
    fclose(fp);
  }
}

/* ----------------------------------------------------------------------
   reset view parameters
   called once from constructor if view is STATIC
   called every snapshot from write() if view is DYNAMIC
------------------------------------------------------------------------- */

void DumpImage::box_center()
{
  if (cxstr) phi = input->variable->compute_equal(cxvar);
  if (cystr) phi = input->variable->compute_equal(cyvar);
  if (czstr) phi = input->variable->compute_equal(czvar);

  xctr = boxxlo + cx*(boxxhi-boxxlo);
  yctr = boxylo + cy*(boxyhi-boxylo);
  zctr = boxzlo + cz*(boxzhi-boxzlo);
}

/* ----------------------------------------------------------------------
   reset view parameters
   called once from constructor if view is STATIC
   called every snapshot from write() if view is DYNAMIC
------------------------------------------------------------------------- */

void DumpImage::view_params()
{
  // camDir = camera direction

  if (thetastr) {
    theta = input->variable->compute_equal(thetavar);
    if (theta < 0.0 || theta > 180.0)
      error->all("Invalid dump image theta value");
    theta *= PI/180.0;
  }
  if (phistr) {
    phi = input->variable->compute_equal(phivar);
    phi *= PI/180.0;
  }

  camDir[0] = sin(theta)*cos(phi);
  camDir[1] = sin(theta)*sin(phi);
  camDir[2] = cos(theta);

  // up vector

  if (upxstr) up[0] = input->variable->compute_equal(upxvar);
  if (upystr) up[1] = input->variable->compute_equal(upyvar);
  if (upzstr) up[2] = input->variable->compute_equal(upzvar);

  // zdist = camera distance = function of zoom & bounding box
  // camPos = camera position = function of camDir and zdist

  if (zoomstr) zoom = input->variable->compute_equal(zoomvar);
  if (zoom <= 0.0) error->all("Invalid dump image zoom value");
  if (perspstr) persp = input->variable->compute_equal(perspvar);
  if (persp < 0.0) error->all("Invalid dump image persp value");

  double delx = 2.0*(boxxhi-boxxlo);
  double dely = 2.0*(boxyhi-boxylo);
  double delz = 2.0*(boxzhi-boxzlo);
  double maxdel = MAX(delx,dely);
  maxdel = MAX(maxdel,delz);

  zdist = maxdel;
  zdist /= tan(FOV);
  zdist += 0.5 * (delx*camDir[0] + dely*camDir[1] + delz*camDir[2]);
  zdist /= zoom;

  camPos[0] = camDir[0] * zdist;
  camPos[1] = camDir[1] * zdist;
  camPos[2] = camDir[2] * zdist;

  // camUp = camDir x (Up x camDir)
  // camDir points at the camera, view direction = -camDir

  if (camDir[0] == up[0] && camDir[1] == up[1] && camDir[2] == up[2]) {
    double tmp = up[0];
    up[0] = up[1];
    up[1] = up[2];
    up[2] = tmp;
  }

  MathExtra::cross3(up,camDir,camRight);
  MathExtra::norm3(camRight);
  MathExtra::cross3(camDir,camRight,camUp);
  if (camUp[0] == 0.0 && camUp[1] == 0.0 && camUp[2] == 0.0)
    error->all("Invalid dump image up vector");
  MathExtra::norm3(camUp);

  // light directions in terms of -camDir = z

  keyLightDir[0] = cos(keyLightTheta) * sin(keyLightPhi);
  keyLightDir[1] = sin(keyLightTheta);
  keyLightDir[2] = cos(keyLightTheta) * cos(keyLightPhi);

  fillLightDir[0] = cos(fillLightTheta) * sin(fillLightPhi);
  fillLightDir[1] = sin(fillLightTheta);
  fillLightDir[2] = cos(fillLightTheta) * cos(fillLightPhi);

  backLightDir[0] = cos(backLightTheta) * sin(backLightPhi);
  backLightDir[1] = sin(backLightTheta);
  backLightDir[2] = cos(backLightTheta) * cos(backLightPhi);

  keyHalfDir[0] = 0 + keyLightDir[0];
  keyHalfDir[1] = 0 + keyLightDir[1];
  keyHalfDir[2] = 1 + keyLightDir[2];
  MathExtra::norm3(keyHalfDir);

  // adjust shinyness of the reflection

  specularHardness = 16.0 * shiny;
  specularIntensity = shiny;

  // adjust strength of the SSAO

  if (ssao) {
    SSAORadius = maxdel * 0.05 * ssaoint;
    SSAOSamples = static_cast<int> (8.0 + 32.0*ssaoint);
    SSAOJitter = PI / 12;
    ambientColor[0] = 0.5;
    ambientColor[1] = 0.5;
    ambientColor[2] = 0.5;
  }
  
  // param for rasterizing spheres

  tanPerPixel = -(maxdel / (double) height);
}

/* ----------------------------------------------------------------------
   set explicit values for all min/max settings in color map
   lo/hi current and lvalue/hvalue settings for lo/hi = MIN/MAX VALUE in entries
   if mlo/mhi = MIN/MAX VALUE, compute bounds on just the sites being visualized
------------------------------------------------------------------------- */

void DumpImage::color_minmax()
{
  double two[2],twoall[2];

  if (mlo == MINVALUE || mhi == MAXVALUE) {
    double lo = BIG;
    double hi = -BIG;
    int m = 0;
    for (int i = 0; i < nchoose; i++) {
      lo = MIN(lo,buf[m]);
      hi = MAX(hi,buf[m+1]);
      m += size_one;
    }
    two[0] = -lo;
    two[1] = hi;
    MPI_Allreduce(two,twoall,2,MPI_DOUBLE,MPI_MAX,world);
  }

  if (mlo == MINVALUE) locurrent = -twoall[0];
  else locurrent = mlovalue;
  if (mhi == MAXVALUE) hicurrent = twoall[1];
  else hicurrent = mhivalue;
  if (locurrent > hicurrent) error->all("Invalid dump image color range");

  if (mstyle == CONTINUOUS) {
    if (mrange == ABSOLUTE) mentry[0].svalue = locurrent;
    else mentry[0].svalue = 0.0;
    if (mrange == ABSOLUTE) mentry[nentry-1].svalue = hicurrent;
    else mentry[nentry-1].svalue = 1.0;
  } else if (mstyle == DISCRETE) {
    for (int i = 0; i < nentry; i++) {
      if (mentry[i].lo == MINVALUE) {
	if (mrange == ABSOLUTE) mentry[i].lvalue = locurrent;
	else mentry[i].lvalue = 0.0;
      }
      if (mentry[i].hi == MAXVALUE) {
	if (mrange == ABSOLUTE) mentry[i].hvalue = hicurrent;
	else mentry[i].hvalue = 1.0;
      }
    }
  }
}

/* ----------------------------------------------------------------------
   create image for atoms on this proc
   every pixel has depth 
------------------------------------------------------------------------- */

void DumpImage::create_image()
{
  int i,j,m,ivalue;
  double diameter;
  double *color;

  // initialze image buffers
  // no need to init surfaceBuffer, since will be based on depth

  int red = background[0];
  int green = background[1];
  int blue = background[2];
  int ix,iy;
  for (iy = 0; iy < height; iy ++) {
    for (ix = 0; ix < width; ix ++) {
      imageBuffer[iy * width * 3 + ix * 3 + 0] = red;
      imageBuffer[iy * width * 3 + ix * 3 + 1] = green;
      imageBuffer[iy * width * 3 + ix * 3 + 2] = blue;
      depthBuffer[iy * width + ix] = -1;
    }
  }

  // render my sites

  double **xyz = app->xyz;

  m = 0;
  for (i = 0; i < nchoose; i++) {
    j = clist[i];
    
    if (scolor == IATTRIBUTE) {
      ivalue = static_cast<int> (buf[m]);
      ivalue = MAX(ivalue,clo);
      ivalue = MIN(ivalue,chi);
      color = colorattribute[ivalue-clo];
    } else if (scolor == DATTRIBUTE) {
      color = value2color(buf[m]);
    }

    if (sdiam == NUMERIC) {
      diameter = sdiamvalue;
    } else if (sdiam == IATTRIBUTE) {
      ivalue = static_cast<int> (buf[m+1]);
      ivalue = MAX(ivalue,dlo);
      ivalue = MIN(ivalue,dhi);
      diameter = diamattribute[ivalue-dlo];
    } else if (sdiam == DATTRIBUTE) {
      diameter = buf[m+1];
    }

    if (shape == SPHERE) draw_sphere(xyz[j],color,diameter);
    else draw_cube(xyz[j],color,diameter);

    m += size_one;
  }

  // render my boundaries bewteen adjacent sites
  // loop over all chosen atoms and all their neighbors, whether chosen or not
  // only draw boundary if 2 values are different & 2 sites share adjacent face

  if (boundflag == YES) {
    int k,flag;
    double c1[3],c2[3],c3[4],c4[4];
    int dimension = domain->dimension;
    double dx = domain->lattice->xlattice;
    double dy = domain->lattice->ylattice;
    double dz = domain->lattice->zlattice;
    int *numneigh = applattice->numneigh;
    int **neighbor = applattice->neighbor;
    int *site = app->iarray[0];

    for (int ii = 0; ii < nchoose; ii++) {
      i = clist[ii];
      for (int jj = 0; jj < numneigh[i]; jj++) {
	j = neighbor[i][jj];
	if (site[i] == site[j]) continue;

	flag = 0;
	if (xyz[i][0] != xyz[j][0]) flag++;
	if (xyz[i][1] != xyz[j][1]) flag++;
	if (xyz[i][2] != xyz[j][2]) flag++;
	if (flag >= 2) continue;

	if (xyz[i][0] != xyz[j][0]) {
	  if (fabs(xyz[i][0]-xyz[j][0]) < 0.5*domain->xprd)
	    c1[0] = c2[0] = c3[0] = c4[0] = 0.5*(xyz[i][0]+xyz[j][0]);
	  else 
	    c1[0] = c2[0] = c3[0] = c4[0] = 
	      0.5*(xyz[i][0]+xyz[j][0]-domain->xprd);
	  c1[1] = xyz[i][1] - 0.5*dy; c1[2] = xyz[i][2] - 0.5*dz;
	  c2[1] = xyz[i][1] - 0.5*dy; c2[2] = xyz[i][2] + 0.5*dz;
	  c3[1] = xyz[i][1] + 0.5*dy; c3[2] = xyz[i][2] + 0.5*dz;
	  c4[1] = xyz[i][1] + 0.5*dy; c4[2] = xyz[i][2] - 0.5*dz;
	} else if (xyz[i][1] != xyz[j][1]) {
	  if (fabs(xyz[i][1]-xyz[j][1]) < 0.5*domain->yprd)
	    c1[1] = c2[1] = c3[1] = c4[1] = 0.5*(xyz[i][1]+xyz[j][1]);
	  else 
	    c1[1] = c2[1] = c3[1] = c4[1] = 
	      0.5*(xyz[i][1]+xyz[j][1]-domain->yprd);
	  c1[0] = xyz[i][0] - 0.5*dx; c1[2] = xyz[i][2] - 0.5*dz;
	  c2[0] = xyz[i][0] - 0.5*dx; c2[2] = xyz[i][2] + 0.5*dz;
	  c3[0] = xyz[i][0] + 0.5*dx; c3[2] = xyz[i][2] + 0.5*dz;
	  c4[0] = xyz[i][0] + 0.5*dx; c4[2] = xyz[i][2] - 0.5*dz;
	} else {
	  if (fabs(xyz[i][2]-xyz[j][2]) < 0.5*domain->zprd)
	    c1[2] = c2[2] = c3[2] = c4[2] = 0.5*(xyz[i][2]+xyz[j][2]);
	  else
	    c1[2] = c2[2] = c3[2] = c4[2] = 
	      0.5*(xyz[i][2]+xyz[j][2]-domain->zprd);
	  c1[0] = xyz[i][0] - 0.5*dx; c1[1] = xyz[i][1] - 0.5*dy;
	  c2[0] = xyz[i][0] - 0.5*dx; c2[1] = xyz[i][1] + 0.5*dy;
	  c3[0] = xyz[i][0] + 0.5*dx; c3[1] = xyz[i][1] + 0.5*dy;
	  c4[0] = xyz[i][0] + 0.5*dx; c4[1] = xyz[i][1] - 0.5*dy;
	}
	
	draw_cylinder(c1,c2,boundcolor,bounddiam,3);
	draw_cylinder(c2,c3,boundcolor,bounddiam,3);
	draw_cylinder(c3,c4,boundcolor,bounddiam,3);
	draw_cylinder(c4,c1,boundcolor,bounddiam,3);
      }
    }
  }

  // render outline of simulation box

  if (boxflag) {
    double diameter = MIN(boxxhi-boxxlo,boxyhi-boxylo);
    if (domain->dimension == 3) diameter = MIN(diameter,boxzhi-boxzlo);
    diameter *= boxdiam;

    double (*corners)[3];
    double corner[8][3];
    corner[0][0] = boxxlo; corner[0][1] = boxylo; corner[0][2] = boxzlo;
    corner[1][0] = boxxhi; corner[1][1] = boxylo; corner[1][2] = boxzlo;
    corner[2][0] = boxxlo; corner[2][1] = boxyhi; corner[2][2] = boxzlo;
    corner[3][0] = boxxhi; corner[3][1] = boxyhi; corner[3][2] = boxzlo;
    corner[4][0] = boxxlo; corner[4][1] = boxylo; corner[4][2] = boxzhi;
    corner[5][0] = boxxhi; corner[5][1] = boxylo; corner[5][2] = boxzhi;
    corner[6][0] = boxxlo; corner[6][1] = boxyhi; corner[6][2] = boxzhi;
    corner[7][0] = boxxhi; corner[7][1] = boxyhi; corner[7][2] = boxzhi;
    corners = corner;

    draw_cylinder(corners[0],corners[1],boxcolor,diameter,3);
    draw_cylinder(corners[2],corners[3],boxcolor,diameter,3);
    draw_cylinder(corners[0],corners[2],boxcolor,diameter,3);
    draw_cylinder(corners[1],corners[3],boxcolor,diameter,3);
    draw_cylinder(corners[0],corners[4],boxcolor,diameter,3);
    draw_cylinder(corners[1],corners[5],boxcolor,diameter,3);
    draw_cylinder(corners[2],corners[6],boxcolor,diameter,3);
    draw_cylinder(corners[3],corners[7],boxcolor,diameter,3);
    draw_cylinder(corners[4],corners[5],boxcolor,diameter,3);
    draw_cylinder(corners[6],corners[7],boxcolor,diameter,3);
    draw_cylinder(corners[4],corners[6],boxcolor,diameter,3);
    draw_cylinder(corners[5],corners[7],boxcolor,diameter,3);
  }

  // render XYZ axes in red/green/blue
  // offset by 10% of box size and scale by axeslen

  if (axesflag) {
    double diameter = MIN(boxxhi-boxxlo,boxyhi-boxylo);
    if (domain->dimension == 3) diameter = MIN(diameter,boxzhi-boxzlo);
    diameter *= axesdiam;

    double (*corners)[3];
    double corner[8][3];
    corner[0][0] = boxxlo; corner[0][1] = boxylo; corner[0][2] = boxzlo;
    corner[1][0] = boxxhi; corner[1][1] = boxylo; corner[1][2] = boxzlo;
    corner[2][0] = boxxlo; corner[2][1] = boxyhi; corner[2][2] = boxzlo;
    corner[4][0] = boxxlo; corner[4][1] = boxylo; corner[4][2] = boxzhi;
    corners = corner;

    double offset = MAX(boxxhi-boxxlo,boxyhi-boxylo);
    if (domain->dimension == 3) offset = MAX(offset,boxzhi-boxzlo);
    offset *= 0.1;
    corners[0][0] -= offset; corners[0][1] -= offset; corners[0][2] -= offset;
    corners[1][0] -= offset; corners[1][1] -= offset; corners[1][2] -= offset;
    corners[2][0] -= offset; corners[2][1] -= offset; corners[2][2] -= offset;
    corners[4][0] -= offset; corners[4][1] -= offset; corners[4][2] -= offset;

    corners[1][0] = corners[0][0] + axeslen*(corners[1][0]-corners[0][0]);
    corners[1][1] = corners[0][1] + axeslen*(corners[1][1]-corners[0][1]);
    corners[1][2] = corners[0][2] + axeslen*(corners[1][2]-corners[0][2]);
    corners[2][0] = corners[0][0] + axeslen*(corners[2][0]-corners[0][0]);
    corners[2][1] = corners[0][1] + axeslen*(corners[2][1]-corners[0][1]);
    corners[2][2] = corners[0][2] + axeslen*(corners[2][2]-corners[0][2]);
    corners[4][0] = corners[0][0] + axeslen*(corners[4][0]-corners[0][0]);
    corners[4][1] = corners[0][1] + axeslen*(corners[4][1]-corners[0][1]);
    corners[4][2] = corners[0][2] + axeslen*(corners[4][2]-corners[0][2]);

    draw_cylinder(corners[0],corners[1],color2rgb("red",0),diameter,3);
    draw_cylinder(corners[0],corners[2],color2rgb("green",0),diameter,3);
    draw_cylinder(corners[0],corners[4],color2rgb("blue",0),diameter,3);
  }
}

/* ----------------------------------------------------------------------
   draw sphere at x with surfaceColor and diameter
   render pixel by pixel onto image plane with depth buffering
------------------------------------------------------------------------- */

void DumpImage::draw_sphere(double *x, double *surfaceColor, double diameter)
{
  int ix,iy;
  double projRad;
  double xlocal[3],surface[3];
  double depth;

  xlocal[0] = x[0] - xctr;
  xlocal[1] = x[1] - yctr;
  xlocal[2] = x[2] - zctr;

  double xmap = MathExtra::dot3(camRight,xlocal);
  double ymap = MathExtra::dot3(camUp,xlocal);
  double dist = MathExtra::dot3(camPos,camDir) - MathExtra::dot3(xlocal,camDir);

  double radius = 0.5*diameter;
  double radsq = radius*radius;
  double pixelWidth = (tanPerPixel > 0) ? tanPerPixel * dist : 
    -tanPerPixel / zoom;
  double pixelRadiusFull = radius / pixelWidth;
  int pixelRadius = static_cast<int> (pixelRadiusFull + 0.5) + 1;

  double xf = xmap / pixelWidth;
  double yf = ymap / pixelWidth;
  int xc = static_cast<int> (xf);
  int yc = static_cast<int> (yf);
  double width_error = xf - xc;
  double height_error = yf - yc;

  // shift 0,0 to screen center (vs lower left)

  xc += width / 2;
  yc += height / 2;

  for (iy = yc - pixelRadius; iy <= yc + pixelRadius; iy++) {
    for (ix = xc - pixelRadius; ix <= xc + pixelRadius; ix++) {
      if (iy < 0 || iy >= height || ix < 0 || ix >= width) continue;

      surface[1] = ((iy - yc) - height_error) * pixelWidth;
      surface[0] = ((ix - xc) - width_error) * pixelWidth;
      projRad = surface[0]*surface[0] + surface[1]*surface[1];
      
      // outside the sphere in the projected image
      
      if (projRad > radsq) continue;
      surface[2] = sqrt(radsq - projRad);
      depth = dist - surface[2];

      surface[0] /= radius;
      surface[1] /= radius;
      surface[2] /= radius;

      draw_pixel (ix, iy, depth, surface, surfaceColor);
    }
  }
}

/* ----------------------------------------------------------------------
   draw axis oriented cube at x with surfaceColor and diameter in size
   render pixel by pixel onto image plane with depth buffering
------------------------------------------------------------------------- */

void DumpImage::draw_cube(double *x, double *surfaceColor, double diameter)
{
  double xlocal[3],surface[3],normal[3];
  double t,tdir[3];
  double depth;

  xlocal[0] = x[0] - xctr;
  xlocal[1] = x[1] - yctr;
  xlocal[2] = x[2] - zctr;

  double xmap = MathExtra::dot3(camRight,xlocal);
  double ymap = MathExtra::dot3(camUp,xlocal);
  double dist = MathExtra::dot3(camPos,camDir) - MathExtra::dot3(xlocal,camDir);

  double radius = 0.5*diameter;
  double pixelWidth = (tanPerPixel > 0) ? tanPerPixel * dist : 
    -tanPerPixel / zoom;

  double halfWidth = diameter;
  double pixelHalfWidthFull = halfWidth / pixelWidth;
  int pixelHalfWidth = static_cast<int> (pixelHalfWidthFull + 0.5);

  double xf = xmap / pixelWidth;
  double yf = ymap / pixelWidth;
  int xc = static_cast<int> (xf);
  int yc = static_cast<int> (yf);
  double width_error = xf - xc;
  double height_error = yf - yc;

  // shift 0,0 to screen center (vs lower left)

  xc += width / 2;
  yc += height / 2;

  for (int iy = yc - pixelHalfWidth; iy <= yc + pixelHalfWidth; iy ++) {
    for (int ix = xc - pixelHalfWidth; ix <= xc + pixelHalfWidth; ix ++) {
      if (iy < 0 || iy >= height || ix < 0 || ix >= width) continue;
      
      double sy = ((iy - yc) - height_error) * pixelWidth;
      double sx = ((ix - xc) - width_error) * pixelWidth;
      surface[0] = camRight[0] * sx + camUp[0] * sy;
      surface[1] = camRight[1] * sx + camUp[1] * sy;
      surface[2] = camRight[2] * sx + camUp[2] * sy;

      // iterate through each of the 6 axis-oriented planes of the box
      // only render up to 3 which are facing the camera
      // these checks short circuit a dot product, testing for > 0

      for (int dim = 0; dim < 3; dim ++) {
        if (camDir[dim] > 0) {          // positive faces camera
          t = (radius - surface[dim]) / camDir[dim];
          normal[0] = camRight[dim];
          normal[1] = camUp[dim];
          normal[2] = camDir[dim];
        } else if (camDir[dim] < 0) {   // negative faces camera
          t = -(radius + surface[dim]) / camDir[dim];
          normal[0] = -camRight[dim];
          normal[1] = -camUp[dim];
          normal[2] = -camDir[dim];
        }
        if (camDir[dim] != 0) {
          tdir[0] = camDir[0] * t;
          tdir[1] = camDir[1] * t;
          tdir[2] = camDir[2] * t;

          bool xin = ((surface[0]+tdir[0]) >= -radius) && 
	    ((surface[0]+tdir[0]) <= radius);
          bool yin = ((surface[1]+tdir[1]) >= -radius) && 
	    ((surface[1]+tdir[1]) <= radius);
          bool zin = ((surface[2]+tdir[2]) >= -radius) && 
	    ((surface[2]+tdir[2]) <= radius);

          switch (dim) {
	  case 0:
	    if (yin & zin) {
	      depth = dist - t;
	      draw_pixel (ix, iy, depth, normal, surfaceColor);
	    }
	    break;
	  case 1:
	    if (xin & zin) {
	      depth = dist - t;
	      draw_pixel (ix, iy, depth, normal, surfaceColor);
	    }
	    break;
	  case 2:
	    if (xin & yin) {
	      depth = dist - t;
	      draw_pixel (ix, iy, depth, normal, surfaceColor);
	    }
	    break;
          }
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   draw cylinder from x to y with surfaceColor and diameter
   render pixel by pixel onto image plane with depth buffering
   if sflag = 0, draw no end spheres
   if sflag = 1, draw 1st end sphere
   if sflag = 2, draw 2nd end sphere
   if sflag = 3, draw both end spheres
------------------------------------------------------------------------- */

void DumpImage::draw_cylinder(double *x, double *y,
			      double *surfaceColor, double diameter, int sflag)
{
  double surface[3], normal[3];
  double mid[3],xaxis[3],yaxis[3],zaxis[3];
  double camLDir[3], camLRight[3], camLUp[3];
  double zmin, zmax;

  if (sflag % 2) draw_sphere(x,surfaceColor,diameter);
  if (sflag/2) draw_sphere(y,surfaceColor,diameter);

  double radius = 0.5*diameter;
  double radsq = radius*radius;

  zaxis[0] = y[0] - x[0];
  zaxis[1] = y[1] - x[1];
  zaxis[2] = y[2] - x[2];

  double rasterWidth = fabs(MathExtra::dot3(zaxis, camRight)) + diameter;
  double rasterHeight = fabs(MathExtra::dot3(zaxis, camUp)) + diameter;

  mid[0] = (y[0] + x[0]) * 0.5 - xctr;
  mid[1] = (y[1] + x[1]) * 0.5 - yctr;
  mid[2] = (y[2] + x[2]) * 0.5 - zctr;

  double len = MathExtra::len3(zaxis);
  MathExtra::scale3(1.0/len,zaxis);
  len *= 0.5;
  zmax = len;
  zmin = -len;

  double xmap = MathExtra::dot3(camRight,mid);
  double ymap = MathExtra::dot3(camUp,mid);
  double dist = MathExtra::dot3(camPos,camDir) - MathExtra::dot3(mid,camDir);

  double pixelWidth = (tanPerPixel > 0) ? tanPerPixel * dist : 
    -tanPerPixel / zoom;

  double xf = xmap / pixelWidth;
  double yf = ymap / pixelWidth;
  int xc = static_cast<int> (xf);
  int yc = static_cast<int> (yf);
  double width_error = xf - xc;
  double height_error = yf - yc;

  // shift 0,0 to screen center (vs lower left)

  xc += width / 2;
  yc += height / 2;

  double pixelHalfWidthFull = (rasterWidth * 0.5) / pixelWidth;
  double pixelHalfHeightFull = (rasterHeight * 0.5) / pixelWidth;
  int pixelHalfWidth = static_cast<int> (pixelHalfWidthFull + 0.5);
  int pixelHalfHeight = static_cast<int> (pixelHalfHeightFull + 0.5);

  if (zaxis[0] == camDir[0] && zaxis[1] == camDir[1] && zaxis[2] == camDir[2])
    return;

  MathExtra::cross3(zaxis,camDir,yaxis);
  MathExtra::norm3(yaxis);
  MathExtra::cross3(yaxis,zaxis,xaxis);
  MathExtra::norm3(xaxis);

  camLDir[0] = MathExtra::dot3(camDir,xaxis);
  camLDir[1] = 0.0;
  camLDir[2] = MathExtra::dot3(camDir,zaxis);

  camLRight[0] = MathExtra::dot3(camRight,xaxis);
  camLRight[1] = MathExtra::dot3(camRight,yaxis);
  camLRight[2] = MathExtra::dot3(camRight,zaxis);
  MathExtra::norm3(camLRight);

  camLUp[0] = MathExtra::dot3(camUp,xaxis);
  camLUp[1] = MathExtra::dot3(camUp,yaxis);
  camLUp[2] = MathExtra::dot3(camUp,zaxis);
  MathExtra::norm3(camLUp);

  double a = camLDir[0] * camLDir[0];

  for (int iy = yc - pixelHalfHeight; iy <= yc + pixelHalfHeight; iy ++) {
    for (int ix = xc - pixelHalfWidth; ix <= xc + pixelHalfWidth; ix ++) {
      if (iy < 0 || iy >= height || ix < 0 || ix >= width) continue;
      
      double sy = ((iy - yc) - height_error) * pixelWidth;
      double sx = ((ix - xc) - width_error) * pixelWidth;
      surface[0] = camLRight[0] * sx + camLUp[0] * sy;
      surface[1] = camLRight[1] * sx + camLUp[1] * sy;
      surface[2] = camLRight[2] * sx + camLUp[2] * sy;

      double b = 2 * camLDir[0] * surface[0];
      double c = surface[0] * surface[0] + surface[1] * surface[1] - radsq;

      double partial = b*b - 4*a*c;
      if (partial < 0) continue;
      partial = sqrt (partial);

      double t = (-b + partial) / (2*a);
      double t2 = (-b - partial) / (2*a);
      if (t2 > t) { t = t2; }

      surface[0] += t * camLDir[0];
      surface[1] += t * camLDir[1];
      surface[2] += t * camLDir[2];

      if (surface[2] > zmax || surface[2] < zmin) continue;

      // convert surface into the surface normal

      normal[0] = surface[0] / radius;
      normal[1] = surface[1] / radius;
      normal[2] = 0.0;

      // in camera space 

      surface[0] = MathExtra::dot3 (normal, camLRight);
      surface[1] = MathExtra::dot3 (normal, camLUp);
      surface[2] = MathExtra::dot3 (normal, camLDir);

      double depth = dist - t;
      draw_pixel (ix, iy, depth, surface, surfaceColor);
    }
  }
}

/* ---------------------------------------------------------------------- */

void DumpImage::draw_pixel(int ix, int iy, double depth, 
			   double *surface, double *surfaceColor)
{
  double diffuseKey,diffuseFill,diffuseBack,specularKey;
  if (depth < 0 || (depthBuffer[ix + iy*width] >= 0 && 
		    depth >= depthBuffer[ix + iy*width])) return;
  depthBuffer[ix + iy*width] = depth;
      
  // store only the tangent relative to the camera normal (0,0,-1)

  surfaceBuffer[0 + ix * 2 + iy*width * 2] = surface[1];
  surfaceBuffer[1 + ix * 2 + iy*width * 2] = -surface[0];
      
  diffuseKey = saturate(MathExtra::dot3(surface, keyLightDir));
  diffuseFill = saturate(MathExtra::dot3(surface, fillLightDir));
  diffuseBack = saturate(MathExtra::dot3(surface, backLightDir));
  specularKey = pow(saturate(MathExtra::dot3(surface, keyHalfDir)),
		    specularHardness) * specularIntensity;
  
  double c[3];
  c[0] = surfaceColor[0] * ambientColor[0];
  c[1] = surfaceColor[1] * ambientColor[1];
  c[2] = surfaceColor[2] * ambientColor[2];
      
  c[0] += surfaceColor[0] * keyLightColor[0] * diffuseKey;
  c[1] += surfaceColor[1] * keyLightColor[1] * diffuseKey;
  c[2] += surfaceColor[2] * keyLightColor[2] * diffuseKey;
  
  c[0] += keyLightColor[0] * specularKey;
  c[1] += keyLightColor[1] * specularKey;
  c[2] += keyLightColor[2] * specularKey;
      
  c[0] += surfaceColor[0] * fillLightColor[0] * diffuseFill;
  c[1] += surfaceColor[1] * fillLightColor[1] * diffuseFill;
  c[2] += surfaceColor[2] * fillLightColor[2] * diffuseFill;
      
  c[0] += surfaceColor[0] * backLightColor[0] * diffuseBack;
  c[1] += surfaceColor[1] * backLightColor[1] * diffuseBack;
  c[2] += surfaceColor[2] * backLightColor[2] * diffuseBack;
      
  c[0] = saturate(c[0]);
  c[1] = saturate(c[1]);
  c[2] = saturate(c[2]);
      
  imageBuffer[0 + ix*3 + iy*width*3] = static_cast<int>(c[0] * 255.0);
  imageBuffer[1 + ix*3 + iy*width*3] = static_cast<int>(c[1] * 255.0);
  imageBuffer[2 + ix*3 + iy*width*3] = static_cast<int>(c[2] * 255.0);
}

/* ---------------------------------------------------------------------- */

void DumpImage::compute_SSAO()
{
  // used for rasterizing the spheres

  double delTheta = 2.0*PI / SSAOSamples;

  // typical neighborhood value for shading

  double pixelWidth = (tanPerPixel > 0) ? tanPerPixel : 
	-tanPerPixel / zoom;
  int pixelRadius = (int) trunc (SSAORadius / pixelWidth + 0.5);

  int x,y,s;
  int hPart = height / nprocs;
  int index = me * hPart * width;
  for (y = me * hPart; y < (me + 1) * hPart; y ++) {
    for (x = 0; x < width; x ++, index ++) {
      double cdepth = depthBuffer[index];
      if (cdepth < 0) { continue; }

      double sx = surfaceBuffer[index * 2 + 0];
      double sy = surfaceBuffer[index * 2 + 1];
      double sin_t = -sqrt(sx*sx + sy*sy);

      double theta = random->uniform() * SSAOJitter;
      double ao = 0.0;

      for (s = 0; s < SSAOSamples; s ++) {
        double hx = cos(theta);
        double hy = sin(theta);
        theta += delTheta;

	// multiply by z cross surface tangent
	// so that dot (aka cos) works here

        double scaled_sin_t = sin_t * (hx*sy + hy*sx);

        // Bresenham's line algorithm to march over depthBuffer

        int dx = static_cast<int> (hx * pixelRadius);
        int dy = static_cast<int> (hy * pixelRadius);
        int ex = x + dx;
        if (ex < 0) { ex = 0; } if (ex >= width) { ex = width - 1; }
        int ey = y + dy;
        if (ey < 0) { ey = 0; } if (ey >= height) { ey = height - 1; }
        double delta; 
        int small, large;
        double lenIncr;
        if (fabs(hx) > fabs(hy)) {
          small = (hx > 0) ? 1 : -1;
          large = (hy > 0) ? width : -width;
          delta = fabs(hy / hx);
        } else {
          small = (hy > 0) ? width : -width;
          large = (hx > 0) ? 1 : -1;
          delta = fabs(hx / hy);
        }
        lenIncr = sqrt (1 + delta * delta) * pixelWidth;

        // initialize with one step
	// because the center point doesn't need testing

        int end = ex + ey * width;
        int ind = index + small;
        double len = lenIncr;
        double err = delta;
        if (err >= 1.0) {
          ind += large;
          err -= 1.0;
        }

        double minPeak = -1;
        double peakLen = 0.0;
        int stepsTaken = 1;
        while ((small > 0 && ind <= end) || (small < 0 && ind >= end)) {
          if (ind < 0 || ind >= (width*height)) {
            break;
          }

          // cdepth - depthBuffer B/C we want it in the negative z direction

          if (minPeak < 0 || (depthBuffer[ind] >= 0 && 
			      depthBuffer[ind] < minPeak)) {
            minPeak = depthBuffer[ind];
            peakLen = len;
          }
          ind += small;
          len += lenIncr;
          err += delta;
          if (err >= 1.0) {
            ind += large;
            err -= 1.0;
          }
          stepsTaken ++;
        }

        if (peakLen > 0) {
          double h = atan ((cdepth - minPeak) / peakLen);
          ao += saturate(sin (h) - scaled_sin_t);
        } else {
          ao += saturate(-scaled_sin_t);
        }
      }
      ao /= (double)SSAOSamples;
      
      double c[3];
      c[0] = (double) (*(unsigned char *) &imageBuffer[index * 3 + 0]);
      c[1] = (double) (*(unsigned char *) &imageBuffer[index * 3 + 1]);
      c[2] = (double) (*(unsigned char *) &imageBuffer[index * 3 + 2]);
      c[0] *= (1.0 - ao);
      c[1] *= (1.0 - ao);
      c[2] *= (1.0 - ao);
      imageBuffer[index * 3 + 0] = (int) c[0];
      imageBuffer[index * 3 + 1] = (int) c[1];
      imageBuffer[index * 3 + 2] = (int) c[2];
    }
  }
}

/* ---------------------------------------------------------------------- */

void DumpImage::write_JPG() 
{
#ifdef SPPARKS_JPEG
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  JSAMPROW row_pointer;

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);
  jpeg_stdio_dest(&cinfo, fp);
  cinfo.image_width = width;
  cinfo.image_height = height;
  cinfo.input_components = 3;
  cinfo.in_color_space = JCS_RGB;

  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, 100, 1);
  jpeg_start_compress(&cinfo, 1);

  while (cinfo.next_scanline < cinfo.image_height) {
    row_pointer = (JSAMPROW) 
      &writeBuffer[(cinfo.image_height - 1 - cinfo.next_scanline) * 3 * width];
    jpeg_write_scanlines(&cinfo, &row_pointer, 1);
  }

  jpeg_finish_compress(&cinfo);
  jpeg_destroy_compress(&cinfo);
#endif
}

/* ---------------------------------------------------------------------- */

void DumpImage::write_PPM() 
{
  int x,y;

  fprintf (fp,"P6\n%d %d\n255\n",width,height);
  for (y = height-1; y >= 0; y --)
    for (x = 0; x < width; x ++)
      fprintf (fp,"%c%c%c",
	       writeBuffer[0 + x*3 + y*width*3],
	       writeBuffer[1 + x*3 + y*width*3],
	       writeBuffer[2 + x*3 + y*width*3]);
}

/* ---------------------------------------------------------------------- */

int DumpImage::modify_param(int narg, char **arg)
{
  int n = DumpText::modify_param(narg,arg);
  if (n) return n;
  
  if (strcmp(arg[0],"backcolor") == 0) {
    if (narg < 2) error->all("Illegal dump_modify command");
    double *color = color2rgb(arg[1],0);
    if (color == NULL) error->all("Invalid color in dump_modify command");
    background[0] = static_cast<int> (color[0]*255.0);
    background[1] = static_cast<int> (color[1]*255.0);
    background[2] = static_cast<int> (color[2]*255.0);
    return 2;

  } else if (strcmp(arg[0],"boundcolor") == 0) {
    if (narg < 2) error->all("Illegal dump_modify command");
    boundcolor = color2rgb(arg[1],0);
    if (boundcolor == NULL) error->all("Invalid color in dump_modify command");
    return 2;

  } else if (strcmp(arg[0],"boxcolor") == 0) {
    if (narg < 2) error->all("Illegal dump_modify command");
    boxcolor = color2rgb(arg[1],0);
    if (boxcolor == NULL) error->all("Invalid color in dump_modify command");
    return 2;

  } else if (strcmp(arg[0],"color") == 0) {
    if (narg < 5) error->all("Illegal dump_modify command");
    username = (char **) 
      memory->srealloc(username,(ncolors+1)*sizeof(char *),"dump:username");
    memory->grow(userrgb,ncolors+1,3,"dump:userrgb");
    int n = strlen(arg[1]) + 1;
    username[ncolors] = new char[n];
    strcpy(username[ncolors],arg[1]);
    userrgb[ncolors][0] = atof(arg[2]);
    userrgb[ncolors][1] = atof(arg[3]);
    userrgb[ncolors][2] = atof(arg[4]);
    if (userrgb[ncolors][0] < 0.0 || userrgb[ncolors][0] > 1.0 || 
	userrgb[ncolors][1] < 0.0 || userrgb[ncolors][1] > 1.0 || 
	userrgb[ncolors][2] < 0.0 || userrgb[ncolors][2] > 1.0)
      error->all("Illegal dump_modify command");
    ncolors++;
    return 5;

  } else if (strcmp(arg[0],"scolor") == 0) {
    if (narg < 3) error->all("Illegal dump_modify command");
    if (scolor != IATTRIBUTE)
      error->all("Dump_modify scolor requires integer attribute "
		 "for dump image color");

    int nlo,nhi;
    bounds(arg[1],clo,chi,nlo,nhi);

    // color arg = "random"
    // assign random RGB values to each attribute

    if (strcmp(arg[2],"random") == 0) {
      RandomPark *randomcolor = new RandomPark(ranmaster->uniform()); 
      for (int i = nlo; i <= nhi; i++) {
	double *rgb;
	if (color_memflag[i-clo] == 0) rgb = new double[3];
	else rgb = colorattribute[i-clo];
	rgb[0] = randomcolor->uniform();
	rgb[1] = randomcolor->uniform();
	rgb[2] = randomcolor->uniform();
	colorattribute[i-clo] = rgb;
	color_memflag[i-clo] = 1;
      }
      delete randomcolor;

    // color arg is a color name
    // ptrs = list of ncount colornames separated by '/'
    // assign each of ncount colors in round-robin fashion to attributes

    } else {
      int ncount = 1;
      char *nextptr;
      char *ptr = arg[2];
      while (nextptr = strchr(ptr,'/')) {
	ptr = nextptr + 1;
	ncount++;
      }
      char **ptrs = new char*[ncount+1];
      ncount = 0;
      ptrs[ncount++] = strtok(arg[2],"/");
      while (ptrs[ncount++] = strtok(NULL,"/"));
      ncount--;
      
      int m = 0;
      for (int i = nlo; i <= nhi; i++) {
	if (color_memflag[i-clo] == 1) delete [] colorattribute[i-clo];
	colorattribute[i-clo] = color2rgb(ptrs[m%ncount],0);
	color_memflag[i-clo] = 0;
	if (colorattribute[i-clo] == NULL)
	  error->all("Invalid color in dump_modify command");
	m++;
      }

      delete [] ptrs;
    }

    return 3;

  } else if (strcmp(arg[0],"sdiam") == 0) {
    if (narg < 3) error->all("Illegal dump_modify command");
    if (sdiam != IATTRIBUTE)
      error->all("Dump_modify sdiam requires integer attribute "
		 "for dump image diameter");

    int nlo,nhi;
    bounds(arg[1],dlo,dhi,nlo,nhi);

    double diam = atof(arg[2]);
    if (diam <= 0.0) error->all("Illegal dump_modify command");
    for (int i = nlo; i <= nhi; i++) diamattribute[i-dlo] = diam;
    return 3;

  } else if (strcmp(arg[0],"amap") == 0) {
    if (narg < 6) error->all("Illegal dump_modify command");
    if (!islower(arg[1][0])) {
      mlo = NUMERIC;
      mlovalue = atof(arg[1]);
    } else if (strcmp(arg[1],"min") == 0) mlo = MINVALUE;
    else error->all("Illegal dump_modify command");
    if (!islower(arg[2][0])) {
      mhi = NUMERIC;
      mhivalue = atof(arg[2]);
    } else if (strcmp(arg[2],"max") == 0) mhi = MAXVALUE;
    else error->all("Illegal dump_modify command");
    if (mlo == NUMERIC && mhi == NUMERIC && mlovalue >= mhivalue)
      error->all("Illega dump_modify command");

    if (strlen(arg[3]) != 2) error->all("Illegal dump_modify command");
    if (arg[3][0] == 'c') mstyle = CONTINUOUS;
    else if (arg[3][0] == 'd') mstyle = DISCRETE;
    else if (arg[3][0] == 's') mstyle = SEQUENTIAL;
    else error->all("Illegal dump_modify command");
    if (arg[3][1] == 'a') mrange = ABSOLUTE;
    else if (arg[3][1] == 'f') mrange = FRACTIONAL;
    else error->all("Illegal dump_modify command");
    if (mstyle == SEQUENTIAL) {
      mbinsize = atof(arg[4]);
      if (mbinsize <= 0.0) error->all("Illegal dump_modify command");
    }
    mbinsizeinv = 1.0/mbinsize;

    nentry = atoi(arg[5]);
    mentry = new MapEntry[nentry];
    int n = 6;
    for (int i = 0; i < nentry; i++) {
      if (mstyle == CONTINUOUS) {
	if (n+2 > narg) error->all("Illegal dump_modify command");
	if (!islower(arg[n][0])) {
	  mentry[i].single = NUMERIC;
	  mentry[i].svalue = atof(arg[n]);
	} else if (strcmp(arg[n],"min") == 0) mentry[i].single = MINVALUE;
	else if (strcmp(arg[n],"max") == 0) mentry[i].single = MAXVALUE;
	else error->all("Illegal dump_modify command");
	mentry[i].color = color2rgb(arg[n+1],0);
	n += 2;
      } else if (mstyle == DISCRETE) {
	if (n+3 > narg) error->all("Illegal dump_modify command");
	if (!islower(arg[n][0])) {
	  mentry[i].lo = NUMERIC;
	  mentry[i].lvalue = atof(arg[n]);
	} else if (strcmp(arg[n],"min") == 0) mentry[i].single = MINVALUE;
	else if (strcmp(arg[n],"max") == 0) mentry[i].single = MAXVALUE;
	else error->all("Illegal dump_modify command");
	if (!islower(arg[n+1][0])) {
	  mentry[i].hi = NUMERIC;
	  mentry[i].hvalue = atof(arg[n+1]);
	} else if (strcmp(arg[n+1],"min") == 0) mentry[i].single = MINVALUE;
	else if (strcmp(arg[n+1],"max") == 0) mentry[i].single = MAXVALUE;
	else error->all("Illegal dump_modify command");
	mentry[i].color = color2rgb(arg[n+2],0);
	n += 3;
      } else if (mstyle == SEQUENTIAL) {
	if (n+1 > narg) error->all("Illegal dump_modify command");
	mentry[i].color = color2rgb(arg[n],0);
	n += 1;
      }
      if (mentry[i].color == NULL)
	error->all("Invalid color in dump_modify command");
    }
    
    if (mstyle == CONTINUOUS) {
      if (nentry < 2) error->all("Invalid color map in dump_modify command");
      if (mentry[0].single != MINVALUE || mentry[nentry-1].single != MAXVALUE)
	error->all("Invalid color map in dump_modify command");
      for (int i = 2; i < nentry-1; i++)
	if (mentry[i].svalue <= mentry[i-1].svalue)
	  error->all("Invalid color map in dump_modify command");
    } else if (mstyle == DISCRETE) {
      if (nentry < 1) error->all("Invalid color map in dump_modify command");
      if (mentry[nentry-1].lo != MINVALUE || mentry[nentry-1].hi != MAXVALUE)
	error->all("Invalid color map in dump_modify command");
    } else if (mstyle == SEQUENTIAL) {
      if (nentry < 1) error->all("Invalid color map in dump_modify command");
    }

    return n;
  }
  
  return 0;
}

/* ----------------------------------------------------------------------
   compute bounds implied by numeric str with a possible wildcard asterik
   lo,hi = inclusive bounds
   5 possibilities:
     (1) i = i to i, (2) * = lo to hi,
     (3) i* = i to hi, (4) *j = lo to j, (5) i*j = i to j
   return nlo,nhi
------------------------------------------------------------------------- */

void DumpImage::bounds(char *str, int lo, int hi, int &nlo, int &nhi)
{
  char *ptr = strchr(str,'*');

  if (ptr == NULL) {
    nlo = MAX(atoi(str),lo);
    nhi = MIN(atoi(str),hi);
  } else if (strlen(str) == 1) {
    nlo = lo;
    nhi = hi;
  } else if (ptr == str) {
    nlo = lo;
    nhi = MIN(atoi(ptr+1),hi);
  } else if (strlen(ptr+1) == 0) {
    nlo = MAX(atoi(str),lo);
    nhi = hi;
  } else {
    nlo = MAX(atoi(str),lo);
    nhi = MIN(atoi(ptr+1),hi);
  }
}

/* ----------------------------------------------------------------------
   convert value into an RGB color via color map
------------------------------------------------------------------------- */

double *DumpImage::value2color(double value)
{
  double lo,hi;

  value = MAX(value,locurrent);
  value = MIN(value,hicurrent);

  if (mrange == FRACTIONAL) {
    if (locurrent == hicurrent) value = 0.0;
    else value = (value-locurrent) / (hicurrent-locurrent);
    lo = 0.0;
    hi = 1.0;
  } else {
    lo = locurrent;
    hi = hicurrent;
  }

  if (mstyle == CONTINUOUS) {
    for (int i = 0; i < nentry-1; i++)
      if (value >= mentry[i].svalue && value <= mentry[i+1].svalue) {
	double fraction = (value-mentry[i].svalue) / 
	  (mentry[i+1].svalue-mentry[i].svalue);
	interpolate[0] = mentry[i].color[0] + 
	  fraction*(mentry[i+1].color[0]-mentry[i].color[0]);
	interpolate[1] = mentry[i].color[1] + 
	  fraction*(mentry[i+1].color[1]-mentry[i].color[1]);
	interpolate[2] = mentry[i].color[2] + 
	  fraction*(mentry[i+1].color[2]-mentry[i].color[2]);
	return interpolate;
      }
  } else if (mstyle == DISCRETE) {
    for (int i = 0; i < nentry; i++)
      if (value >= mentry[i].lvalue && value <= mentry[i].hvalue)
	return mentry[i].color;
  } else {
    int ibin = static_cast<int> ((value-lo) * mbinsizeinv);
    return mentry[ibin%nentry].color;
  }

  return NULL;
}

/* ----------------------------------------------------------------------
   search the list of color names for the string color
   return a pointer to the 3 floating point RGB values
   search user-defined color names first, then the list of NCOLORS names
   if index is set, just return pointer to index-1 color
------------------------------------------------------------------------- */

double *DumpImage::color2rgb(char *color, int index)
{
  static char *name[NCOLORS] = { 
    "aliceblue",
    "antiquewhite",
    "aqua",
    "aquamarine",
    "azure",
    "beige",
    "bisque",
    "black",
    "blanchedalmond",
    "blue",
    "blueviolet",
    "brown",
    "burlywood",
    "cadetblue",
    "chartreuse",
    "chocolate",
    "coral",
    "cornflowerblue",
    "cornsilk",
    "crimson",
    "cyan",
    "darkblue",
    "darkcyan",
    "darkgoldenrod",
    "darkgray",
    "darkgreen",
    "darkkhaki",
    "darkmagenta",
    "darkolivegreen",
    "darkorange",
    "darkorchid",
    "darkred",
    "darksalmon",
    "darkseagreen",
    "darkslateblue",
    "darkslategray",
    "darkturquoise",
    "darkviolet",
    "deeppink",
    "deepskyblue",
    "dimgray",
    "dodgerblue",
    "firebrick",
    "floralwhite",
    "forestgreen",
    "fuchsia",
    "gainsboro",
    "ghostwhite",
    "gold",
    "goldenrod",
    "gray",
    "green",
    "greenyellow",
    "honeydew",
    "hotpink",
    "indianred",
    "indigo",
    "ivory",
    "khaki",
    "lavender",
    "lavenderblush",
    "lawngreen",
    "lemonchiffon",
    "lightblue",
    "lightcoral",
    "lightcyan",
    "lightgoldenrodyellow",
    "lightgreen",
    "lightgrey",
    "lightpink",
    "lightsalmon",
    "lightseagreen",
    "lightskyblue",
    "lightslategray",
    "lightsteelblue",
    "lightyellow",
    "lime",
    "limegreen",
    "linen",
    "magenta",
    "maroon",
    "mediumaquamarine",
    "mediumblue",
    "mediumorchid",
    "mediumpurple",
    "mediumseagreen",
    "mediumslateblue",
    "mediumspringgreen",
    "mediumturquoise",
    "mediumvioletred",
    "midnightblue",
    "mintcream",
    "mistyrose",
    "moccasin",
    "navajowhite",
    "navy",
    "oldlace",
    "olive",
    "olivedrab",
    "orange",
    "orangered",
    "orchid",
    "palegoldenrod",
    "palegreen",
    "paleturquoise",
    "palevioletred",
    "papayawhip",
    "peachpuff",
    "peru",
    "pink",
    "plum",
    "powderblue",
    "purple",
    "red",
    "rosybrown",
    "royalblue",
    "saddlebrown",
    "salmon",
    "sandybrown",
    "seagreen",
    "seashell",
    "sienna",
    "silver",
    "skyblue",
    "slateblue",
    "slategray",
    "snow",
    "springgreen",
    "steelblue",
    "tan",
    "teal",
    "thistle",
    "tomato",
    "turquoise",
    "violet",
    "wheat",
    "white",
    "whitesmoke",
    "yellow",
    "yellowgreen"
  };

  static double rgb[NCOLORS][3] = {
    {240/255.0, 248/255.0, 255/255.0},
    {250/255.0, 235/255.0, 215/255.0},
    {0/255.0, 255/255.0, 255/255.0},
    {127/255.0, 255/255.0, 212/255.0},
    {240/255.0, 255/255.0, 255/255.0},
    {245/255.0, 245/255.0, 220/255.0},
    {255/255.0, 228/255.0, 196/255.0},
    {0/255.0, 0/255.0, 0/255.0},
    {255/255.0, 255/255.0, 205/255.0},
    {0/255.0, 0/255.0, 255/255.0},
    {138/255.0, 43/255.0, 226/255.0},
    {165/255.0, 42/255.0, 42/255.0},
    {222/255.0, 184/255.0, 135/255.0},
    {95/255.0, 158/255.0, 160/255.0},
    {127/255.0, 255/255.0, 0/255.0},
    {210/255.0, 105/255.0, 30/255.0},
    {255/255.0, 127/255.0, 80/255.0},
    {100/255.0, 149/255.0, 237/255.0},
    {255/255.0, 248/255.0, 220/255.0},
    {220/255.0, 20/255.0, 60/255.0},
    {0/255.0, 255/255.0, 255/255.0},
    {0/255.0, 0/255.0, 139/255.0},
    {0/255.0, 139/255.0, 139/255.0},
    {184/255.0, 134/255.0, 11/255.0},
    {169/255.0, 169/255.0, 169/255.0},
    {0/255.0, 100/255.0, 0/255.0},
    {189/255.0, 183/255.0, 107/255.0},
    {139/255.0, 0/255.0, 139/255.0},
    {85/255.0, 107/255.0, 47/255.0},
    {255/255.0, 140/255.0, 0/255.0},
    {153/255.0, 50/255.0, 204/255.0},
    {139/255.0, 0/255.0, 0/255.0},
    {233/255.0, 150/255.0, 122/255.0},
    {143/255.0, 188/255.0, 143/255.0},
    {72/255.0, 61/255.0, 139/255.0},
    {47/255.0, 79/255.0, 79/255.0},
    {0/255.0, 206/255.0, 209/255.0},
    {148/255.0, 0/255.0, 211/255.0},
    {255/255.0, 20/255.0, 147/255.0},
    {0/255.0, 191/255.0, 255/255.0},
    {105/255.0, 105/255.0, 105/255.0},
    {30/255.0, 144/255.0, 255/255.0},
    {178/255.0, 34/255.0, 34/255.0},
    {255/255.0, 250/255.0, 240/255.0},
    {34/255.0, 139/255.0, 34/255.0},
    {255/255.0, 0/255.0, 255/255.0},
    {220/255.0, 220/255.0, 220/255.0},
    {248/255.0, 248/255.0, 255/255.0},
    {255/255.0, 215/255.0, 0/255.0},
    {218/255.0, 165/255.0, 32/255.0},
    {128/255.0, 128/255.0, 128/255.0},
    {0/255.0, 128/255.0, 0/255.0},
    {173/255.0, 255/255.0, 47/255.0},
    {240/255.0, 255/255.0, 240/255.0},
    {255/255.0, 105/255.0, 180/255.0},
    {205/255.0, 92/255.0, 92/255.0},
    {75/255.0, 0/255.0, 130/255.0},
    {255/255.0, 240/255.0, 240/255.0},
    {240/255.0, 230/255.0, 140/255.0},
    {230/255.0, 230/255.0, 250/255.0},
    {255/255.0, 240/255.0, 245/255.0},
    {124/255.0, 252/255.0, 0/255.0},
    {255/255.0, 250/255.0, 205/255.0},
    {173/255.0, 216/255.0, 230/255.0},
    {240/255.0, 128/255.0, 128/255.0},
    {224/255.0, 255/255.0, 255/255.0},
    {250/255.0, 250/255.0, 210/255.0},
    {144/255.0, 238/255.0, 144/255.0},
    {211/255.0, 211/255.0, 211/255.0},
    {255/255.0, 182/255.0, 193/255.0},
    {255/255.0, 160/255.0, 122/255.0},
    {32/255.0, 178/255.0, 170/255.0},
    {135/255.0, 206/255.0, 250/255.0},
    {119/255.0, 136/255.0, 153/255.0},
    {176/255.0, 196/255.0, 222/255.0},
    {255/255.0, 255/255.0, 224/255.0},
    {0/255.0, 255/255.0, 0/255.0},
    {50/255.0, 205/255.0, 50/255.0},
    {250/255.0, 240/255.0, 230/255.0},
    {255/255.0, 0/255.0, 255/255.0},
    {128/255.0, 0/255.0, 0/255.0},
    {102/255.0, 205/255.0, 170/255.0},
    {0/255.0, 0/255.0, 205/255.0},
    {186/255.0, 85/255.0, 211/255.0},
    {147/255.0, 112/255.0, 219/255.0},
    {60/255.0, 179/255.0, 113/255.0},
    {123/255.0, 104/255.0, 238/255.0},
    {0/255.0, 250/255.0, 154/255.0},
    {72/255.0, 209/255.0, 204/255.0},
    {199/255.0, 21/255.0, 133/255.0},
    {25/255.0, 25/255.0, 112/255.0},
    {245/255.0, 255/255.0, 250/255.0},
    {255/255.0, 228/255.0, 225/255.0},
    {255/255.0, 228/255.0, 181/255.0},
    {255/255.0, 222/255.0, 173/255.0},
    {0/255.0, 0/255.0, 128/255.0},
    {253/255.0, 245/255.0, 230/255.0},
    {128/255.0, 128/255.0, 0/255.0},
    {107/255.0, 142/255.0, 35/255.0},
    {255/255.0, 165/255.0, 0/255.0},
    {255/255.0, 69/255.0, 0/255.0},
    {218/255.0, 112/255.0, 214/255.0},
    {238/255.0, 232/255.0, 170/255.0},
    {152/255.0, 251/255.0, 152/255.0},
    {175/255.0, 238/255.0, 238/255.0},
    {219/255.0, 112/255.0, 147/255.0},
    {255/255.0, 239/255.0, 213/255.0},
    {255/255.0, 239/255.0, 213/255.0},
    {205/255.0, 133/255.0, 63/255.0},
    {255/255.0, 192/255.0, 203/255.0},
    {221/255.0, 160/255.0, 221/255.0},
    {176/255.0, 224/255.0, 230/255.0},
    {128/255.0, 0/255.0, 128/255.0},
    {255/255.0, 0/255.0, 0/255.0},
    {188/255.0, 143/255.0, 143/255.0},
    {65/255.0, 105/255.0, 225/255.0},
    {139/255.0, 69/255.0, 19/255.0},
    {250/255.0, 128/255.0, 114/255.0},
    {244/255.0, 164/255.0, 96/255.0},
    {46/255.0, 139/255.0, 87/255.0},
    {255/255.0, 245/255.0, 238/255.0},
    {160/255.0, 82/255.0, 45/255.0},
    {192/255.0, 192/255.0, 192/255.0},
    {135/255.0, 206/255.0, 235/255.0},
    {106/255.0, 90/255.0, 205/255.0},
    {112/255.0, 128/255.0, 144/255.0},
    {255/255.0, 250/255.0, 250/255.0},
    {0/255.0, 255/255.0, 127/255.0},
    {70/255.0, 130/255.0, 180/255.0},
    {210/255.0, 180/255.0, 140/255.0},
    {0/255.0, 128/255.0, 128/255.0},
    {216/255.0, 191/255.0, 216/255.0},
    {253/255.0, 99/255.0, 71/255.0},
    {64/255.0, 224/255.0, 208/255.0},
    {238/255.0, 130/255.0, 238/255.0},
    {245/255.0, 222/255.0, 179/255.0},
    {255/255.0, 255/255.0, 255/255.0},
    {245/255.0, 245/255.0, 245/255.0},
    {255/255.0, 255/255.0, 0/255.0},
    {154/255.0, 205/255.0, 50/255.0}
  };

  if (index) return rgb[index-1];

  for (int i = 0; i < ncolors; i++)
    if (strcmp(color,username[i]) == 0) return userrgb[i];
  for (int i = 0; i < NCOLORS; i++)
    if (strcmp(color,name[i]) == 0) return rgb[i];
  return NULL;
}
