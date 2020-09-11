#!/bin/bash

SPPARKS=$HOME/jaks.git/spparks.cloned/src/spk_tutka.gnu
PATHDATA=stitching.dat
declare -a THICKNESS
declare -a LAYER_PTR
declare -a PATHS
declare -a WINDOWS
declare -a LAYER_WINDOWS
let ZSTART=0
let DEPTH_HAZ=4


function get_thicknesses_paths_and_windows {

local F
local var
local LAYER_BOTTOM
local Z0
let Z1=$ZSTART

let local L=0
let local P=0
let local PTR=0
while read LINE; do

  F=$(echo $LINE | cut -d ' ' -f1)
  #echo $F

  case "$F" in

    layer)
      var=$(echo $LINE | cut -d ' ' -f2)
      if [ "$var" == "thickness" ]; then
	THICKNESS+=($(echo $LINE | cut -d ' ' -f3))
	LAYER_PTR+=($PTR)
        let Z1=$Z1+${THICKNESS[$L]}
        let LAYER_BOTTOM=${Z1}-${THICKNESS[$L]}
      else
	echo "INPUT ERROR: expected 'layer thickness'"
        exit 1
      fi
      if [ $L -ge 1 ]; then
	let Z0=$Z1-$DEPTH_HAZ
      else
	let Z0=$ZSTART
      fi
      let L=$L+1
      ;;
    path)
      # am path 1 start $X0 $Y0 end $X1 $Y1 speed 9
      # am path 1 PATH speed 9
      var=$(echo $LINE | cut -d ' ' -f2)
      if [ "$var" != "start" ]; then
	echo "INPUT ERROR: expected 'start'"
        exit 1
      fi
      var=$(echo $LINE | cut -d ' ' -f5)
      if [ "$var" != "end" ]; then
	echo "INPUT ERROR: expected 'end'"
        exit 1
      fi
      # Save this 'path' to array of PATHS
      PATHS+=("$(echo $LINE | cut -d ' ' -f 2,3,4,5,6,7)")

      var=$(echo $LINE | cut -d ' ' -f8)
      if [ "$var" != "block" ]; then
	echo "INPUT ERROR: expected 'block'"
        exit 1
      fi
      
      LAYER_WINDOWS+=("$(echo $LINE | cut -d ' ' -f 9,10,11,12) ${LAYER_BOTTOM} ${Z1}")
      WINDOWS+=("$(echo $LINE | cut -d ' ' -f 9,10,11,12) ${Z0} ${Z1}")
      #echo "PROCESS Layer # ${L}; Z0=${Z0}; Z1=${Z1}; LAYER_BOTTOM=${LAYER_BOTTOM}"
      #echo "PROCESS Layer # ${L}; LAYER_WINDOW=${LAYER_WINDOWS[$PTR]}"

      # Now move PTR in preparation for next PATH
      let PTR+=1
      ;;
    \#)
         #echo COMMENT: ${LINE}
      ;;
    *)
      echo "Usage: {layer|domain}"
      exit 1
      ;;

  esac

done < $1

# Add one extra layer pointer to start of non existent next layer
LAYER_PTR+=($PTR)

}

get_thicknesses_paths_and_windows $PATHDATA
NUM_LAYERS=${#THICKNESS[@]}

#echo "Number of layers = ${NUM_LAYERS}"
#echo "Layer thicknesses = ${THICKNESS[@]}"
#echo "Layer PTR = ${LAYER_PTR[@]}"


let local PTR=0
for (( L=0; L<$NUM_LAYERS; L++ )); do
  
  # Number of paths on layer
  let LAYNUM=$L+1
  let NUM_PATHS=${LAYER_PTR[$L+1]}-${LAYER_PTR[$L]}
  echo "Layer # ${LAYNUM}; Number of paths on layer = ${NUM_PATHS}"
  # Initialize layer
  for (( P=${LAYER_PTR[$L]}; P<${LAYER_PTR[$L+1]}; P++ )); do
    WINDOW=${WINDOWS[$P]}
    echo "LAYER_WINDOW=${LAYER_WINDOWS[$P]}"
    echo "WINDOW=${WINDOWS[$P]}"
    echo "PATH=${PATHS[$P]}"
    #####################################
    # Run potts model to initialize layer
    cat in.init | sed s"/WINDOW/${WINDOW}/" > in.potts_init
    
    #  
    # Run SPPARKS to initialize microstructure on layer
    SEED=$RANDOM
    $SPPARKS -var SEED $SEED < in.potts_init
    #####################################
  done
  # Run AM model on layer
  for (( P=${LAYER_PTR[$L]}; P<${LAYER_PTR[$L+1]}; P++ )); do
    WINDOW=${WINDOWS[$P]}
    echo "LAYER_WINDOW=${LAYER_WINDOWS[$P]}"
    echo "WINDOW=${WINDOWS[$P]}"
    echo "PATH=${PATHS[$P]}"
    #####################################
    # Run potts model to initialize layer
    #cat in.init | sed s"/WINDOW/${WINDOW}/" > in.potts_init
    
    #  
    # Run SPPARKS to initialize microstructure on layer
    SEED=$RANDOM
    #$SPPARKS -var SEED $SEED < in.potts_init
    #####################################
  done

done
