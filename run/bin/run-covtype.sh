#!/bin/bash
# ----------------------------------------------------------------------------
#  Copyright 2001-2006 The Apache Software Foundation.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ----------------------------------------------------------------------------

#   Copyright (c) 2001-2002 The Apache Software Foundation.  All rights
#   reserved.

#   Copyright (C) 2011-2019 University of Waikato, Hamilton, NZ

BASEDIR=`dirname $0`/..
BASEDIR=`(cd "$BASEDIR"; pwd)`
REPO=$BASEDIR/lib
CLASSPATH=$REPO/*

JCMD=java
if [ -f "$JAVA_HOME/bin/java" ]
then
  JCMD="$JAVA_HOME/bin/java"
fi

# check options
MEMORY=10024m
# MAIN=moa.gui.GUI
MAIN=moa.DoTask
ARGS=
OPTION=
WHITESPACE="[[:space:]]"
for ARG in "$@"
do
  if [ "$ARG" = "-h" ] || [ "$ARG" = "-help" ] || [ "$ARG" = "--help" ]
  then
  	echo "Start script for MOA: Massive Online Analysis"
  	echo ""
  	echo "-h/-help/--help"
  	echo "    prints this help"
  	echo "-memory <memory>"
  	echo "    for supplying maximum heap size, eg 512m or 1g (default: 512m)"
  	echo "-main <classname>"
  	echo "    the class to execute (default: moa.gui.GUI)"
  	echo ""
  	echo "Note: any other options are passed to the Java class as arguments"
  	echo ""
  	exit 0
  fi

  if [ "$ARG" = "-memory" ] || [ "$ARG" = "-main" ]
  then
  	OPTION=$ARG
  	continue
  fi

  if [ "$OPTION" = "-memory" ]
  then
    MEMORY=$ARG
    OPTION=""
    continue
  elif [ "$OPTION" = "-main" ]
  then
    MAIN=$ARG
    OPTION=""
    continue
  fi

  if [[ $ARG =~ $WHITESPACE ]]
  then
    ARGS="$ARGS \"$ARG\""
  else
    ARGS="$ARGS $ARG"
  fi
done

###############################################################################
# 
###############################################################################

DATADIR=/home/hwu344/intellij/phantom-tree/data/covtype
OUTPUTDIR=/home/hwu344/intellij/phantom-tree/run/bin/covtype
min_obs=2000
performance_window=5000
num_phantom_tree=30
conv_rate=0.15

count=0

drift_point=73501
for dataset_prefix in "covtype-0-1-73501" "covtype-0-2-73501" "covtype-0-3-73501" "covtype-0-4-73501" "covtype-0-5-73501" ; do

# covtype-1-2-48334.arff
# covtype-1-3-48334.arff
# covtype-1-4-48334.arff
# covtype-1-5-48334.arff
# 
# covtype-2-3-120836.arff
# covtype-2-4-120836.arff
# covtype-2-5-120836.arff
# 
# covtype-3-4-72502.arff
# covtype-3-5-72502.arff
# covtype-4-5-72501.arff

    #     for min_obs in 500 1000 3000  ; do
    #         for conv_rate in 0.2 0.25 ; do

input=${DATADIR}/${dataset_prefix}.arff

# enable patching
output=${OUTPUTDIR}/enable-patch/rf/${dataset_prefix}/${performance_window}/${num_phantom_tree}/${min_obs}/${conv_rate}
mkdir -p $output
output=${output}/result.csv
> $output

nohup "$JCMD" \
  -classpath "$CLASSPATH" \
  -Xmx$MEMORY \
  -javaagent:"$REPO"/sizeofag-1.0.4.jar \
  $MAIN \
"EvaluatePrequential -l (transfer.TransferFramework -o $performance_window -m $min_obs -l (meta.AdaptiveRandomForest -l (ARFHoeffdingTree -c 0.01 -t 0.1 -l MC) -a 1.0 -x (ADWINChangeDetector -a 0.001) -p (ADWINChangeDetector -a 0.01) -w -u -q) -d (ADWINChangeDetector -a 0.001) -t (PhantomTree -k $num_phantom_tree -n (GaussianNumericAttributeClassObserver -n 2) -g 50 -s (InfoGainSplitCriterion -f 0.1) -c 0.01 -b -l MC) -b $conv_rate  -f $drift_point) -s (ArffFileStream -f $input) -f 1000 -d $output" & 

# disable patching
output=${OUTPUTDIR}/disable-patch/rf/${dataset_prefix}
mkdir -p $output
output=${output}/result.csv
> $output

nohup "$JCMD" \
  -classpath "$CLASSPATH" \
  -Xmx$MEMORY \
  -javaagent:"$REPO"/sizeofag-1.0.4.jar \
  $MAIN \
"EvaluatePrequential -l (transfer.TransferFramework -m $min_obs -l (meta.AdaptiveRandomForest -l (ARFHoeffdingTree -c 0.01 -t 0.1 -l MC) -a 1.0 -x (ADWINChangeDetector -a 0.001) -p (ADWINChangeDetector -a 0.01) -w -u -q) -d (ADWINChangeDetector -a 0.001) -t (PhantomTree -k 5 -n (GaussianNumericAttributeClassObserver -n 2) -g 50 -s (InfoGainSplitCriterion -f 0.1) -c 0.01 -b -l MC) -b 0.15 -x -f $drift_point) -s (ArffFileStream -f $input) -f 1000 -d $output" &


###############################
# ECPF

output=${OUTPUTDIR}/ecpf/rf/${dataset_prefix}
mkdir -p $output
output=${output}/result.csv
> $output

nohup "$JCMD" \
  -classpath "$CLASSPATH" \
  -Xmx$MEMORY \
  -javaagent:"$REPO"/sizeofag-1.0.4.jar \
  $MAIN \
"EvaluatePrequential -l (meta.ECPF -x $drift_point -g 1000 -l (meta.AdaptiveRandomForest -l (ARFHoeffdingTree -c 0.01 -t 0.1 -l MC) -a 1.0 -x (ADWINChangeDetector -a 0.001) -p (ADWINChangeDetector -a 0.01) -w -u -q)) -s (ArffFileStream -f $input) -f 1000 -d $output" &

################################
# aotradaboost

output=${OUTPUTDIR}/aotradaboost/rf/${dataset_prefix}
mkdir -p $output
output=${output}/result.csv
> $output

nohup "$JCMD" \
  -classpath "$CLASSPATH" \
  -Xmx$MEMORY \
  -javaagent:"$REPO"/sizeofag-1.0.4.jar \
  $MAIN \
"EvaluatePrequential -l (InstanceTransfer.TransForest -b 20 -l (AdaptiveRandomForest -l (ARFHoeffdingTree -c 0.01 -t 0.1 -l MC) -a 1.0 -x (ADWINChangeDetector -a 0.001) -p (ADWINChangeDetector -a 0.01) -w -u -q) -x (ADWINChangeDetector -a 0.001) -p (ADWINChangeDetector -a 0.01) -f $drift_point) -s (ArffFileStream -f $input) -f 1000 -d $output" &


################################
# base

output=${OUTPUTDIR}/base/rf/${dataset_prefix}
mkdir -p $output
output=${output}/result.csv
> $output

nohup "$JCMD" \
  -classpath "$CLASSPATH" \
  -Xmx$MEMORY \
  -javaagent:"$REPO"/sizeofag-1.0.4.jar \
  $MAIN \
"EvaluatePrequential -l (meta.AdaptiveRandomForest -l (ARFHoeffdingTree -c 0.01 -t 0.1 -l MC) -a 1.0 -x (ADWINChangeDetector -a 0.001) -p (ADWINChangeDetector -a 0.01) -w -u -q) -s (ArffFileStream -f $input) -f 1000 -d $output" &


count=$(($count+5))
# count=$(($count+1))
if [ $(($count%5)) == 0 -a $count -ge 5 ] ; then
    wait $PIDS
    PIDS=""
fi

done
