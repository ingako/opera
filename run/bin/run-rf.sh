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

# DATADIR=/home/hwu344/intellij/phantom-tree/data/mnist-max-pooling
# OUTPUTDIR=/home/hwu344/intellij/phantom-tree/run/bin/mnist-max-pooling

DATADIR=/home/hwu344/intellij/phantom-tree/data/fashion-mnist-max-pooling
OUTPUTDIR=/home/hwu344/intellij/phantom-tree/run/bin/fashion-mnist-max-pooling


for dataset in flip01 flip01234 flip01234567 ; do
for seed in {0..9} ; do
# input=${DATADIR}/${dataset}.arff

# # enable patching
# output=${OUTPUTDIR}/enable-patch/rf
# mkdir -p $output
# output=${output}/${dataset}.csv
# > $output

# nohup "$JCMD" \
#   -classpath "$CLASSPATH" \
#   -Xmx$MEMORY \
#   -javaagent:"$REPO"/sizeofag-1.0.4.jar \
#   $MAIN \
# "EvaluatePrequential -l (transfer.TransferFramework -l (meta.AdaptiveRandomForest -l (ARFHoeffdingTree -c 0.01 -t 0.1 -l MC) -a 1.0 -x (ADWINChangeDetector -a 0.001) -p (ADWINChangeDetector -a 0.01) -w -u -q) -d (ADWINChangeDetector -a 0.001) -t (PhantomTree -k 5 -n (GaussianNumericAttributeClassObserver -n 2) -g 50 -s (InfoGainSplitCriterion -f 0.1) -c 0.01 -b -l MC) -b 0.15 -f 70000) -s (ArffFileStream -f $input) -f 1000 -d $output" &
# 
# 
# 
# # disable patching
# output=${OUTPUTDIR}/disable-patch/rf
# mkdir -p $output
# output=${output}/${dataset}.csv
# > $output
# 
# nohup "$JCMD" \
#   -classpath "$CLASSPATH" \
#   -Xmx$MEMORY \
#   -javaagent:"$REPO"/sizeofag-1.0.4.jar \
#   $MAIN \
# "EvaluatePrequential -l (transfer.TransferFramework -l (meta.AdaptiveRandomForest -l (ARFHoeffdingTree -c 0.01 -t 0.1 -l MC) -a 1.0 -x (ADWINChangeDetector -a 0.001) -p (ADWINChangeDetector -a 0.01) -w -u -q) -d (ADWINChangeDetector -a 0.001) -t (PhantomTree -k 5 -n (GaussianNumericAttributeClassObserver -n 2) -g 50 -s (InfoGainSplitCriterion -f 0.1) -c 0.01 -b -l MC) -b 0.15 -x -f 70000) -s (ArffFileStream -f $input) -f 1000 -d $output" &


input=${DATADIR}/${dataset}/${seed}.arff

# transfer only
output=${OUTPUTDIR}/transfer-only/rf/${dataset}/
mkdir -p $output
output=${output}/${seed}.csv
> $output

nohup "$JCMD" \
  -classpath "$CLASSPATH" \
  -Xmx$MEMORY \
  -javaagent:"$REPO"/sizeofag-1.0.4.jar \
  $MAIN \
"EvaluatePrequential -l (transfer.TransferFramework -l (meta.AdaptiveRandomForest -l (ARFHoeffdingTree -c 0.01 -t 0.1 -l MC) -a 1.0 -x (ADWINChangeDetector -a 0.001) -p (ADWINChangeDetector -a 0.01) -w -u -q) -d (ADWINChangeDetector -a 0.001) -t (PhantomTree -k 5 -n (GaussianNumericAttributeClassObserver -n 2) -g 50 -s (InfoGainSplitCriterion -f 0.1) -c 0.01 -b -l MC) -b 0.15 -e -f 70000) -s (ArffFileStream -f $input) -f 1000 -d $output" &

count=$(($count+1))
if [ $(($count%16)) == 0 -a $count -ge 16 ] ; then
    wait $PIDS
    PIDS=""
fi



done
done
