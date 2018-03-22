#!/bin/bash

let kfold=$1-1
let kfold1=$1

start=$SECONDS

for ifold in $(seq 0 $kfold); do
    python l1m5_fm.py $ifold $kfold1 &
done

let duration=$(( SECONDS - start ))
if (( $duration > 3600 )) ; then
    let "hours=SECONDS/3600"
    let "minutes=(SECONDS%3600)/60"
    let "seconds=(SECONDS%3600)%60"
    echo "Completed in $hours hour(s), $minutes minute(s) and $seconds second(s)" 
elif (( $duration > 60 )) ; then
    let "minutes=(SECONDS%3600)/60"
    let "seconds=(SECONDS%3600)%60"
    echo "Completed in $minutes minute(s) and $seconds second(s)"
else
    echo "Completed in $duration seconds"
fi

