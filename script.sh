#!/bin/sh
for (( i=0; i<=255; i++ ))
do
  echo "$i"
  python main.py random minimax -s $i
done