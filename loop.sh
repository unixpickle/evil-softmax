#!/bin/bash
#
# Run the program repeatedly with the supplied arguments
# and print 0 or 1 for every failure or success.
#
# Example:
#
#     ./loop.sh -maxsteps 200000
#

while true
do
  out=$(go run main.go $@ 2>&1 | tail -n 1 | grep 'correct=true')
  if [ "$out" = "" ]; then
    echo 0
  else
    echo 1
  fi
done
