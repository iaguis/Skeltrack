#!/bin/bash

DISPLAY=:0 ./test-skeleton | grep 0. > measures.txt
scp measures.txt locke:igalia/skeltrack-measures/measure.txt
ssh locke DISPLAY=:0 ~/igalia/skeltrack-measures/plot-measures.m ~/igalia/skeltrack-measures/ref.txt ~/igalia/skeltrack-measures/measure.txt ~/igalia/skeltrack-measures/measure.pdf
ssh locke DISPLAY=:0 okular ~/igalia/skeltrack-measures/measure.pdf &
