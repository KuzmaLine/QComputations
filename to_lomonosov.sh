#!/usr/bin/bash

scp -i ~/.ssh/mgu -r ./* s02200417_2309@lomonosov2.parallel.ru:_scratch/diploma/.
ssh lomonosov
