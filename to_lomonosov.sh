#!/usr/bin/bash

scp -i ~/.ssh/mgu -r ~/Quantum/diploma/src/* s02200417_2309@lomonosov2.parallel.ru:~/_scratch/diploma/src/.
ssh lomonosov
