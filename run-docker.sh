#!/bin/bash
docker build -t juryselectionimage Environment/
docker run --init -it -v $(PWD)/:/juryselection -w /juryselection/Code juryselectionimage ./execute_all.sh
