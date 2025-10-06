#!/bin/bash

copyvectors=BrahuiYorubaSimulation.copyvectors.txt
idfile=BrahuiYorubaSimulation.idfile.1_id_per_pop.txt
paramfile=BrahuiYorubaSimulation.SourcefindParamfile.1_id_per_pop.txt

SOURCEFIND=../sourcefindv2

${SOURCEFIND} \
	--chunklengths ${copyvectors} \
	--parameters ${paramfile} \
	--output BrahuiYorubaSimulation.1_id_per_pop.txt \
	--idfile BrahuiYorubaSimulation.idfile.txt
	# --target BrahuiYorubaSimulation \
