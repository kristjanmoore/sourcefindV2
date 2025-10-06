#!/bin/bash

copyvectors=BrahuiYorubaSimulation.copyvectors.txt
idfile=BrahuiYorubaSimulation.idfile.txt
paramfile=BrahuiYorubaSimulation.SourcefindParamfile.txt

SOURCEFIND=../sourcefindv2

${SOURCEFIND} \
	--chunklengths ${copyvectors} \
	--parameters ${paramfile} \
	--target BrahuiYorubaSimulation \
	--output BrahuiYorubaSimulation.Cpp.txt \
	--idfile BrahuiYorubaSimulation.idfile.txt
