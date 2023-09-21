python3 targets.py

java -jar  -Dconfig.file=sge.conf ~/.local/bin/cromwell-53.jar run NGS_PCR.wdl -i NGS_PCR.wdl.json -o options.json
