task createDummyFastqFiles {
	String inFile
	command {
		python3 /tudelft.net/staff-umbrella/protonddr/repos/repair-outcome-prediction/local/preprocessing/Tijsterman_Analyser/FORECasT/dummy.py ${inFile}
	}
}

task Analyzer {
	String inFile
	String subject
	String left
	String right
	String param = " -m 2 -c -e 0.05 "
	String outputFile
	String leftPrimer
	String rightPrimer
	String alias
	command {
		java -jar ~/.local/bin/Tijsterman_Analyzer_FASTQ_4.3.jar ${param} \
		-infile ${inFile} -subject ${subject} -left ${left} -right ${right} -o ${outputFile} -alias "${alias}  -leftPrimer ${leftPrimer} -rightPrimer ${rightPrimer} "
	}
	runtime {
		cpu : 3
		memory : 20		
		#job_name : "SIQ"
	}
	output {
		File out = "${outputFile}"
		File statOut =  "${outputFile}_stats.txt"
	}
}
workflow NGS_PCR {
	File samples
	Array[Array[String]] sampleArray = read_tsv(samples)

	scatter (sample in sampleArray){
		Int lenSample = length(sample)

		call createDummyFastqFiles {
                        input:
			        inFile = sample[0]
		}
			
		call Analyzer as AnalyzerAss{
			input:
				inFile = sample[1],
				subject = sample[2],
				left = sample[3],
				right = sample[4],
				leftPrimer = sample[5],
				rightPrimer = sample[6],
				outputFile = sample[7],
				alias = sample[8]
		}
	}
}
