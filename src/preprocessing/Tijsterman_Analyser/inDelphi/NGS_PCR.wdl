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
		memory : 15		
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
			
		call Analyzer as AnalyzerAss{
			input:
				inFile = sample[0],
				subject = sample[1],
				left = sample[2],
				right = sample[3],
				leftPrimer = sample[4],
				rightPrimer = sample[5],
				outputFile = sample[6],
				alias = sample[7]
		}
	}
}
