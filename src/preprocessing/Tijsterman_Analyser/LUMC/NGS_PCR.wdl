task concatFiles {
	Array[File] inFiles
	String outFile
	#Only get the header from the first line and then concatenate the rest
	command {
		head -n 1 ${inFiles[0]} > ${outFile}
		sed -s '1d' ${sep=" " inFiles} >> ${outFile}
	}
	output {
		File out = "${outFile}"
	}
}
task concatArrays {
    Array[File] inA
    Array[File] inB
    Array[File] inC
    command {
        cat ${write_lines(inA)}
        cat ${write_lines(inB)}
        cat ${write_lines(inC)}
    }
    output {
        Array[File] out = read_lines(stdout())
    }
}
task PEAR {
	String r1
	String r2
	String out
	command {
		pear -z -u 0 -f ${r1} -r ${r2} -o ${out} -j 4
	}
	output {
		File assembled = "${out}.assembled.fastq"
		File discarded = "${out}.discarded.fastq"
		File unassF = "${out}.unassembled.forward.fastq"
		File unassR = "${out}.unassembled.reverse.fastq"
	}
	runtime {
		cpu : 4
		#job_name : "PEAR"
	}
}
task FLASH2 {
	String r1
	String r2
	String out
	command {
		flash2 ${r1} ${r2} -M 5000 -O -x 0 -o ${out} -t 4
	}
	output {
		File assembled = "${out}.extendedFrags.fastq"
		File unassF = "${out}.notCombined_1.fastq"
		File unassR = "${out}.notCombined_2.fastq"
	}
	
	runtime {
		cpu : 4
	}
}
task Analyzer {
	String inFile
	String inFileF
	String inFileR
	String subject
	String left
	String right
	String param = " -m 2 -c -e 0.05 "
	String outputFile
	String leftPrimer
	String rightPrimer
	String minPassedPrimer
	String alias
	String? hdr
	command {
		java -jar ~/.local/bin/Tijsterman_Analyzer_FASTQ_4.3.jar ${param} \
		-infile ${inFile} -subject ${subject} -left ${left} -right ${right} -o ${outputFile}_indels -leftPrimer ${leftPrimer} -rightPrimer ${rightPrimer} \
		-minPassedPrimer ${minPassedPrimer} -alias "${alias}" -infileF ${inFileF} -infileR ${inFileR}
	}
	runtime {
		cpu : 3
		memory : 15		
		#job_name : "SIQ"
	}
	output {
		File out = "${outputFile}_indels"
		File statOut =  "${outputFile}_indels_stats.txt"
	}
}
workflow NGS_PCR {
	File samples
	Array[Array[String]] sampleArray = read_tsv(samples)
	String outputFile

	scatter (sample in sampleArray){
		Int lenSample = length(sample)
		if ( lenSample > 11 ) {
			String hdr = sample[11]
		}
			
		#switched to FLASH2
		call FLASH2 {
			input: 
				r1 = sample[0],
				r2 = sample[1],
				out = sample[2]
		}
		call Analyzer as AnalyzerAss{
			input:
				inFile = FLASH2.assembled,
				inFileF = FLASH2.unassF,
				inFileR = FLASH2.unassR,
				subject = sample[3],
				left = sample[4],
				right = sample[5],
				leftPrimer = sample[6],
				rightPrimer = sample[7],
				outputFile = sample[2],
				alias = sample[2],
				minPassedPrimer = sample[8]
		}
	}
	call concatFiles{
		input:
			inFiles = AnalyzerAss.out,
			outFile = outputFile
	}
	call concatFiles as concatStats {
		input:
			inFiles = AnalyzerAss.statOut,
			outFile = "${outputFile}_stats.txt"
	}
    }
