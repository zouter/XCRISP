if [ $# -lt 2 ]; then
    echo "1st argument: cromwell-executions folder, 2nd argument: destination directory in HOME"
    exit 1
fi

mkdir -p $OUTPUT_DIR/processed_data/Tijsterman_Analyser/$2/
find ./$1/call-AnalyzerAss/ -name "*_*" -exec cp {} $OUTPUT_DIR/processed_data/Tijsterman_Analyser/$2/ \;
tar -czvf ~/$2.tar.gz $OUTPUT_DIR/processed_data/Tijsterman_Analyser/$2/
echo "Done"
