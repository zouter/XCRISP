# 1. Join paired end reads
python3 join_paired_end_reads.py

# 2. Break up fasta files for easier preprocessing
python3 partition_paired_reads.py

# 3. Convert the targets.tsv to fasta
python3 convert_oligo_tsv_to_fasta.py

# 4. Map reads to targets
python3 map_reads_to_targets.py

# 6. Split NULL mapping files by oligo id
python3 split_null_mappings.py

# 5. Split mapped reads by oligo id
# Need to remove limits on this files
python3 split_mapped_reads_by_id.py
python3 combine_fastq_files.py

# 6. Processes mappings of plasmid library to form expanded set of oligo templates (accounting for synthesis mutations)
python3 compile_mapped_null_profiles.py

# 7. Map reads to expanded target list
python3 map_reads_to_expanded_targets.py
