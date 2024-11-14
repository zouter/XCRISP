from Bio import SeqIO
from Bio.Align import PairwiseAligner
from mpi4py import MPI
import numpy as np
import pandas as pd
import os

def get_sequence_with_flank_lengths(sequence, pam_index, left_flank_len, right_flank_len, include_spacer=False):
    l = sequence[pam_index-left_flank_len:pam_index]
    r = sequence[pam_index:pam_index+right_flank_len]
    if include_spacer:
        return l + "|" + r
    else: 
        return l + r

def read_fasta(file_path):
    sequences = []
    # Parse the FASTA file and store sequences in the list
    for record in SeqIO.parse(file_path, "fasta"):
        id = record.id
        description = record.description.split(" ")
        oritentation = description[2]
        if oritentation != "FORWARD":
            continue
        pam_index = int(description[1])
        sequence = str(record.seq)
        left_flank_len = 30
        right_flank_len = 25
        after = get_sequence_with_flank_lengths(sequence, 
                                                pam_index, 
                                                left_flank_len, 
                                                right_flank_len, 
                                                include_spacer=False)
        sequences.append((id, after))  # Convert Seq object to string

        # limit strings for testing
        if len(sequences) > 5:
            break
    return sequences

def smith_waterman_score(seq1, seq2):
    """Calculate the Smith-Waterman alignment score for two sequences."""
    aligner = PairwiseAligner()
    aligner.mode = 'local'
    alignment = aligner.align(seq1, seq2)
    return alignment.score

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Define the file path to your FASTA file
    file_paths =  [f'{os.environ["PROTONDDR"]}/repos/repair-outcome-prediction/local/data/inDelphi/train.fasta',
                   f'{os.environ["PROTONDDR"]}/repos/repair-outcome-prediction/local/data/inDelphi/test.fasta', 
                   f'{os.environ["PROTONDDR"]}/repos/repair-outcome-prediction/local/data/FORECasT/train.fasta',
                   f'{os.environ["PROTONDDR"]}/repos/repair-outcome-prediction/local/data/FORECasT/test.fasta']
    
    # Master process reads the file and distributes sequences
    if rank == 0:
        sequences = []
        for fp in file_paths:
            sequences += read_fasta(fp)
        n = len(sequences)
        print("Sample sequences:")
        for s in sequences[:5]:
            print(s)
        print("\n")

        # Generate all unique pairs of indices for pairwise comparison
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        num_pairs = len(pairs)
        print("Generating all pairwise combos")
    else:
        sequences = None
        pairs = None
        num_pairs = None

    # Broadcast the number of pairs to all processes
    num_pairs = comm.bcast(num_pairs if rank == 0 else None, root=0)
    
    # Scatter the sequences and pairs across all processes
    if rank == 0:
        # Divide the pairs evenly across processes
        chunk_sizes = [num_pairs // size + (1 if i < num_pairs % size else 0) for i in range(size)]
        pair_chunks = []
        start = 0
        for chunk_size in chunk_sizes:
            pair_chunks.append(pairs[start:start + chunk_size])
            start += chunk_size
    else:
        pair_chunks = None

    # Scatter the pair chunks and broadcast sequences to each process
    pair_chunk = comm.scatter(pair_chunks, root=0)
    sequences = comm.bcast(sequences, root=0)

    # Each process calculates the Smith-Waterman score for its chunk of pairs
    local_scores = {}
    for i, j in pair_chunk:
        score = smith_waterman_score(sequences[i][1], sequences[j][1])
        id1 = sequences[i][0]
        id2 = sequences[j][0]
        local_scores[(id1, id2)] = score
    print(f"Rank {rank} working on {len(pair_chunk)} pairs")

    # Gather all scores at the root process
    all_scores = comm.gather(local_scores, root=0)

    # Master process compiles and prints the results
    if rank == 0:
        # Flatten the list of lists into a single list of scores
        # all_scores = [score for sublist in all_scores for score in sublist]
        # for (i, j), score in all_scores:
        #     print(f"Score between sequence {i} and sequence {j}: {score}")
        data = {}
        for d in all_scores:
            data.update(d)

        # Step 1: Convert to DataFrame, with tuples split into two columns
        df = pd.DataFrame(list(data.items()), columns=['Tuple', 'Value'])
        df[['Target1', 'Target2']] = pd.DataFrame(df['Tuple'].tolist(), index=df.index)
        df = df.drop(columns='Tuple')

        # Step 2: Pivot the DataFrame to get the desired format
        result_df = df.pivot(index='Target1', columns='Target2', values='Value')

        # Step 1: Count non-NaN values in each row
        result_df['NonNaNCount'] = result_df.isna().sum(axis=1)
        sorted_df = result_df.sort_values('NonNaNCount', ascending=False).drop(columns='NonNaNCount')
        sorted_df = sorted_df.T
        sorted_df['NonNaNCount'] = sorted_df.isna().sum(axis=1)
        sorted_df = sorted_df.sort_values('NonNaNCount', ascending=False).drop(columns='NonNaNCount')

        sorted_df.to_csv(f"{os.environ['PROTONDDR']}/repos/x-crisp/data/processed/alignment.tsv", sep="\t")

if __name__ == "__main__":
    main()