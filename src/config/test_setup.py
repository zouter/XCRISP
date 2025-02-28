from src.data.data_loader import get_details_from_fasta

MIN_NUMBER_OF_READS = 100
assert(MIN_NUMBER_OF_READS in [100, 1000])

TEST_FILES = [
    ("FORECasT", "test", "test"),
    ("FORECasT", "test", "TREX_A_test"),
    ("FORECasT", "test", "HAP1_test"),
    ("inDelphi", "LibA", "0105-mESC-Lib1-Cas9-Tol2-BioRep2-techrep1"),
    ("inDelphi", "LibA", "052218-U2OS-+-LibA-postCas9-rep1_transfertest"),
    ("inDelphi", "LibA", "0226-PRLmESC-Lib1-Cas9_transfertest"),
    ]
TRANSFER_TEST_FILES = [
    ("inDelphi", "LibA", "052218-U2OS-+-LibA-postCas9-rep1_transfertest", "U2OS"),
    ("inDelphi", "LibA", "0226-PRLmESC-Lib1-Cas9_transfertest", "mESC-NHEJ-deficient"),
    ("FORECasT", "test", "TREX_A_test", "TREX_A"),
    ("FORECasT", "test", "HAP1_test", "HAP1"),
    # ("FORECasT", "test", "test", "WT")
]
TEST_FILE_STR = "./src/data/{}/{}{}.fasta"

def correct_inDelphi(o):
    o["TargetSequence"] = "GTCAT" + o["TargetSequence"] + "AGATCGGAAG"
    o["PAM Index"] = o["PAM Index"] + 5

    if o["Strand"] == "REVERSE":
        o["TargetSequence"] = str(Seq(o["TargetSequence"]).reverse_complement())
        o["PAM Index"] = o["PAM Index"] + 5
    return o

def read_test_file(test_file):
    if test_file in ["train", "test"]:
        oligo_f = TEST_FILE_STR.format("FORECasT", test_file, "")
        return list(get_details_from_fasta(oligo_f).values())
    elif test_file in ["WT", "WT2"]:
        oligo_f = TEST_FILE_STR.format("LUMC", test_file, ".forward")
        return list(get_details_from_fasta(oligo_f).values())
    else:
        oligo_f = TEST_FILE_STR.format("inDelphi", "LibA", ".forward")
        oligos = list(get_details_from_fasta(oligo_f).values())
        return [correct_inDelphi(o) for o in oligos] 
    
