import time
import math
import pandas as pd
import numpy as np

class SequenceTooShort(Exception):
    pass

def gen_indel(sequence, cut_site, max_deletion_length=30):
    '''This is the function that used to generate all possible unique indels and 
    list the redundant classes which will be combined after'''
    nt = ['A','T','C','G']
    up = sequence[:cut_site]
    down = sequence[cut_site:]
    # default values
    # window_start = -2
    # window_end = +3

    # Each indel must at least touch the cutsite
    window_start = 1
    window_end = 1

    if min(len(up), len(down)) < max_deletion_length:
        raise SequenceTooShort()

    dmax = max_deletion_length
    uniqe_seq ={}
    for dstart in range(1,cut_site+window_end):
        for dlen in range(1,dmax):
            if len(sequence) > dlen+dstart > cut_site-window_start:
                seq = sequence[0:dstart]+sequence[dstart+dlen:]
                indel = sequence[0:dstart] + '-'*dlen + sequence[dstart+dlen:]
                array = ["{}+{}".format(dstart-cut_site, dlen),indel,sequence,cut_site,'del',dstart-cut_site,dlen,None,None,None]
                try: 
                    uniqe_seq[seq]
                    if dstart-cut_site <1:
                        uniqe_seq[seq] = array
                except KeyError: uniqe_seq[seq] = array
    for base in nt:
        seq = sequence[0:cut_site]+base+sequence[cut_site:]
        indel = sequence[0:cut_site]+'-'+sequence[cut_site:]
        array = ["1+{}".format(base),sequence,indel,cut_site,'ins',0,1,base,None,None]
        try: uniqe_seq[seq] = array
        except KeyError: uniqe_seq[seq] = array
        for base2 in nt:
            seq = sequence[0:cut_site] + base + base2 + sequence[cut_site:]
            indel = sequence[0:cut_site]+'--'+sequence[cut_site:]
            array = ["2+{}".format(base+base2),sequence,indel,cut_site,'ins',0,2,base+base2,None,None]
            try: uniqe_seq[seq] = array
            except KeyError:uniqe_seq[seq] = array
    uniq_align = label_mh(list(uniqe_seq.values()),4)
    for read in uniq_align:
        if read[-2]=='mh':
            merged=[]
            for i in range(0,read[-1]+1):
                merged.append((read[5]-i,read[6]))
            read[-3] = merged
    return uniq_align

def label_mh(sample,mh_len):
    '''Function to label microhomology in deletion events'''
    for k in range(len(sample)):
        read = sample[k]
        # if read[-4] == 15 and read[-5] == 0:
        #     print("del size 15")
        if read[4] == 'del':
            idx = read[3] + read[5]
            idx2 = idx + read[6]
            x = mh_len if read[6] > mh_len else read[6]
            for i in range(x,0,-1):
                if read[2][idx-i:idx] == read[2][idx2-i:idx2] and i <= read[6]:
                    sample[k][-2] = 'mh'
                    sample[k][-1] = i
                    break
            if sample[k][-2]!='mh':
                sample[k][-1]=0
    return sample

def gen_indels_v3(seq, cutsite, max_deletion_length=30, deletion_window_length=30, right_side_offset=-1, left_side_offset=0, verbose=False):
    unique_seq = {}
    if verbose: 
        start_time = time.time()
    for del_len in range(1, max_deletion_length+1):
        for pos in range(deletion_window_length+left_side_offset, -deletion_window_length+del_len+right_side_offset-1, -1):
            valid = False
            # start at cutsite, move backwards to start at cutsite - del
            left_edge = cutsite - del_len + pos
            right_edge = cutsite + pos
            if right_edge >= len(seq) : continue 
            if left_edge <= 0: 
                if verbose: print("outside bounds"); 
                break
            valid = (left_edge <= cutsite + left_side_offset) and (right_edge >=cutsite+right_side_offset)

            # split seq
            left_seq = seq[:left_edge]
            right_seq = seq[right_edge:]
            del_seq = seq[left_edge:right_edge] 

            # generated all sequences for each deletion length and position
            printable_seq = left_seq + "-" * del_len + right_seq
            new_seq = left_seq + right_seq
            if verbose and valid: print("{} {} {}".format(printable_seq, valid, pos))

            # detail each unique sequence, append all possible positions of MH
            relative_pos = left_edge - cutsite
            if new_seq not in unique_seq:
                unique_seq[new_seq] = {
                    "Type": "DELETION",
                    "Size": del_len,
                    "Start": relative_pos,
                    "Positions": [],
                    "DelSeq": del_seq,
                    "homologyLength": 0,
                    "homology": "",
                    "numRepeats": 0,
                    "leftFlank": left_seq,
                    "rightFlank": right_seq,
                    "valid": valid
                }
            unique_seq[new_seq]["Positions"].append(relative_pos)
            unique_seq[new_seq]["valid"] = unique_seq[new_seq]["valid"] | valid

    for k in list(unique_seq.keys()):
        # remove invalid sequences (not touching the cutsite)
        if not unique_seq[k]["valid"]:
            del unique_seq[k]
            continue
        del unique_seq[k]["valid"]
        
        # add microhomology details
        if len(unique_seq[k]["Positions"]) > 1:
            mh_len = len(unique_seq[k]["Positions"]) - 1

            # if deletion length > mh length = microhomology or single repeat
            if unique_seq[k]["Size"] > mh_len:
                unique_seq[k]["homologyLength"] = mh_len
                unique_seq[k]["homology"] = unique_seq[k]["DelSeq"][-mh_len:]
            # if mh_len >= deletion length = repeats (maybe even multiple)
            else:
                n_repeats, _ = divmod(mh_len, unique_seq[k]["Size"])
                unique_seq[k]["homologyLength"] = unique_seq[k]["Size"]
                unique_seq[k]["numRepeats"] = n_repeats
                unique_seq[k]["homology"] = unique_seq[k]["DelSeq"]

    # add insertions
    nucleotides = ['A', 'C', 'T', 'G']
    for nt1 in nucleotides:
        left_flank = seq[:cutsite]
        right_flank = seq[cutsite:]
        new_seq = left_flank + new_seq + right_flank

        unique_seq[new_seq] = {
            "Type": "INSERTION",
            "Size": 1,
            "Start": 0,
            "Positions": [0],
            "InsSeq": nt1,
            "leftFlank": left_seq,
            "rightFlank": right_seq,
            "valid": True
        }

        for nt2 in nucleotides:
            ins_seq = nt1 + nt2
            new_seq = left_flank + ins_seq + right_flank
            unique_seq[new_seq] = {
                "Type": "INSERTION",
                "Size": 2,
                "Start": 0,
                "Positions": [0],
                "InsSeq": ins_seq,
                "leftFlank": left_seq,
                "rightFlank": right_seq,
                "valid": True
            }

    new_seq = left_flank + "X" + right_flank
    unique_seq[new_seq] = {
        "Type": "INSERTION",
        "Size": 3,
        "Start": 0,
        "Positions": [0],
        "InsSeq": "X",
        "leftFlank": left_seq,
        "rightFlank": right_seq,
        "valid": True
    }


    indels_df = pd.DataFrame(list(unique_seq.values()))
    indels_df["Indel"] = indels_df.apply(lambda row: "{}+{}".format(row["Start"] if row["Type"] == "DELETION" else row["Size"], row["Size"] if row["Type"] == "DELETION" else row["InsSeq"] ), axis=1)
    
    if verbose:
        print("--- %s seconds ---" % (time.time() - start_time))
    
    return indels_df

if __name__ == "__main__":
    seq = "TAAAAGATAAATATTCAGAATCTTCTTTTTAATTCCTGATTTTATTTCTATAGGACTGAAAGACTTGCTCGAGATGTCATGAAGGAGATGGGAGGCCATCACATTGTGGCCCTCTGTGTGCTCAAGGGGGGCTATAAGTTCTTTGCTGACCTGCTGGAT"
    cutsite = 82
    indels = gen_indels_v3(seq, cutsite, verbose=True, max_deletion_length=60, right_side_offset=0,left_side_offset=0)
    print(indels)
  