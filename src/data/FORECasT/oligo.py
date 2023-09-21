import io, csv, os
from Bio import SeqIO

def get_oligo_ids(oligo_file):
    ids = []
    for record in SeqIO.parse(oligo_file, 'fasta'):
        oligo_idx = eval(record.id[5:].split('_')[0])
        ids.append(oligo_idx)
    return ids

def get_file_for_oligo_id(id):
    return "Oligo_{}".format(id)

def load_pam_lookup(filename):
    lookup = {}
    for record in SeqIO.parse(filename, "fasta"):
        toks = record.description.split()
        lookup[toks[0]] = (eval(toks[1]), toks[2])
    return lookup

def load_exp_oligo_lookup(exp_oligo_file, expanded_targets=False):
    lookup = {}
    for record in SeqIO.parse(exp_oligo_file, "fasta"):
        oligo_id, pam_loc, pam_dir = str(record.description).split()
        if not expanded_targets:
            oligo_id = oligo_id.split('_')[0]
        else:
            oligo_id = oligo_id[:oligo_id.rfind(":")]
        seq = str(record.seq)
        oligo_idx = oligo_id[5:]
        filename = get_file_for_oligo_id(oligo_idx)
        if filename not in lookup: lookup[filename] = []
        lookup[filename].append((oligo_id, pam_loc, pam_dir, seq))
    return lookup

#Removes the guide sequence from the ID
def get_short_oligo_id(full_oligo_id):
    return full_oligo_id.split('_')[0]

def split_exp_target_map_fasta(output_dir):
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    lines_per_file = 1000
    smallfile = None
    smallfiles = []
    with open(get_exp_oligo_file()) as bigfile:
        for lineno, line in enumerate(bigfile):
            if lineno % lines_per_file == 0:
                if smallfile:
                    smallfile.close()
                    smallfiles.append(smallfile)
                small_filename = '{}/small_file_{}.fasta'.format(output_dir, lineno + lines_per_file)
                smallfile = open(small_filename, "w")
            smallfile.write(line)
        if smallfile:
            smallfile.close()
            smallfiles.append(smallfile)
    return smallfiles