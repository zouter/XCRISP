import os, sys, io, csv, time

sys.path.append("../../common")
from data import get_output_folder, get_exp_oligo_file
from oligo import get_file_for_oligo_id, get_oligo_ids

def closeFiles(fhandles):
    for id in fhandles:
        fhandles[id].close()

def write_to_file(oligo_id, lines, mapped_reads_dir):
    print("Writing...")
    o = mapped_reads_dir + get_file_for_oligo_id( oligo_id )
    if not os.path.exists(o): os.makedirs(o)
    filename = get_file_for_oligo_id( oligo_id )
    filename += '_mappings.txt'
    fout = io.open(o + '/' + filename, 'w')
    for line in lines:
        fout.write(u'%s\n' % line)
    fout.close()

def split_null_mappings(oligo_ids, mapping_files_dir, mapped_reads_dir):
    start_time = time.time()
    lines = {}
    mapping_files = os.listdir(mapping_files_dir)
    for i, map_file in enumerate(mapping_files):
        f = io.open(mapping_files_dir + map_file)
        rdr = csv.reader(f, delimiter='\t')

        for toks in rdr:
            if '@@@' in toks[0]: continue
            oid = toks[1].split()[0]
            if oid == 'None':
                continue
            if ',' in oid:
                continue # Ignore ambiguously mapped reads
            oidx = eval(oid[5:].split('_')[0])
            
            if (oidx in oligo_ids):
                if oidx not in lines: lines[oidx] = []
                lines[oidx].append('\t'.join(toks))
        print("Finished {} of {} files".format(i, len(mapping_files)))

    for oidx in oligo_ids:
        if oidx in lines and len(lines[oidx]) > 0:
            print("Writing {} lines to {}".format(len(lines[oidx]), oidx))
            write_to_file(oidx, lines[oidx], mapped_reads_dir)

    print("--- %s seconds ---" % (time.time() - start_time))
    print("Finished Oligo %s" % oligo_id)


if __name__ == "__main__":  
    exp_oligo_file = get_exp_oligo_file()
    mapping_files_dir = get_output_folder() + "/mapping_files/NULL/"
    mapped_reads_dir = get_output_folder() + "/mapped_reads/NULL/"
    if not os.path.exists(mapped_reads_dir): os.makedirs(mapped_reads_dir)

    if len(sys.argv) > 2:
        oligo_id = [eval(x) for x in sys.argv[1:]]
        split_null_mappings(oligo_id, mapping_files_dir, mapped_reads_dir)
    else:
        oligo_ids = [str(x) for x in get_oligo_ids(exp_oligo_file)]
        n = len(oligo_ids)
        step = 2000
        for i in range(34000, n, step):
            end = n if (i+step > n) else i+step
            cmd = "python3 split_null_mappings.py {}\n".format(" ".join(oligo_ids[i:end]))
            print(cmd)
            os.system(cmd)

