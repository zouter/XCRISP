import os, io, csv
import numpy as np

from indel import tok_full_indel

def get_profile_counts(profile):
    total = sum([profile[x] for x in profile])
    if total == 0:
        return []
    indel_total = total
    if '-' in profile:
        indel_total -= profile['-']
        null_perc = profile['-']*100.0/indel_total if indel_total != 0 else 100.0
        null_profile = (profile['-'],'-',profile['-']*100.0/total, null_perc)
    counts = [(profile[x],x, profile[x]*100.0/total, profile[x]*100.0/indel_total) for x in profile if x != '-']
    counts.sort(reverse = True)
    if '-' in profile:
        counts = [null_profile] + counts
    return counts

def get_highest_indel(p1):
    cnts = [(p1[x],x) for x in p1]
    cnts.sort(reverse=True)
    if (len(p1) == 1 and '-' in p1) or len(p1) == 0:
        return '-'
    if cnts[0][1] == '-' and len(cnts)>1:
        return cnts[1][1]
    else:
        return cnts[0][1]

#Read in profile from indel summary file
def read_summary_to_profile(filename, profile, oligoid=None, noexclude=False, remove_long_indels=False, remove_wt=False, wt_thresh=3.0):
    if not os.path.isfile(filename): return 0,0,0
    
    dirname = '/'.join(filename.split('/')[:-3])
    filename_suffix = '/'.join(filename.split('/')[-3:])
    wt_p, wt_p_wfilter = {}, {}
    # if 'WT' not in dirname and dirname != '' and not noexclude and remove_wt:
    #     wt_filename = get_WT_dir(dirname) + '/' + filename_suffix
    #     #if wt_filename[0] == '/' and wt_filename[1:7] != 'lustre': wt_filename = wt_filename[1:]
    #     if not os.path.isfile(wt_filename):
    #         print('Warning: Could not find', wt_filename)
    #     else:
    #         read_summary_to_profile(wt_filename, wt_p, oligoid=oligoid, noexclude=True, remove_wt=False)
    #         _, wt_acc, _ = read_summary_to_profile(wt_filename, wt_p_wfilter, oligoid=oligoid, noexclude=False, remove_wt=False)
    #         if wt_acc < 10.0: return 0,0,0    #Need at least 20% acceptable reads in the wild type
    #                                       #(to remove oligos that are really messed up)

    total, accepted = 0,0
    f = io.open(filename)
    reader = csv.reader(f, delimiter='\t')
    if '-' not in profile:
        profile['-'] = 0
    orig_null = profile['-']
    curr_oligo_id = None
    wt_indels = []
    for toks in reader:
        if toks[0][:3] == '@@@':
            curr_oligo_id = toks[0][3:].split()[0]
            continue
        if oligoid != curr_oligo_id:
            continue
        indel = toks[0]
        oligo_indel = toks[1]
        num_reads = eval(toks[2])
        total += num_reads
        if not noexclude:
            if oligo_indel != '-':
                if not is_allowable_oligo_indel(oligo_indel):
                    continue
            #Only allow indels that span the cut site and which are
            #not present in the corresponding WT sample
            if indel != '-':
                itype, isize, details, muts = tok_full_indel(indel)
                if itype != '-' and (details['L'] > 5 or details['R'] < -5):
                    continue
                if remove_long_indels and isize > 30:
                    continue
                if indel in wt_p and remove_wt: 
                    #Check the levels of the indel in the WT sample,
                    #only include it if present at at least 3 x that level (including NULLS)
                    # - will need to wait til we know total reads to do this
                    wt_indels.append((indel, num_reads))
                    continue
        if indel not in profile:
            profile[indel] = 0
        profile[indel] += num_reads
        accepted += num_reads
    for indel, num_reads in wt_indels:
        if num_reads*1.0/total > wt_p[indel]*wt_thresh/sum([wt_p[x] for x in wt_p]):
            if indel not in profile: profile[indel] = 0
            profile[indel] += num_reads
            accepted += num_reads
    f.close()
    if total == 0:
        perc_accepted = 0.0
    else:
        perc_accepted = accepted*100.0/total
    return accepted, perc_accepted, profile['-']-orig_null

def is_allowable_oligo_indel(oligo_indel):
    itype, isize, details, muts = tok_full_indel(oligo_indel)
    #Exclude reads from oligos with any mutations in the guide or PAM sequence
    is_ok = True
    mut_locs = [x for x in muts if x[0] not in ['N','I','D']]
    if len(mut_locs) > 0: 
        if any([x[1] > -20 and x[1] < 6 for x in mut_locs]):
            is_ok = False
        if len(mut_locs) > 5:
            is_ok = False
    #Only allow oligo indels if they're size 1 or 2 insertion/deletions outside the guide or PAM sequence
    ins_del_muts = [x for x in muts if x[0] in ['I','D']]
    if len(ins_del_muts) > 0:
        if any([x[1] > 2 for x in ins_del_muts]):
            is_ok = False
    if oligo_indel[0] != '-':
        if isize > 2 or (details['L'] < 6 and details['R'] > -20):   
            is_ok = False
    return is_ok

def fetch_indel_size_counts(p1):
    inframe, outframe, size_counts, = 0,0,{'I':{},'D':{}}
    for i in range(1,21):
        size_counts['I'][i] = 0
        size_counts['D'][i] = 0
    for indel in p1:
        if indel == '-':
            continue
        itype,isize,details, muts = tok_full_indel(indel)
        net_isize = isize - details['I'] - details['D']
        if net_isize % 3 == 0:
            inframe += p1[indel]
        else:
            outframe += p1[indel]
        if net_isize not in size_counts[itype]:
            size_counts[itype][net_isize] = 0
        size_counts[itype][net_isize] += p1[indel]
    return inframe, outframe, size_counts