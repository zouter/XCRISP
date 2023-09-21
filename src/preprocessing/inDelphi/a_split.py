import sys, os, fnmatch, datetime, subprocess, imp
sys.path.append('/scratch/cfseale/repos/')
# https://github.com/maxwshen/mylib
from tqdm import tqdm
from datetime import date

import signal
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

# Default params
inp_dir = os.environ["DATA_DIR"] + "inDelphi/"
out_dir = os.environ["OUTPUT_DIR"] + 'indelphi_a_split_{}/'.format(date.today())
os.makedirs(out_dir, exist_ok=True)
if len(os.listdir(out_dir)) > 0:
    os.system("rm {}*".format(out_dir))

##
# Functions
##

def line_count(fn):
  try:
    ans = subprocess.check_output(['wc', '-l', fn.strip()])
    ans = int(ans.split()[0])
  except OSError as err:
    print('OS ERROR:', err)
  return ans

def split(inp_fn, out_nm):
  inp_fn_numlines = line_count(inp_fn)

  num_splits = 60
  split_size = int(inp_fn_numlines / num_splits)
  if num_splits * split_size < inp_fn_numlines:
    split_size += 1
  while split_size % 4 != 0:
    split_size += 1
  print('Using split size {}'.format(split_size))

  split_num = 0
  for idx in tqdm(range(1, inp_fn_numlines, split_size)):
    start = idx
    end = start + split_size  
    out_fn = out_dir + out_nm + '_%s.fastq' % (split_num)
    command = 'tail -n +%s %s | head -n %s > %s' % (start, inp_fn, end - start, out_fn)
    split_num += 1
    if len(sys.argv) > 1 and sys.argv[1] == 'dry':
        print(command)
    else:
        os.system(command)

  return

##
# Main
##
def main(): 
  # Function calls
  files = os.listdir(inp_dir)

  files = [f for f in files if "SRR7536407" in f]

  for fn in files:
    print(fn)
    if fn[-2:] == "gz":
        os.system("gunzip {}".format(inp_dir + fn))
        fn = fn.replace(".gz", "")
    split(inp_dir + fn, fn.replace('.fastq', ''))

if __name__ == '__main__':
  main()
