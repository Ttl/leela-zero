import sys
from subprocess import Popen, PIPE, STDOUT
import os
import glob
import multiprocessing

leela_cmd = "../../src/leelaz -w ../../src/tests/32_4_se.txt --cpu-only"

out_dir = "../../converted_chunks/"

def process_chunk(chunk):
    print(chunk)
    chunk_name = os.path.split(chunk)[-1]
    chunk_name = os.path.splitext(chunk_name)[0]
    out = os.path.join(out_dir, chunk_name)
    out += "_f"
    cmd = leela_cmd.split(' ')
    p = Popen(cmd, stdin=PIPE, stderr=PIPE, stdout=PIPE)
    out, err = p.communicate("add_features {} {}".format(chunk, out))
    if p.returncode != 0:
        print(err)
        print('leelaz, non-zero return code: {}'.format(p.returncode))

if __name__ == "__main__":
    try:
        os.makedirs(out_dir)
    except OSError:
        pass
    chunks = glob.glob(sys.argv[1] + "*.gz")
    print("Found {} chunks".format(len(chunks)))
    pool = multiprocessing.Pool(processes=8)
    # Python2 Keyboardinterrupt workaround
    results = pool.map_async(process_chunk, chunks).get(999999999999999)
