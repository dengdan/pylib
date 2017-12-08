import util
import sys


path = sys.argv[1]
suffix = None
if len(sys.argv) > 2:
    suffix = sys.argv[2]
    
files = util.io.ls('.', suffix)
print(len(files)) 