import sys
import util
name = sys.argv[1]
print(util.proc.ps_aux_grep(name))
yes = input('kill them all?[n] y/n.')
if yes == 'yes' or yes == 'y' or yes == 'Y':
    util.proc.kill(name)
