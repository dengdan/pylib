import util
cmd = 'cat /proc/cpuinfo | grep name | cut -f2 -d: | uniq -c'
print(util.cmd.cmd(cmd))