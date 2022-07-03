#encoding = utf-8

def cmd(cmd, decode = "utf-8"):
    try:
        import commands
        return commands.getoutput(cmd)
    except:
        import subprocess
        return subprocess.check_output(cmd, shell = True).decode(decode)
        
#         return os.system(cmd)