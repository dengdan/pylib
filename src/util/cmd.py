#encoding = utf-8

def cmd(cmd):
    try:
        import  commands
        return commands.getoutput(cmd)
    except:
        import os
        return os.system(cmd)
