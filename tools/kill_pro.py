
def kill(name):
    from utils.proc_util import pkill, ps_aux_grep
    import utils.str_util
    lines = utils.str_util.split(ps_aux_grep(name), '\n')
    for line in lines:
        if "kill_pro.py" not in line and "ps aux|grep " not in line:
            print(line)
    yes = input('kill them all?[n] y/n.')
    if yes == 'yes' or yes == 'y' or yes == 'Y':
        pkill(name, excludes = 'kill_pro.py')
    else:
        try:
            yes = int(yes)
            pkill(yes)
        except:
            pass

if __name__ == "__main__":
    import fire    
    fire.Fire(kill)
