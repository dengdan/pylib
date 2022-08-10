
def kill(name):
    from utils.proc_util import pkill, ps_aux_grep
    import utils.str_util
    lines = utils.str_util.split(ps_aux_grep(name), '\n')
    excludes = ["kill_pro.py", "grep "]
    for line in lines:
        show = True
        for e in excludes:
            if e in line:
                show = False
                break
        if show:
            print(line)
    yes = input('kill them all?[n] y/n.')
    if yes == 'yes' or yes == 'y' or yes == 'Y':
        pkill(name, excludes = excludes)
    else:
        try:
            yes = int(yes)
            pkill(yes)
        except:
            pass

if __name__ == "__main__":
    import fire    
    fire.Fire(kill)
