
def get_pid(pattern, excludes = None):
    import utils.cmd_util
    cmd = 'ps aux|grep %s'%(pattern)
    results = utils.cmd_util.cmd(cmd)
    results = str(results).split('\n')
    pids = []
    for result in results:
        info = result.split()
        skip = False
        if excludes:
            for e in excludes:
                if e in info:
                    skip = True
                    break
        if skip:
            continue
        if len(info) > 0:
            pid = int(info[1])
            pids.append(pid)
    pids.sort()
    return pids
  
def pkill(pid, excludes = None):
    import utils.cmd_util
    if type(pid) == list:
        for p in pid:
            pkill(p)
    elif type(pid) == int:
        cmd = 'kill -9 %d'%(pid)
        print(utils.cmd_util.cmd(cmd))
    elif type(pid) == str:
        pids = get_pid(pid, excludes=excludes)
        pkill(pids)
    else:
        raise ValueError('Not supported parameter type:', type(pid))

def ps_aux_grep(pattern):
    import utils.cmd_util
    cmd = 'ps aux|grep %s'%(pattern)
    return utils.cmd_util.cmd(cmd)