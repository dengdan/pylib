def timestamp2str(timestamp, fmt = "%Y-%m-%d %H:%M:%S.%f"): 
    import datetime
    d = datetime.datetime.fromtimestamp(timestamp) 
    str1 = d.strftime(fmt) 
    # 2015-08-28 16:43:37.283000' 
    return str1 
  

def get_timestamp(s, fmt = "%Y-%m-%d %H:%M:%S"):
    import time
    timeArray = time.strptime(s, fmt)
    timestamp = time.mktime(timeArray)
    return timestamp