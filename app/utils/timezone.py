# 中国时区
import pytz


CN_TIMEZONE = pytz.timezone('Asia/Shanghai')

# 时区转换函数


def to_cn_timezone(dt):
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=pytz.UTC)
    return dt.astimezone(CN_TIMEZONE)
