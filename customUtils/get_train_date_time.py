from datetime import datetime
import pytz

kst = pytz.timezone('Asia/Seoul')
print(datetime.now(kst).strftime('%y%m%d_%H%M'))

# import ntplib
# from datetime import datetime, timezone, timedelta

# def get_kst_time_via_ntp(ntp_server='pool.ntp.org'):
#     client = ntplib.NTPClient()
#     response = client.request(ntp_server, version=3)
#     utc_time = datetime.fromtimestamp(response.tx_time, tz=timezone.utc)
#     kst = utc_time.astimezone(timezone(timedelta(hours=9)))
#     return kst

# if __name__ == "__main__":
#     print(get_kst_time_via_ntp().strftime('%Y-%m-%d %H:%M:%S'))
