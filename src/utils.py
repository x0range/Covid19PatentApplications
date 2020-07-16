import logging
import os


from settings import LOG_DIR


def setup_logging(log_name: str, level:str = logging.INFO):
    """
    Call this function from your scripts with os.path.basename(__file__) to initialize a log with name of your file in
    the LOG_DIR. Logs can be used to monitor your scripts with 'tail -f logfile...'
    You can switch the logleve
    @param log_name:
    @param level:
    """
    logging.basicConfig(filename=os.path.join(LOG_DIR, log_name + '.log'),
                        level=level,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')


def get_headers():
    """
    Return http headers to make requests
    @return:
    """
    return {
        'Connection': 'keep-alive',
        'Pragma': 'no-cache',
        'pragma': 'no-cache',
        'cache-control': 'no-cache',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:77.0) Gecko/20100101 Firefox/77.0',
        'dnt': '1',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3',
        'accept-encoding': 'gzip, deflate, br',
        'accept-language': 'en-US;q=0.8,en;q=0.7',
    }
