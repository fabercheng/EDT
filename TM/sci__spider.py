from download import sci_hub_crawler
from scraping_using_lxml import get_link_xpath
from cache import Cache
from advanced_doi_crawler import doi_crawler


def sci_spider(savedrec_html_filepath, robot_url=None, user_agent='sheng', proxies=None, num_retries=2,
                delay=3, start_url='sci-hub.do', useSSL=True, gett_link=ge_link_xpath,
               nolimit=False, cache=None):
    """
    Given a literature export file (from Web of Science), download the corresponding pdf file (from sci-hub) (according to DOI)
    :param savedrec_html_filepath: An export file (.html) of search results with literature records (each record may or may not have a doi)
    :param robot_url: url of robots.txt on sci-bub
    :param user_agent: User agent, do not set to 'Twitterbot'
    :param proxies: proxy
    :param num_retries: Download retries
    :param delay: Download interval
    :param start_url: sci-hub Homepage Domain
    :param useSSL: Whether to enable SSL, the HTTP protocol name is 'https' after it is enabled
    :param get_link: Grab the function object of the download link, calling method get_link(html) -> html -- the requested web text
                     The function used is in scraping_using_%s.py % (bs4, lxml, regex), the default xpath selector is used
    :param nolimit: do not be limited by robots.txt if True
    :param cache: A cache object, which we use entirely as a dictionary in this code
    """
    print('trying to collect the doi list...')
    doi_list = doi_crawler(savedrec_html_filepath)  # Get doi list
    if not doi_list:
        print('doi list is empty, crawl aborted...')
    else:
        print('doi_crawler process succeed.')
        print('now trying to download the pdf files from sci-hub...')
        sci_hub_crawler(doi_list, robot_url, user_agent, proxies, num_retries, delay, start_url,
                    useSSL, get_link, nolimit, cache)
    print('Done.')


if __name__ == '__main__':
    from time import time
    start = time()
    filepath = './Users/devonne/Desktop/wos-endonote-doi.txt'  # The original html where the doi is located
    cache_dir = './cache.txt'  # Cache path
    cache = Cache(cache_dir)
    sci_spider(filepath, nolimit=True, cache=cache)
    print('time spent: %ds' % (time() - start))





