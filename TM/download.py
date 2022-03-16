import requests
from urllib.robotparser import RobotFileParser
import time
from urllib.parse import urlparse
from filename import get_valid_filename


def doi_parser(doi, start_url, useSSL=True):
    """Parse doi to url"""
    HTTP = 'https' if useSSL else 'http'
    url = HTTP + '://{}/{}'.format(start_url, doi)
    return url


def get_robot_parser(robot_url):
    rp = RobotFileParser()
    rp.set_url(robot_url)
    rp.read()
    return rp


def wait(url, delay=3, domains={}):
    """wait until the interval between two
    downloads of the same domain reaches time delay"""
    domain = urlparse(url).netloc  # get the domain
    last_accessed = domains.get(domain)  # the time last accessed
    if delay > 0 and last_accessed is not None:
        sleep_secs = delay - (time.time() - last_accessed)
        if sleep_secs > 0:
            time.sleep(sleep_secs)
    domains[domain] = time.time()


def download(url, headers, proxies=None, num_retries=2):
    print('Downloading: ', url)
    try:
        resp = requests.get(url, headers=headers, proxies=proxies, verify=False)
        html = resp.text
        if resp.status_code >= 400:
            print('Download error: ', resp.text)
            html = None
            if num_retries and 500 <= resp.status_code < 600:
                return download(url, headers, proxies, num_retries-1)
    except requests.exceptions.RequestException as e:
        print('Download error', e)
        return None
    return html


def download_pdf(result, headers, proxies=None, num_retries=2, doi=None):
    url = result['onclick']
    components = urlparse(url)
    if len(components.scheme) == 0:
        url = 'https:{}'.format(url)
    print('Downloading: ', url)
    try:
        resp = requests.get(url, headers=headers, proxies=proxies, verify=False)
        if resp.status_code >= 400:
            print('Download error: ', resp.status_code)
            if num_retries and 500 <= resp.status_code < 600:
                return download(result, headers, proxies, num_retries-1)
        if len(result['title']) < 5:  # Handling the case where the title is empty
            filename = get_valid_filename(doi) + '.pdf'
        else:
            filename = get_valid_filename(result['title']) + '.pdf'
        print(filename)
        #  ok, let's write it to file
        with open(filename, 'wb') as fp:
            fp.write(resp.content)
    except requests.exceptions.RequestException as e:
        print('Download error', e)
        return False
    return True


def sci_hub_crawler(doi_list, robot_url=None, user_agent='sheng', proxies=None,num_retries=2,
                delay=3, start_url='sci-hub.do', useSSL=True, get_link=None, nolimit=False, cache=None):
    """
    Given a list of literature doi, crawl the pdf file of the corresponding document
    :param doi_list: doi list
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
    :return:
    """
    headers = {'User-Agent': user_agent}
    HTTP = 'https' if useSSL else 'http'
    if not get_link:
        print('Crawl failed, no get_link method.')
        return None
    if not robot_url:
        robot_url = HTTP + '://{}/robots.txt'.format(start_url)
    # print(robot_url)
    try:
        rp = get_robot_parser(robot_url)
    except Exception as e:
        rp = None
        print('get_robot_parser() error: ', e)
    domains={}  # save the timestamp of accessed domains
    download_succ_cnt:int = 0  # the number of pdfs that're successfully downloaded
    for doi in doi_list:
        url = doi_parser(doi, start_url, useSSL)
        if cache and cache[url]:
            print('already downloaded: ', cache[url])
            download_succ_cnt += 1
            continue
        if rp and rp.can_fetch(user_agent, url) or nolimit:
            wait(url, delay, domains)
            html = download(url, headers, proxies, num_retries)
            result = get_link(html)
            if result and download_pdf(result, headers, proxies, num_retries, doi):
                if cache:
                    cache[url] = 'https:{}'.format(result['onclick'])  # cache
                download_succ_cnt += 1
        else:
            print('Blocked by robots.txt: ', url)
    print('%d of total %d pdf success' % (download_succ_cnt, len(doi_list)))


if __name__ == '__main__':
    from scraping_using_lxml import get_link_xpath, get_link_cssselect
    from scraping_using_bs4 import get_link_using_bs4
    from scraping_using_regex import get_link_using_regex
    from random import choice
    """k_cssselect, get_link_using_bs4, get_link_using_regex]
    # get_link = choice(get_links_methods)
    get_link = get_link_xpath
    print('use %s as get_link_method.' % get_link.__name__)
    # print('obey the limits in robots.txt: ')
    # sci_hub_crawler(dois, get_link = get_link, user_agent='sheng')
    print('no any limit: ')
    sci_hub_crawler(dois, get_link = get_link, user_agent='sheng', nolimit=True)
    print('Done.')"""