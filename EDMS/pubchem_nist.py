import requests
from bs4 import BeautifulSoup
import time

def get_detailed_page_link(compound_page_url):
    response = requests.get(compound_page_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    link_element = soup.select_one("#maincontent > div > div:nth-child(5) > div > div > p > a")
    return link_element['href'] if link_element else None

def get_gc_ms_data(detailed_url):
    response = requests.get(detailed_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    selectors = [
        "#GC-MS > div.px-1.py-3.space-y-2 > div:nth-child(3) > div:nth-child(2) > div > div > div:nth-child(4) > "
        "div.text-left.sm\\:table-cell.sm\\:align-middle.sm\\:p-2.sm\\:border-t.sm\\:border-gray-300.dark\\:sm"
        "\\:border-gray-300\\/20.pb-1.pl-2 > div",
        "#GC-MS > div.px-1.py-3.space-y-2 > div:nth-child(3) > div:nth-child(2) > div > div > div:nth-child(5) > "
        "div.text-left.sm\\:table-cell.sm\\:align-middle.sm\\:p-2.sm\\:border-t.sm\\:border-gray-300.dark\\:sm"
        "\\:border-gray-300\\/20.pb-1.pl-2 > div",
        "#GC-MS > div.px-1.py-3.space-y-2 > div:nth-child(3) > div:nth-child(2) > div > div > div:nth-child(6) > "
        "div.text-left.sm\\:table-cell.sm\\:align-middle.sm\\:p-2.sm\\:border-t.sm\\:border-gray-300.dark\\:sm"
        "\\:border-gray-300\\/20.pb-1.pl-2 > div"
    ]

    gc_ms_data = []
    for selector in selectors:
        element = soup.select_one(selector)
        if element:
            gc_ms_data.append(element.get_text().strip())

    return gc_ms_data


def get_compound_links(page_url):
    pass


def main():
    base_url = "https://www.ncbi.nlm.nih.gov/pcsubstance"
    query_params = "?term=%22NIST%20Mass%20Spectrometry%20Data%20Center%22%5BSourceName%5D%20AND%20hasnohold%5Bfilt%5D"

    total_pages = 13298

    for page in range(1, total_pages + 1):
        page_url = f"{base_url}{query_params}&page={page}"
        compound_links: None = get_compound_links(page_url)

        for link in compound_links:
            compound_page_url = "https://www.ncbi.nlm.nih.gov" + link
            detailed_page_link = get_detailed_page_link(compound_page_url)

            if detailed_page_link:
                gc_ms_data = get_gc_ms_data("https://www.ncbi.nlm.nih.gov" + detailed_page_link)
                print(gc_ms_data)

        time.sleep(1)

if __name__ == "__main__":
    main()