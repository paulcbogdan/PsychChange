import os
import pickle
import time
from pathlib import Path

import bs4 as bs
import pandas as pd
import requests
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service

from utils import read_csv_fast

chrome_options = ChromeOptions()
chrome_options.add_argument("--enable-javascript")
chrome_options.add_argument("--headless=new")

# You must fill out PATH_TO_SELENIUM with a path to your chromedriver
PATH_TO_CHROME = r'chrome-win64\chromedriver.exe'
service_chr = Service(PATH_TO_CHROME)
driver = None


def get_Wiley_xml(doi, ):
    url = fr'https://onlinelibrary.wiley.com/doi/full-xml/{doi}'
    print(f'Wiley: {url=}')
    return get_url(url)


def get_Springer_html(doi):
    url = fr'https://link.springer.com/article/{doi}'
    print(f'Springer-Nature: {url=}')
    return get_url(url)


def get_Frontiers_html(doi):
    url = fr'https://www.frontiersin.org/journals/psychology/articles/{doi}/full'
    print(f'Frontiers: {url=}')
    return get_url(url)


def get_SAGE_xml(doi, ):
    url = fr'https://journals.sagepub.com/doi/full-xml/{doi}'
    print(f'SAGE: {url=}')
    return get_url(url)


def get_url(url):
    global driver
    if driver is None:
        driver = webdriver.Chrome(service=service_chr, options=chrome_options)
    driver.get(url)
    html = driver.page_source
    html_lower = html.lower()
    if 'error 404' in html_lower or 'http 404' in html_lower:
        return '404'
    elif ('needs to review the security' in html_lower or
          "too many accesses" in html_lower):
        return 'security'
    return html


def get_Elsevier_API(doi):
    # fill out your API_KEY from Elsevier's developer portal (it's free)
    headers = {'X-ELS-APIKey': API_KEY,
               'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                             'AppleWebKit/537.36 (KHTML, like Gecko) '
                             'Chrome/84.0.4147.105 Safari/537.36',
               'Accept': 'application/xml'}
    url = 'https://api.elsevier.com/content/article/doi/' + doi
    url_for_print = url
    for i, (header, val) in enumerate(headers.items()):
        if header == 'user-agent':
            continue
        header = header.replace('X-ELS-APIKey', 'ApiKey')
        if i == 0:
            url_for_print += '?' + header + '=' + val.replace(r'/', '%2F')
        else:
            url_for_print += '&' + header + '=' + val.replace(r'/', '%2F')
    print(f'Elsevier API: {url_for_print}')
    try:
        r = requests.get(url, headers=headers)
    except requests.exceptions.ConnectionError:
        return 'security'
    if r.status_code == 404:
        return '404'
    elif r.status_code != 200:
        print('Strange Elsevier status code')
        print(f'\t{r.status_code=}')
        print(f'\t{doi=}')
        return 'security'

    print(f'\t{r.status_code=}')
    data = bs.BeautifulSoup(r.text, 'lxml')
    return str(data)


def get_by_publisher(doi, publisher):
    if 'Elsevier' in publisher:
        return get_Elsevier_API(doi)
    elif publisher == 'Wiley':
        return get_Wiley_xml(doi)
    elif publisher == 'SAGE_Publications':
        return get_SAGE_xml(doi)
    elif publisher == 'Frontiers_Media_SA':
        return get_Frontiers_html(doi)
    elif publisher == 'Springer_Science_and_Business_Media_LLC':
        return get_Springer_html(doi)
    else:
        raise ValueError


def process_URL_list(url_l_str, publisher):
    # from lens.org 'Source URLs'
    #   (may be plural multiple, separated by semicolons and/or spaces)
    if pd.isna(url_l_str):
        return None
    if ';' in url_l_str:
        url_l = url_l_str.split(';')
        url_l_str = ' '.join(url_l)
    url_l = url_l_str.split(' ')
    url_l = [url for url in url_l if r'/pdf' not in url and
             r'.pdf' not in url]
    if publisher == 'Springer_Science_and_Business_Media_LLC':
        url_l = [url for url in url_l if 'link.springer.com' in url]
    elif publisher == 'Frontiers_Media_SA':
        url_l = [url for url in url_l if 'frontiersin.org' in url or
                 'doi.org' in url]
    if len(url_l) == 0:  # For Springer, a URL will be attempted from the DOI
        return None
    return url_l[0]


def download_fulltexts(wait_time=16, attempts=5):
    dir_out_root = r'../doi_tree_html'
    Path(dir_out_root).mkdir(parents=True, exist_ok=True)
    df = read_csv_fast(fr'../dataframes/df_lens_Aug24.csv')

    df = df.sort_values('year', ascending=False)
    print(df['publisher'].value_counts())

    st_time_html = time.time()
    for idx, row in df.iterrows():
        doi = row['doi']
        publisher = row['publisher']
        year = row['year']
        try:
            journal = row['journal'].replace(':', '')
        except AttributeError as e:  # seemingly a CrossRef/Lens error. rare
            print(f'Missing journal ({publisher}, {year}): {doi=}')
            continue

        doi = doi.split(';')[0]
        doi = doi.replace('https://doi.org/', '')
        doi = doi.split(' ')[0]
        doi_str = doi.replace('/', '_').replace('.', '-')
        doi_str = doi_str.replace('(', '_').replace(')', '_')

        fp_out = rf'{dir_out_root}/{publisher}/{journal}/{year}/{doi_str}.html'
        if os.path.exists(fp_out):
            continue
        fp_404 = fr'../doi_tree_html/404s/{doi_str}'
        if os.path.exists(fp_404):
            continue

        url_l_str = row['source_url']

        try:
            html = get_by_publisher(doi, publisher)
        except WebDriverException as e:
            # rare case where the lens.org link doesn't work
            print(f'WebDriverException:', url_l_str)
            print(f'{e=}, publisher: {publisher}, DOI: {doi}')
            continue
        except TimeoutException as e:  # Error on my end, mostly for debugging
            print(f'TimeoutException:', url_l_str)
            print(f'{e=}, publisher: {publisher}, DOI: {doi}')
            continue

        if html == '404':  # Avoid re-rerunning known 404s. Save & ignore later
            print(f'\tHTML 404ed')
            fp_404 = fr'../doi_tree_html/404s/{doi_str}'
            try:
                with open(fp_404, 'wb') as f:
                    pickle.dump('', f)
            except OSError:
                print(f'Failed to save 404: {fp_404}')
                continue

            end_time_html = time.time()
            time_dif = end_time_html - st_time_html
            if time_dif < .5:
                time.sleep(.5 - time_dif)
            st_time_html = time.time()

            continue
        elif html == 'security':
            print(f'\tSecurity/API blocked. Waiting {wait_time} seconds...')
            if publisher != 'Elsevier_BV':
                global driver
                driver = None
            cnt = 0
            hit_after_waiting = False
            while html == 'security':
                time.sleep(wait_time * (2 ** cnt))
                try:
                    html = get_by_publisher(doi, publisher)
                except WebDriverException as e:
                    print(f'WebDriverException:', url_l_str)
                    print(f'{e=}, publisher: {publisher}, DOI: {doi}')
                    break
                print(f'\tSecurity/API block attempt #: {cnt}')
                cnt += 1
                if cnt == attempts:
                    print('\tGiving up')
                    break
            else:
                print('\tGot it, patience is a virtue...')
                hit_after_waiting = True
            if not hit_after_waiting:
                print('\tFailed to get URL')
                continue
        elif html == 'bad_URL':
            raise ValueError
        if html == 'security': continue

        end_time_html = time.time()
        time_dif = end_time_html - st_time_html
        print(f'\tSuccess! (Time needed: {time_dif:.2f} s)')
        st_time_html = time.time()
        # Elsevier is a champ and can take usually API calls quickly
        if publisher == 'Elsevier_BV':
            pass
        else:  # Retrieve non-Elsevier no faster than two per second
            #   If you start retrieving too quickly, you will trigger
            #   security/API blocks
            if time_dif < .5:
                time.sleep(.5 - time_dif)

        Path(os.path.dirname(fp_out)).mkdir(parents=True, exist_ok=True)
        with open(fp_out, 'w', encoding='utf-8') as file:
            file.write(html)


if __name__ == '__main__':
    download_fulltexts()
