import gc
import os
import re
from functools import cache
from pathlib import Path

import bs4
from tqdm import tqdm

from utils import read_csv_fast


@cache
def re_compile(re_str):
    return re.compile(re_str, re.IGNORECASE)


def general_parse(soup, search='h2'):
    val_str = '(result|finding)'
    results_secs_ = soup.find_all(search, string=re_compile(val_str))
    if len(results_secs_) > 0:
        results_secs = []
        for i in range(len(results_secs_)):
            for sibling in results_secs_[i].next_siblings:
                if sibling.name == search:
                    break
                results_secs.append(sibling)
        return results_secs
    else:
        return []


def process_non_APA(fp_text, publisher):
    with open(fp_text, 'r', encoding='utf-8') as f:
        html = f.read()

    if ('This XML file does not appear to have any style information '
        'associated with it.') in html:
        html = html.split(
            'This XML file does not appear to have any style information '
            'associated with it.')[0]

    soup = bs4.BeautifulSoup(html, features='lxml')  # TODO: maybe test 'xml'

    tag_str = '(sec|section)'
    val_str = '(result|finding)'

    for link in soup.find_all('link', href=True):
        link.decompose()

    if publisher == 'Frontiers_Media_SA':
        for caption in soup.find_all(class_='FigureDesc'):  # Frontiers
            caption.decompose()
        for caption in soup.find_all(class_='TableDesc'):  # Frontiers
            caption.decompose()
        for caption in soup.find_all(class_='imagedescription'):  # Frontiers
            caption.decompose()
        for table in soup.select('tabular'):
            table.decompose()
        for table in soup.select('table'):
            table.decompose()

        results_secs = general_parse(soup, search='h2')
        results_secs_ = []
        for sec in results_secs:
            try:
                if 'teaser' in sec['class']: continue
            except AttributeError:
                pass
            except TypeError:
                pass
            except KeyError:
                pass
            if len(sec.text) < 100:
                continue
            results_secs_.append(sec)
        results_secs = results_secs_

        if len(results_secs) == 0:
            results_secs = general_parse(soup, search='h3')


    elif publisher == 'Elsevier_BV':
        for caption in soup.select(r'ce\:table-footnote'):  # Elsevier
            caption.decompose()
        for caption in soup.select(r'ce\:simple-para'):  # Elsevier
            caption.decompose()

        results_secs = soup.find_all('ce:section-title',
                                     string=re_compile(val_str), )
        results_secs = [result.parent for result in results_secs]

    elif publisher == 'Springer_Science_and_Business_Media_LLC':
        for caption in soup.find_all(
                class_="c-article-section__figure-description"):  # Springer
            caption.decompose()
        for table in soup.select('tabular'):  # Wiley
            table.decompose()
        for table in soup.select('table'):
            table.decompose()
        results_secs = soup.find_all(re_compile(tag_str),
                                     {'data-title': re_compile(val_str)})

        if len(results_secs) == 0:
            results_secs = general_parse(soup, search='h2')
            if len(results_secs) == 0:
                results_secs = general_parse(soup, search='h3')


    elif publisher == 'Wiley':
        for caption in soup.select('caption'):  # Wiley
            caption.decompose()
        for caption in soup.select('note'):  # Wiley
            caption.decompose()

        for table in soup.select('tabular'):  # Wiley
            table.decompose()
        for table in soup.select('table'):
            table.decompose()

        results_secs = soup.find_all('title', type='main',
                                     string=re_compile(val_str))
        results_secs = [result.parent for result in results_secs]

    elif publisher == 'SAGE_Publications':
        for caption in soup.select('table-wrap'):  # SAGE
            caption.decompose()

        results_secs = soup.find_all(re_compile(tag_str),
                                     {'sec-type': re_compile(val_str)})

        if len(results_secs) == 0:
            results_secs = soup.find_all('title', string=re_compile(val_str))
            results_secs = [result.parent for result in results_secs]

    if len(results_secs):
        # remove sections that are a subset of another subsection, unless they
        #   are identical, in which case just pick the first one
        results_secs_pruned = []
        for i in range(len(results_secs)):
            sec = results_secs[i].text
            for j in range(len(results_secs)):
                if i == j: continue
                sec_j = results_secs[j].text
                if sec in sec_j:
                    if len(sec) == len(sec_j):
                        if i < j:
                            results_secs_pruned.append(sec)
                    break
            else:
                results_secs_pruned.append(sec)

        result_text = (' - ' * 3).join([result for result in
                                        results_secs_pruned])

        result_text = (result_text.replace('Results', 'Results ').
                       replace('Result', 'Result ').
                       replace('Discussion', 'Discussion '))
        plain_text = soup.text
        # Elsevier p-values are regularly processed as p\n<\n0.05 or so
        result_text = result_text.replace('\n', ' ')
    else:
        plain_text = soup.text
        result_text = None

    plain_text = plain_text.replace('\n', ' ')

    return result_text, plain_text


def make_plaintext_files():
    df = read_csv_fast(fr'../dataframes/df_lens_Aug24.csv')
    html_dir = r'../doi_tree_html'

    df['doi_str'] = df['doi'].str.replace('/', '_').str.replace('.', '-')
    df.sort_values('year', inplace=True, ascending=True)

    for row in tqdm(df.itertuples(), desc='making plaintexts',
                    total=df.shape[0]):
        fp_in = (fr'{html_dir}\{row.publisher}'
                 fr'\{row.journal}\{row.year}\{row.doi_str}.html')
        if not os.path.exists(fp_in):
            continue

        fp_out = (fr'..\data\plaintexts_Aug24\{row.publisher}\{row.journal}' +
                  fr'\{row.year}\{row.doi_str}.txt')
        if os.path.exists(fp_out):
            continue
        fp_out_res = (fr'..\data\plaintexts_res_Aug24\{row.publisher}' +
                      fr'\{row.journal}\{row.year}\{row.doi_str}.txt')
        if os.path.exists(fp_out_res):
            continue

        Path(fp_out_res).parent.mkdir(parents=True, exist_ok=True)
        result_text, plain_text = process_non_APA(fp_in, row.publisher)

        if result_text is not None:
            with open(fp_out_res, 'w', encoding='utf-8') as file:
                file.write(result_text)
            del result_text

        Path(fp_out).parent.mkdir(parents=True, exist_ok=True)
        with open(fp_out, 'w', encoding='utf-8') as file:
            file.write(plain_text)

        del plain_text
        gc.collect()  # Might just be a personal computer thing, but
        #  I've been having memory issues


if __name__ == '__main__':
    make_plaintext_files()
