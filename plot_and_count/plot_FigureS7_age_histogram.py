from pathlib import Path

import matplotlib.pyplot as plt

from utils import read_csv_fast

if __name__ == '__main__':
    fig = plt.figure(figsize=(6, 5))

    fp = f'../dataframes/df_combined_semi_pruned_Jan21.csv'
    df = read_csv_fast(fp)
    key = 'age_last'
    df['age_dif'] = df['year'] - df['age_first']
    age_low = 1960
    age_high = df[key].max()
    if key != 'age_dif': assert age_high == 2024
    bins = int(age_high - age_low)
    plt.hist(df[key], range=(age_low, age_high), bins=bins + 1,
             color='dodgerblue')
    plt.xlim(age_low, age_high)
    plt.gca().spines[['right', 'top', ]].set_visible(False)
    plt.grid(color='lightgray', linestyle='-', linewidth=0.5,
             alpha=0.5, zorder=-2)

    plt.ylabel('Number of papers', fontsize=13)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlabel('Year of paper\'s last author\'s'
               '\nfirst last author paper', fontsize=13)
    plt.gca().set_facecolor('whitesmoke')

    plt.subplots_adjust(left=0.15,
                        bottom=0.15,
                        right=0.985,
                        top=0.925,
                        wspace=0.24,
                        hspace=.36
                        )

    Path(r'../figs_and_tables').mkdir(parents=True, exist_ok=True)
    plt.savefig('../figs_and_tables/Figure_S7_age_histogram.png', dpi=600)
    plt.show()
