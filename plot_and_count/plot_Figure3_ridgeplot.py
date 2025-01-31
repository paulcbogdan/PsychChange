from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colors as mcolors, ticker as mtick
from matplotlib.colors import to_rgb

from utils import read_csv_fast

pd.options.mode.chained_assignment = None


def custom_base_palette(base_color, n_colors=10):
    base_rgb = np.array(to_rgb(base_color))

    # Create a range of intensities for interpolation
    print(f'{base_color=}')
    if base_color == (1.0, 0.4980392156862745, 0.054901960784313725):
        intensities = np.linspace(0.8, 1.2, n_colors)
    else:
        intensities = np.linspace(0.8, 1.2, n_colors)

    # Generate the palette by scaling the base color
    palette = [tuple(base_rgb * intensity) for intensity in intensities]
    return palette


def plot_ridges():
    fp = fr'../dataframes/df_combined_semi_pruned_Jan21.csv'
    df = read_csv_fast(fp, easy_override=False)
    subjects = ['General_Psychology',
                'Experimental_and_Cognitive_Psychology',
                'Developmental_and_Educational_Psychology',
                'Social_Psychology',
                'Clinical_Psychology',
                'Applied_Psychology']
    colors = mcolors.TABLEAU_COLORS
    for subject, color in zip(subjects, colors):
        if 'Misc' in subject: continue  # skip miscellaneous, few papers
        plot_ridge(df, 'p_fragile', subject, color)


def plot_ridge(df, key, subject, cname, by2=True):
    import seaborn as sns
    import matplotlib.pyplot as plt
    if by2:
        df['year'] = df['year'].apply(lambda x: x - 1 if x % 2 == 1 else x)

    df_s = df[df[subject] == True]
    df_s = df_s[df_s['num_ps'] > 4]
    df = df_s[['year', key]]
    df['x'] = df[key]
    df['g'] = df['year'].astype(int)

    sns.set_theme(style="white", rc={"axes.facecolor":
                                         (0, 0, 0, 0)})

    # Initialize the FacetGrid object
    color = mcolors.to_rgb(cname)
    pal = custom_base_palette(color, n_colors=21)

    g = sns.FacetGrid(df, row="g", hue="g",
                      aspect=4.5 / 1.2, height=.5 * 1.2,
                      palette=pal
                      )

    years = sorted(df['year'].unique().tolist())

    # Draw the densities in a few steps
    g.map(sns.kdeplot, "x",
          bw_adjust=.5, clip_on=False,
          fill=True, alpha=1, linewidth=1.5,
          clip=(0, 1),
          )

    g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw_adjust=.5)

    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        center = x.mean()
        label = f'{label}-'
        label = label
        ax = plt.gca()
        height = .15
        ax.text(center, height, label, fontweight="bold", color='w',
                ha="center", va="center", transform=ax.transAxes,
                fontsize=15,
                # path_effects=[pe.withStroke(linewidth=1, foreground="k")]
                )

    g.map(label, "x")
    plt.gca().xaxis.set_major_formatter(
        mtick.StrMethodFormatter('{x:.0%}'))
    plt.tick_params(axis='x', which='both', bottom=True,
                    width=1)
    plt.xticks([0, .25, .5, .75, 1.0],
               ['    0%', '  25%', '50%', '75%  ', '100%    '],
               fontsize=12.5)
    plt.xlabel('')

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.45, left=.05, right=.91, top=.93)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    subject = subject.replace("_", " ")
    plt.suptitle(subject.replace('Psych', '\nPsych'),
                 y=0.97 if by2 else 1.01,
                 fontsize=16)
    if 'and' in subject:
        plt.suptitle(subject.replace('and', '&\n').replace('Psych', '\nPsych'),
                     y=.999 if by2 else 1.02, fontsize=16)
    plt.xlim(0, 1)
    for ax in g.axes.flat:
        ax.set_ylim(0, 3)
        ax.spines['bottom'].set_linewidth(1)

    g.set(yticks=[], ylabel="")
    g.despine(bottom=False, left=True)
    fp = fr'../figs_and_tables/Figure_3_ridgeplot/{subject}.png'
    Path(r'../figs_and_tables/Figure_3_ridgeplot').mkdir(parents=True, exist_ok=True)
    plt.savefig(fp, dpi=600)
    plt.show()


if __name__ == '__main__':
    plot_ridges()
