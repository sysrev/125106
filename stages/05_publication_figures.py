import matplotlib.pyplot as plt, seaborn as sns, pandas as pd, os, sqlite3, json

metrics = pd.read_csv('cache/02_analyze/metrics.csv')

#%% AGGREGATE PLOT ====================================================================
correct_cols = ['TP', 'TN', 'FP', 'FN', 'P', 'N']
tot_metrics = metrics.groupby('short_label')[correct_cols].sum()
tot_metrics['Sensitivity'] = tot_metrics['TP'] / tot_metrics['P']
tot_metrics['Specificity'] = tot_metrics['TN'] / tot_metrics['N']
tot_metrics['Accuracy'] = (tot_metrics['TP'] + tot_metrics['TN']) / (tot_metrics['P'] + tot_metrics['N'])
tot_metrics['Balanced Accuracy'] = (tot_metrics['Sensitivity'] + tot_metrics['Specificity']) / 2
tot_metrics = tot_metrics.sort_values('Balanced Accuracy', ascending=False)
tot_metrics = tot_metrics[['Sensitivity', 'Specificity', 'Accuracy', 'Balanced Accuracy','TP', 'TN', 'FP', 'FN', 'P', 'N']]
tot_metrics.to_csv('cache/05_build_figures/tot_metrics.csv', index=True, sep="\t")

colors = {
    "Location": "#377538",
    "Disease": "#99C9EC",
    "Weather Variable": "#D9CD7F",
    "Weather Disease Impact": "#9D4C97",
    "Include": "#322185",
    "Measures of Disease": "#BE6E78"
}

# Build a damn awesome plot for the balanced accuracy
df_sorted = tot_metrics.sort_values(by="Balanced Accuracy", ascending=False)
plt.figure(figsize=(7, 10))
barplot = sns.barplot(
    x='Balanced Accuracy', y='short_label', data=df_sorted,
    palette=colors.values()  # Use the first N colors from the solarized palette
)

plt.title('Balanced Accuracy by Label')
plt.xlabel('Balanced Accuracy')
plt.ylabel('')
plt.grid(True, linestyle='--', alpha=0.6)  # Add grid lines for better readability
plt.gca().set_facecolor((1, 1, 1, 0))  # Transparent background on the plot area
plt.gcf().set_facecolor((1, 1, 1, 0))  # Transparent background on the figure
plt.tight_layout()
plt.savefig('balanced_accuracy.png')

#%% CATEGORY SPECIFIC PLOT =================================================================
# sort metrics in order of aggregated balanced accuracy and then by balanced accuracy
pdf = metrics.merge(tot_metrics[['Balanced Accuracy']], on='short_label', suffixes=('', '_aggregate'))
pdf = pdf.sort_values(by=['Balanced Accuracy_aggregate','Balanced Accuracy'], ascending=False)
pdf['colors'] = pdf['short_label'].map(colors)
pdf['label'] = pdf['short_label'] + ' = ' + pdf['answer'].astype(str)

os.makedirs('cache/05_build_figures', exist_ok=True)
pdf[['short_label', 'answer', 'Sensitivity', 'Specificity', 'Accuracy', 'Balanced Accuracy','TP', 'TN', 'FP', 'FN', 'P', 'N','Articles']].to_csv('cache/05_build_figures/by_category_metrics.csv', index=False, sep="\t")

plt.figure(figsize=(7, 10))
colors = list(pdf['colors'])

barplot = sns.barplot(x='Balanced Accuracy', y='label', data=pdf, palette=colors)

# Iterate over the patches to set the zorder manually
for patch in barplot.patches:
    patch.set_zorder(3)

plt.title('Balanced Accuracy by Label')
plt.xlabel('Balanced Accuracy')
plt.ylabel('')

# Set the grid with zorder just below the bars
plt.grid(True, which='major', linestyle='--', linewidth=1, color='black', zorder=2)

plt.gca().set_facecolor((1, 1, 1, 0))  # Set the plot background to be transparent
plt.gcf().set_facecolor((1, 1, 1, 0))  # Set the figure background to be transparent

plt.tight_layout()

plt.savefig('balanced_accuracy_2.png', bbox_inches='tight', pad_inches=0.1, transparent=True)

#%% Disease Distributions ================================================================================
conn = sqlite3.connect('.sr/sr.sqlite')
labels = pd.read_sql('SELECT * FROM labels', conn)

autolabel = pd.read_sql('SELECT * FROM auto_labels', conn)
autolabel = autolabel[['article_id', 'label_id', 'answer']]
autolabel = autolabel.merge(labels, left_on='label_id', right_on='label_id')
autolabel = autolabel[['article_id', 'short_label', 'answer']]
autolabel['answer'] = autolabel['answer'].apply(json.loads)
autolabel = autolabel.explode('answer')

# filter out null or na predictions
autolabel = autolabel[~autolabel['answer'].isna()]

# get included articles, and disease + weather variable ~ location
included_articles = autolabel[autolabel['short_label'] == 'Include'][autolabel['answer'] == True]['article_id']
plotdf = autolabel[autolabel['article_id'].isin(included_articles)]

# get locations as a column and remove locations with less than 10 articles
location = plotdf[plotdf['short_label'] == 'Location'][['article_id', 'answer']].rename(columns={'answer': 'location'})
location = location[location['location'] != 'Unknown']
location['count'] = location.groupby('location')['location'].transform('count')
location = location[location['count'] > 10]

plotdf = plotdf.merge(location, on='article_id', how='inner')

def mkplot(pdf, variable, output_file):
    pdf = pdf[pdf['short_label'].isin([variable])]
    pdf = pdf[pdf['short_label'] != 'Location'].groupby(['location','short_label','answer','count']).count().reset_index()
    pdf = pdf[['location','short_label','answer','article_id','count']].rename(columns={'short_label':'label','article_id': 'ans_count'})
    pdf['proportion'] = pdf['ans_count'] / pdf['count']

    # Ensuring all values are shown in each location, including those with zero count
    all_values = pdf['answer'].unique()
    all_locations = pdf['location'].unique()
    pdf = pdf.set_index(['location', 'answer']).reindex(pd.MultiIndex.from_product([all_locations, all_values], names=['location', 'answer']), fill_value=0).reset_index()
    
    # Sorting locations by the total number of articles
    location_order = pdf[['location','count']].drop_duplicates().groupby('location')['count'].sum().sort_values(ascending=False).index

    # Prepare to plot with adjusted figure size for PowerPoint slide
    plt.figure(figsize=(12, 7.5))
    g = sns.FacetGrid(pdf, col="location", col_wrap=4, sharex=False, sharey=True, height=4, col_order=location_order)
    g.map_dataframe(sns.barplot, x="answer", y="proportion", ci=None, palette='viridis')

    # Adjusting titles to include the total articles
    for ax, location in zip(g.axes.flat, location_order):
        total_articles = max(pdf[pdf['location'] == location]['count'])
        ax.set_title(f"{location} ({total_articles} articles)")

    # Set axis labels and layout
    g.set_axis_labels("", "Proportion of Articles")
    g.set_xticklabels(rotation=45, fontsize=8)
    
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_file, format='png', dpi=300)

mkplot(plotdf, 'Disease', 'disease_distribution.png')
mkplot(plotdf, 'Weather Variable', 'weather_variable_distribution.png')



