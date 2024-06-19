import pandas as pd, sqlite3, json
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score

conn = sqlite3.connect('.sr/sr.sqlite')

labels = pd.read_sql('SELECT label_id, short_label FROM labels WHERE enabled = 1', conn)

# automatic answers
autolabel = pd.read_sql('SELECT * FROM auto_labels', conn)
autolabel = autolabel[['article_id', 'label_id', 'answer']]
autolabel = autolabel.merge(labels, left_on='label_id', right_on='label_id')
autolabel = autolabel[['article_id', 'short_label', 'answer']]
autolabel['answer'] = autolabel['answer'].apply(json.loads)
autolabel = autolabel.explode('answer')
autolabel['prediction'] = autolabel['answer']

# user answers
completions = pd.read_sql('SELECT * FROM article_label', conn)
completions = completions[completions['user_id'].isin([13580,13517])]
completions = completions[['article_id', 'label_id', 'user_id', 'answer']]
completions = completions.merge(labels, left_on='label_id', right_on='label_id')
completions = completions[['article_id', 'user_id', 'short_label', 'answer']]
completions['answer'] = completions['answer'].apply(json.loads)
completions = completions.explode('answer')
completions['tot_users'] = completions.groupby(['article_id', 'short_label'])['user_id'].transform('nunique')
completions['ans_users'] = completions.groupby(['article_id', 'short_label', 'answer'])['user_id'].transform('nunique')
completions = completions[completions['ans_users'] == completions['tot_users']]
completions.drop(columns=['tot_users', 'ans_users','user_id'], inplace=True)
completions = completions.drop_duplicates()

# filter to article_id + short_label in autolabel
art_lbl = autolabel[['article_id', 'short_label']].drop_duplicates().set_index(['article_id', 'short_label'])
completions = completions.join(art_lbl, on=['article_id', 'short_label'], how='inner')
completions['user_answer'] = completions['answer']

# find all combinations of unique vlues of article_id, short_label, answer
lbl_answers = completions[['short_label', 'answer']].drop_duplicates().reset_index(drop=True)
lbl_answers = completions[['article_id']].merge(lbl_answers, how='cross')
art_lbl_ans = lbl_answers.merge(completions, on=['article_id','short_label','answer'], how='left')
art_lbl_ans = art_lbl_ans[['article_id', 'short_label', 'answer', 'user_answer']]
art_lbl_ans = art_lbl_ans.merge(autolabel, on=['article_id', 'short_label', 'answer'], how='left')

# user_answer = 1 if user_answer == answer else 0
art_lbl_ans['user_answer'] = (art_lbl_ans['user_answer'] == art_lbl_ans['answer']).astype(int)
art_lbl_ans['prediction'] = (art_lbl_ans['prediction'] == art_lbl_ans['answer']).astype(int)

df = art_lbl_ans

# remove answer == False when short_label **only** has answers True and False, booleans have simpler metrics
labels_with_only_true_false = df.groupby('short_label')['answer'].apply(lambda x: set(x) == {True, False})
labels_with_only_true_false = labels_with_only_true_false[labels_with_only_true_false].index.tolist()
df = df[~((df['short_label'].isin(labels_with_only_true_false)) & (df['answer'] == False))]

# write df to csv
df.to_csv('cache/02_analyze/evaluate.csv', index=False)

def calc_metrics(x):
    # Calculate the confusion matrix
    cm = confusion_matrix(x['user_answer'], x['prediction'], labels=[1, 0])
    TP, FN, FP, TN = cm.ravel()
    
    # Calculate metrics based on the confusion matrix
    Sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
    Specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    Accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0
    Balanced_Accuracy = (Sensitivity + Specificity) / 2
    articles = x['article_id'].nunique()
    
    # Calculate totals for positives and negatives
    Total_Positives = TP + FN  # Total actual positives
    Total_Negatives = TN + FP  # Total actual negatives

    return pd.Series({
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN, 'P': Total_Positives, 'N': Total_Negatives,
        'Articles': articles,  # Number of articles with this label
        'Sensitivity': Sensitivity, 'Specificity': Specificity, 
        'Accuracy': Accuracy, 'Balanced Accuracy': Balanced_Accuracy, 
    })

# Group by 'short_label' and 'answer' and apply the function
metrics = df.groupby(['short_label', 'answer']).apply(calc_metrics).reset_index()
metrics.to_csv('cache/02_analyze/metrics.csv', index=False)