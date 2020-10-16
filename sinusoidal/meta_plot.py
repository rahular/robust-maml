import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks", font="Times New Roman")

df_uniform = pd.read_pickle('results/bayes_uniform_discrete/res.pkl')
df_skewed = pd.read_pickle('results/bayes_skewed_discrete/res.pkl')
df_uniform['Task Distribution'] = 'Identical'
df_skewed['Task Distribution'] = 'Skewed'

df_mm_uniform = pd.read_pickle('results/minimax_uniform_newsoftmax/res.pkl')
df_mm_skewed = pd.read_pickle('results/minimax_skewed_newsoftmax/res.pkl')
df_mm_uniform['Task Distribution'] = 'Identical'
df_mm_skewed['Task Distribution'] = 'Skewed'

df_uniform['Detector'] = 'Bayes'
df_skewed['Detector'] = 'Bayes'
df_mm_uniform['Detector'] = 'Minimax'
df_mm_skewed['Detector'] = 'Minimax'

df = pd.concat([df_uniform, df_skewed, df_mm_uniform, df_mm_skewed], ignore_index=True, sort=False)
df.rename(columns = {'k_shot':'K-Shot'}, inplace = True)
# df = df.loc[df['K-Shot'] == 20]
df.dropna()
plot = sns.relplot(data=df, x="grad_steps", y="loss", hue="Detector", style="Task Distribution", col="K-Shot", kind="line", col_wrap=2)
sns.despine()
plot.set(xlabel='Gradient Steps', ylabel='Mean Squared Error')
#plt.legend(title='K')
plot.savefig("meta-plot.pdf", bbox_inches='tight', pad_inches=0)
