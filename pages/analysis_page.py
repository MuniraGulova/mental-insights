import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from navigation import make_sidebar

make_sidebar()
st.title('Data Analysis Page')

data = pd.read_csv('data/data.csv')
df = data.copy()
df['mental_health'] = df['mental_health'].apply(lambda x: 1 if x == "Yes" else 0)
df = df[df['gender'] != 'Other']

sample_1 = df[df['mental_health'] == 1].sample(n=3, random_state=42)
sample_0 = df[df['mental_health'] == 0].sample(n=2, random_state=42)
final_sample = pd.concat([sample_1, sample_0])
df = df.drop(final_sample.index)

df_mod = final_sample.copy()

y = df['mental_health']
X = df.drop(['mental_health'], axis=1)



with st.expander('📂 Data', expanded=False):
    st.write('Target: mental_health')
    st.dataframe(df)

column_descriptions = {
    'tech_company': 'Работает ли человек в техкомпании (да/нет)',
    'benefits': 'Есть ли льготы/бенефиты по ментальному здоровью (да/нет)',
    'workplace_resources': 'Есть ли ресурсы для поддержки ментального здоровья (да/нет)',
    'mh_employer_discussion': 'Можно ли обсуждать ментальное здоровье с работодателем (да/нет)',
    'mh_coworker_discussion': 'Можно ли обсуждать ментальное здоровье с коллегами (да/нет)',
    'medical_coverage': 'Покрывает ли страховка ментальное здоровье (да/нет)',
    'mental_health': 'Есть ли проблемы с ментальным здоровьем (да/нет) — **таргет**',
    'mh_share': 'Готовность делиться проблемами ментального здоровья (да/нет)',
    'age': 'Возраст (числовой признак)',
    'gender': 'Пол (категориальный признак)',
    'country': 'Страна (категориальный признак)'
}

st.write('---')
st.markdown("<h3 style='text-align: center;'>📊 Analysis of feature distribution</h3>", unsafe_allow_html=True)
st.write('---')
for col in df.columns:
    col1, col2 = st.columns([2, 1])

    with col1:
        plt.figure(figsize=(5, 3))
        sns.countplot(x=df[col], order=df[col].value_counts().index)
        plt.title(f'Distribution of {col}')
        plt.yticks(rotation=45)
        plt.xticks(fontsize=7, rotation=90)
        st.pyplot(fig=plt.gcf())

    with col2:
        description = column_descriptions.get(col)
        st.write(f"#### {col}")
        st.write(description)

st.write('---')
st.markdown("<h3 style='text-align: center;'>💬 Mental Health Discussion at Work & Gender Proportions</h3>",
            unsafe_allow_html=True)
st.write('---')
col1, col2 = st.columns(2)

with col1:
    st.write("##### How easy is discussing Mental Health at Work?")
    plt.figure(figsize=(4, 4))
    df['mh_employer_discussion'].value_counts().plot(
        kind='pie', autopct='%1.1f%%', colors=['cyan', 'lightblue'],
        wedgeprops={'edgecolor': 'white'}, textprops={'fontsize': 8}
    )
    st.pyplot(plt)

with col2:
    st.write("##### Proportions of Gender in Tech Industry")
    plt.figure(figsize=(4, 4))
    df['gender'].value_counts().plot(
        kind='pie', autopct='%1.1f%%', colors=['cyan', 'lightblue'],
        wedgeprops={'edgecolor': 'white'}, textprops={'fontsize': 8}
    )
    st.pyplot(plt)

st.header('📌 Relationship of features with the target variable')

columns = list(X.columns)

for i in range(0, len(columns), 2):
    col1, col2 = st.columns(2)

    with col1:
        plt.figure(figsize=(5, 3))
        sns.countplot(x=X[columns[i]], hue=y, order=X[columns[i]].value_counts().index)
        plt.title(f'{columns[i]} vs Target')
        plt.xticks(fontsize=7, rotation=90)
        st.pyplot(plt)

    if i + 1 < len(columns):
        with col2:
            plt.figure(figsize=(5, 3))
            sns.countplot(x=X[columns[i + 1]], hue=y, order=X[columns[i + 1]].value_counts().index)
            plt.title(f'{columns[i + 1]} vs Target')
            plt.xticks(fontsize=7, rotation=90)
            st.pyplot(plt)


def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k - 1, r - 1))


corr_matrix = pd.DataFrame(index=df.columns, columns=df.columns)

for col1 in df.columns:
    for col2 in df.columns:
        corr_matrix.loc[col1, col2] = cramers_v(df[col1], df[col2])

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix.astype(float), annot=True, cmap="coolwarm")
plt.title("Cramér's V Correlation Heatmap")
st.pyplot(plt)

print(df_mod)
