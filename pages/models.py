import streamlit as st
import pandas as pd
from pages.analysis_page import df_mod, X, y 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, TargetEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from navigation import make_sidebar

make_sidebar()


st.title("🧠 Workplace Mental Health Survey")

st.write("Explore how workplace factors and personal demographics relate to mental health at work.")
st.dataframe(df_mod)

st.header("👥 Demographic Information")

selected_gender = st.selectbox(
    "Select Your Gender",
    X['gender'].unique()
)

age = st.number_input(
    'Enter Your Age',
    min_value=int(X['age'].min()),
    max_value=int(X['age'].max()),
    value=int(X['age'].mean())
)

country = st.selectbox(
    "🌍 Select Your Country",
    X['country'].unique()
)

st.header("🏢 Workplace Environment")

tech_company = st.selectbox(
    "Do you work for a technology company?",
    X['tech_company'].unique(), help='Можете ли вы обсудить ментальное здоровье с коллегами?'
)

mh_coworker_discussion = st.selectbox(
    "Are you comfortable discussing mental health issues with your coworkers?",
    X['mh_coworker_discussion'].unique(), help='Можете ли вы обсудить ментальное здоровье с коллегами?'
)

medical_coverage = st.selectbox(
    "Does your employer provide mental health coverage?",
    X['medical_coverage'].unique(), help='Предоставляет ли работодатель медицинское покрытие для ментального здоровья?'
)

benefits = st.selectbox(
    "Do you feel your employer offers sufficient mental health benefits?",
    X['benefits'].unique(), help='Оцениваете ли вы льготы по ментальному здоровью как достаточные?'
)

workplace_resources = st.selectbox(
    "Do you feel there are enough mental health resources available in your workplace?",
    X['workplace_resources'].unique(),
    help='Оцениваете ли вы доступные ресурсы для психического здоровья на рабочем месте как достаточные?'
)

mh_employer_discussion = st.selectbox(
    "Would you feel comfortable discussing a mental health issue with your employer?",
    X['mh_employer_discussion'].unique(),
    help='Комфортно ли вам обсуждать проблемы психического здоровья с вашим работодателем?'
)

st.header("💬 Personal Comfort Sharing")

mh_share = st.slider(
    'How comfortable are you sharing your mental health status at work? (0 = Not comfortable at all, 10 = Very comfortable)',
    0, 10, 5, help='Готовность делиться своим состоянием на работе (шкала 0-10)'
)
st.write('---')

# =================================

choice = pd.DataFrame({'tech_company': tech_company, 'benefits': benefits, 'workplace_resources': workplace_resources,
                       'mh_employer_discussion': mh_employer_discussion,
                       'mh_coworker_discussion': mh_coworker_discussion, 'medical_coverage': medical_coverage,
                       'mh_share': mh_share, 'age': age, 'gender': selected_gender, 'country': country}, index=[0])
st.dataframe(choice)
st.write('---')
st.markdown("<h2 style='text-align: center;'> 🍂 Prediction  </h2>", unsafe_allow_html=True)
st.write('---')

models = st.multiselect('Models',
                        ('LogisticRegression', 'DecisionTree', 'KNeighborsClassifier', 'RandomForest', 'Bagging'))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.copy()
X_test = X_test.copy()

choice = choice.copy()

column_list_OneHotEnc = ['benefits', 'workplace_resources', 'country']
column_list_label = ['mh_employer_discussion', 'mh_coworker_discussion', 'medical_coverage', 'tech_company', 'gender']
num_columns = ['age', 'mh_share']

label_encoders = {}

for col in column_list_label:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])
    choice[col] = le.transform(choice[col])
    label_encoders[col] = le

OneHotEnc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_train_onehot = OneHotEnc.fit_transform(X_train[column_list_OneHotEnc])

col_name = OneHotEnc.get_feature_names_out(column_list_OneHotEnc)

df_one = pd.DataFrame(X_train_onehot, columns=col_name, index=X_train.index)
X_train = pd.concat([df_one, X_train.drop(columns=column_list_OneHotEnc)], axis=1)

X_test_onehot = OneHotEnc.transform(X_test[column_list_OneHotEnc])
df_test_onehot = pd.DataFrame(X_test_onehot, columns=col_name, index=X_test.index)
X_test = pd.concat([df_test_onehot, X_test.drop(columns=column_list_OneHotEnc)], axis=1)

choice_onehot = OneHotEnc.transform(choice[column_list_OneHotEnc])
df_choice_onehot = pd.DataFrame(choice_onehot, columns=col_name, index=choice.index)
choice = pd.concat([df_choice_onehot, choice.drop(columns=column_list_OneHotEnc)], axis=1)

scaler = StandardScaler()

X_train[num_columns] = scaler.fit_transform(X_train[num_columns])
X_test[num_columns] = scaler.transform(X_test[num_columns])
choice[num_columns] = scaler.transform(choice[num_columns])

with st.expander('Input features after encoding and scaling'):
    st.dataframe(choice)

logreg_bg = LogisticRegression(C=1, penalty='l2', solver='liblinear')

models_dict = {
    'LogisticRegression': LogisticRegression(C=1, penalty='l2', solver='liblinear', random_state=42),
    'KNeighborsClassifier': KNeighborsClassifier(metric='manhattan', n_neighbors=19, weights='uniform'),
    'DecisionTree': DecisionTreeClassifier(criterion='entropy', random_state=42, max_depth=5, max_features='log2',
                                           min_samples_split=5, min_samples_leaf=2),
    'RandomForest': RandomForestClassifier(max_depth=5, max_features=0.3, min_samples_leaf=10, min_samples_split=5,
                                           n_estimators=100, random_state=42),
    'Bagging': BaggingClassifier(logreg_bg, random_state=42, n_jobs=-1, bootstrap=True, max_features=0.8,
                                 max_samples=0.8,
                                 n_estimators=100)
}

yes = 0
count = 0
i = 0
k = 0
for select_model in models:
    st.subheader(f'Prediction for {select_model}')

    model = models_dict[select_model]
    model = model.fit(X_train, y_train)

    prediction = model.predict(choice)
    pred_proba = model.predict_proba(choice)

    df_prediction_proba = pd.DataFrame(pred_proba, columns=['No', 'Yes'])

    st.dataframe(
        df_prediction_proba,
        column_config={
            'No': st.column_config.ProgressColumn(
                'No',
                format='%.2f',
                width='medium',
                min_value=0,
                max_value=1,
            ),
            'Yes': st.column_config.ProgressColumn(
                'Yes',
                format='%.2f',
                width='medium',
                min_value=0,
                max_value=1,
            )
        },
        hide_index=True
    )

    result_label = 'Yes' if prediction == 1 else 'No'
    count = i + 1

    if result_label == 'Yes':
        st.success(f"✅ Predicted category: **{result_label}**")
        yes = k + 1
    else:
        st.markdown(f'<p style="color:red; font-weight:bold; font-size:18px;">❌ Predicted category: {result_label}</p>',
                    unsafe_allow_html=True)

if yes > (count / 2):
    st.image("image/mental.jpg", width=200)
