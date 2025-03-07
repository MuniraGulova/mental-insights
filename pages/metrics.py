import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
import seaborn as sns
from pages.models import X_test, X_train, y_test, y_train, models, models_dict
from navigation import make_sidebar

make_sidebar()

st.title('Model Evaluation Dashboard')

models = st.multiselect('Models',
                        ('LogisticRegression', 'DecisionTree', 'KNeighborsClassifier', 'RandomForest', 'Bagging'))

tab_metrics, tab_roc, tab_confusion = st.tabs(["📊 Метрики", "📈 ROC-AUC", "🔎 Confusion Matrix"])

roc_data = {}
metrics_data = []
with tab_metrics:
    for select_model in models:
        model = models_dict[select_model]
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_metrics = {
            'Model': select_model,
            'Accuracy (Train)': accuracy_score(y_train, y_train_pred),
            'Precision (Train)': precision_score(y_train, y_train_pred),
            'Recall (Train)': recall_score(y_train, y_train_pred),
            'F1 Score (Train)': f1_score(y_train, y_train_pred),
            'Accuracy (Test)': accuracy_score(y_test, y_test_pred),
            'Precision (Test)': precision_score(y_test, y_test_pred),
            'Recall (Test)': recall_score(y_test, y_test_pred),
            'F1 Score (Test)': f1_score(y_test, y_test_pred)
        }

        metrics_data.append(train_metrics)

        st.divider()

        # Сохраняем данные для ROC
        y_test_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_test_proba)
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        roc_data[select_model] = (fpr, tpr, roc_auc)

df_metrics = pd.DataFrame(metrics_data)
st.dataframe(df_metrics)

with tab_roc:
    st.write("### ROC Curves for All Models")
    plt.figure(figsize=(8, 6))

    for model_name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f'{model_name} (AUC={auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    st.pyplot(plt)

with tab_confusion:
    st.write("### Confusion Matrices")

    for select_model in models:
        st.subheader(f"🔎 Confusion Matrix for {select_model}")

        model = models_dict[select_model]
        model.fit(X_train, y_train)

        # Train CM
        y_train_pred = model.predict(X_train)
        cm_train = confusion_matrix(y_train, y_train_pred)

        st.write("#### Train Set")
        fig, ax = plt.subplots()
        sns.heatmap(cm_train, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
        plt.title(f'{select_model} - Train Set Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(fig)

        # Test CM
        y_test_pred = model.predict(X_test)
        cm_test = confusion_matrix(y_test, y_test_pred)

        st.write("#### Test Set")
        fig, ax = plt.subplots()
        sns.heatmap(cm_test, annot=True, fmt="d", cmap="Oranges", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
        plt.title(f'{select_model} - Test Set Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(fig)


