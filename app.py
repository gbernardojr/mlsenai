import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score, roc_curve, auc
)
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Demo ML - Random Forest", layout="wide")
st.title("🧠 Demonstração Interativa de Machine Learning")
st.markdown("Carregue um CSV, escolha o target e veja o modelo treinar + gráficos + previsões")

# ─── Sessão para manter estado ───────────────────────────────────────
if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.is_classification = None
    st.session_state.label_encoder = None
    st.session_state.features = None
    st.session_state.X_columns = None

# ─── 1. Upload do arquivo CSV ────────────────────────────────────────
st.header("1. Carregar os dados")
uploaded_file = st.file_uploader("Escolha o arquivo .csv", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Parse dteday and extract features
        if 'dteday' in df.columns:
            df['dteday'] = pd.to_datetime(df['dteday'], errors='coerce')
            df['dteday_year'] = df['dteday'].dt.year
            df['dteday_month'] = df['dteday'].dt.month
            df['dteday_day'] = df['dteday'].dt.day
            df['dteday_dayofweek'] = df['dteday'].dt.dayofweek
            df['dteday_sin_month'] = np.sin(2 * np.pi * df['dteday_month'] / 12)
            df['dteday_cos_month'] = np.cos(2 * np.pi * df['dteday_month'] / 12)
            st.info("✅ dteday parsed and features extracted: dteday_year, _month, _dayofweek, sin/cos_month")
        
        # Remove useless 'instant' column (index)
        if 'instant' in df.columns:
            df = df.drop(columns=['instant'])
            st.info("✅ Coluna 'instant' removida (sem valor preditivo)")
        
        st.success(f"✅ Dataset carregado: {df.shape[0]} linhas × {df.shape[1]} colunas")


        
        with st.expander("Visualizar dados (primeiras linhas)"):
            st.dataframe(df.head(8))
        
        # ─── 2. Escolher coluna target ────────────────────────────────
        st.header("2. Escolha a coluna alvo (rótulo)")
        possible_targets = df.columns.tolist()
        target_col = st.selectbox("Coluna Target", possible_targets)
        
        if target_col:
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            # Filtrar apenas colunas numéricas como features (simplificação), EXCLUINDO casual/registered (leakage)
            numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
            exclude_cols = ['casual', 'registered']
            numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
            if not numeric_cols:
                st.error("❌ Não foram encontradas colunas numéricas para usar como features.")
                st.stop()
            
            X = X[numeric_cols]
            st.info(f"Usando {len(numeric_cols)} features numéricas (sem casual/registered): {', '.join(numeric_cols)}")
            
            # ─── Detectar tipo de problema ────────────────────────────
            is_classification = True
            if pd.api.types.is_numeric_dtype(y):
                unique_ratio = y.nunique() / len(y)
                if unique_ratio > 0.05:
                    is_classification = False
            
            problem_type = "Classificação" if is_classification else "Regressão"
            st.subheader(f"Tipo de problema detectado: **{problem_type}**")
            
            # ─── Preparar y ───────────────────────────────────────────
            le = None
            if is_classification and y.dtype == object:
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
            else:
                y_encoded = y
            
            # ─── Botão Treinar ────────────────────────────────────────
            if st.button(f"🚀 Treinar Random Forest ({problem_type})", type="primary"):
                with st.spinner("Treinando modelo..."):
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y_encoded if is_classification else y,
                        test_size=0.20, random_state=42
                    )
                    
                    if is_classification:
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                    else:
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                    
                    model.fit(X_train, y_train)
                    
                    # Salvar no session state
                    st.session_state.model = model
                    st.session_state.is_classification = is_classification
                    st.session_state.label_encoder = le
                    st.session_state.features = X.columns.tolist()
                    st.session_state.X_columns = X.columns
                    st.session_state.X_stats = X.describe()
                    
                    # ─── Avaliação ────────────────────────────────────────
                    y_pred = model.predict(X_test)
                    
                    st.success("Modelo treinado com sucesso!")
                    
                    st.header("3. Resultados da Validação (20% teste)")
                    
                    col1, col2 = st.columns(2)
                    

                    with col1:
                        if is_classification:
                            acc = accuracy_score(y_test, y_pred)
                            st.metric("Acurácia", f"{acc:.3%}")
                            with st.expander("📊 O que significa Acurácia?"):
                                st.markdown("""
                                **Cálculo**: % de previsões corretas = (TP + TN) / Total  
                                **Interpretação**: 
                                - >90%: Excelente
                                - 80-90%: Bom  
                                - 70-80%: Razoável
                                - <70%: Precisa melhorias
                                **Limitação**: Não funciona bem com datasets desbalanceados
                                """)
                            st.text("Relatório de Classificação")
                            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                            st.dataframe(pd.DataFrame(report).T)
                            with st.expander("📊 Precision, Recall e F1-Score"):
                                st.markdown("""
                                **Precision** = TP / (TP + FP): % de "sim" que realmente eram "sim"
                                **Recall** = TP / (TP + FN): % de "sim" reais que foram encontrados  
                                **F1** = 2 * (Precision * Recall) / (Precision + Recall): Média harmônica
                                **Macro avg**: Média simples entre classes
                                **Weighted avg**: Média ponderada pelo suporte de cada classe
                                """)
                        else:
                            mse = mean_squared_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            st.metric("MSE", f"{mse:.4f}")
                            with st.expander("📊 O que significa MSE?"):
                                st.markdown("""
                                **Cálculo**: Média((y_real - y_pred)²)  
                                **Interpretação**: Erro quadrático médio (menor = melhor)
                                Penaliza mais erros grandes que erros pequenos
                                """)
                            st.metric("R²", f"{r2:.4f}")
                            with st.expander("📊 O que significa R²?"):
                                st.markdown("""
                                **Cálculo**: 1 - (SS_res / SS_tot)  
                                **Interpretação**: % da variância explicada pelo modelo
                                - 1.0: previsão perfeita
                                - 0.8-0.9: excelente
                                - 0.6-0.8: bom
                                - <0.5: fraco
                                - <0: pior que média simples
                                """)

                    
                    with col2:
                        # Gráfico de importância de features
                        importances = model.feature_importances_
                        indices = np.argsort(importances)[::-1]
                        fig, ax = plt.subplots(figsize=(6, 5))
                        ax.barh(range(len(indices)), importances[indices], align="center")
                        ax.set_yticks(range(len(indices)))
                        ax.set_yticklabels([X.columns[i] for i in indices])
                        ax.set_title("Importância das Features")
                        st.pyplot(fig)
                    
                    # Gráficos adicionais
                    st.subheader("Visualizações")
                    tab1, tab2 = st.tabs(["Matriz de Confusão / Pred vs Real", "Outros"])
                    
                    with tab1:
                        if is_classification:
                            cm = confusion_matrix(y_test, y_pred)
                            fig_cm, ax_cm = plt.subplots(figsize=(7, 6))
                            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm,
                                        xticklabels=le.classes_ if le else None,
                                        yticklabels=le.classes_ if le else None)
                            ax_cm.set_xlabel("Previsão")
                            ax_cm.set_ylabel("Real")
                            ax_cm.set_title("Matriz de Confusão")
                            st.pyplot(fig_cm)
                            
                            # Curva ROC se binário
                            if len(np.unique(y)) == 2:
                                y_prob = model.predict_proba(X_test)[:, 1]
                                fpr, tpr, _ = roc_curve(y_test, y_prob)
                                roc_auc = auc(fpr, tpr)
                                fig_roc, ax_roc = plt.subplots()
                                ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
                                ax_roc.plot([0,1],[0,1],'--')
                                ax_roc.legend()
                                ax_roc.set_title("Curva ROC")
                                st.pyplot(fig_roc)
                        else:
                            fig_scatter, ax_scatter = plt.subplots()
                            ax_scatter.scatter(y_test, y_pred, alpha=0.6)
                            ax_scatter.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
                            ax_scatter.set_xlabel("Valor Real")
                            ax_scatter.set_ylabel("Valor Previsto")
                            ax_scatter.set_title("Previsão × Real")
                            st.pyplot(fig_scatter)
                    
                    # Salvar modelo
                    joblib.dump(model, "modelo_rf.pkl")
                    joblib.dump(X.columns.tolist(), "features.pkl")
                    if le:
                        joblib.dump(le, "label_encoder.pkl")
                    st.info("Modelo salvo como `modelo_rf.pkl`")

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")

# ─── 4. Previsão Interativa ───────────────────────────────────────────
st.header("4. Previsão Interativa")
if st.session_state.model is not None:
    import colunas_descricao
    import joblib
    expected_features = joblib.load("features.pkl")
    st.info(f"Modelo treinado com features: {', '.join(expected_features)}")
    
    # Date input for future prediction
    prediction_date = st.date_input("📅 Data da previsão (futura)", value=pd.Timestamp.now().date())
    
    if "X_stats" in st.session_state:
        stats = st.session_state.X_stats
        stats_min = stats.loc['min']
        stats_max = stats.loc['max']
    else:
        stats_min = pd.Series([0.0] * len(expected_features), index=expected_features)
        stats_max = pd.Series([1.0] * len(expected_features), index=expected_features)
    
    # Initialize new_data for ALL expected features
    new_data = {col: 0.0 for col in expected_features}
    
    # Auto-fill dteday features from prediction date
    new_data['dteday_year'] = prediction_date.year
    new_data['dteday_month'] = prediction_date.month
    new_data['dteday_day'] = prediction_date.day
    new_data['dteday_dayofweek'] = prediction_date.weekday()
    new_data['dteday_sin_month'] = np.sin(2 * np.pi * new_data['dteday_month'] / 12)
    new_data['dteday_cos_month'] = np.cos(2 * np.pi * new_data['dteday_month'] / 12)
    
    X_data = None  # Para unique_vals
    if 'X' in locals():
        X_data = X
    
    # Inputs for NON-dteday features
    for col in expected_features:
        if col.startswith('dteday_'):
            continue  # Already set
            
        descricao = colunas_descricao.COLUNAS_DESCRICAO.get(col, "")
        label = f"{col}: {descricao}"
        min_val = stats_min[col]
        max_val = stats_max[col]
        
        # Categorical cols: use selectbox
        if col in ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']:
            unique_vals = sorted(X_data[col].unique()) if X_data is not None and col in X_data.columns else [0,1,2,3,4]
            default_idx = unique_vals.index(int((min_val + max_val)/2)) if len(unique_vals)>1 else 0
            new_data[col] = st.selectbox(label, options=unique_vals, key=f"pred_input_{col}", index=default_idx)
        else:
            default = float((min_val + max_val)/2)
            new_data[col] = st.number_input(label, min_value=float(min_val), max_value=float(max_val), value=default, step=0.01, key=f"pred_input_{col}")
    
    if st.button("🔮 Fazer Previsão", type="primary"):

        # Reorder to match exact feature order
        X_new = pd.DataFrame([new_data])[expected_features]
        pred = st.session_state.model.predict(X_new)[0]
        
        if st.session_state.is_classification and st.session_state.label_encoder:
            pred = st.session_state.label_encoder.inverse_transform([int(pred)])[0]
        
        st.success(f"**Previsão para {prediction_date}: {pred}**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Data", prediction_date)
            st.metric("Valor Previsto", pred)
        with col2:
            st.json({k: v for k, v in new_data.items() if k in expected_features})
else:
    st.warning("👆 Primeiro treine o modelo carregando dados e clicando em 'Treinar'")



st.markdown("---")
st.caption("Desenvolvido para fins educacionais · Random Forest · Streamlit")
