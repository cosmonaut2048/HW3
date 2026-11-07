from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st
import numpy as np

from sklearn import metrics
import matplotlib.pyplot as plt

APP_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

@st.cache_data
def load_test(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data
def load_config(path: str) -> dict:
    return json.loads(Path(path).read_text())

@st.cache_data
def test_proba(_model, X: pd.DataFrame) -> np.ndarray:
    return _model.predict_proba(X)[:, 1]

model = load_model(str(APP_DIR / 'model.joblib'))
test_df = load_test(str(APP_DIR / 'test.csv'))
config = load_config(str(APP_DIR / 'config.json'))

FEATURES = [c for c in test_df.columns if c != 'target']
default_threshold = float(config['threshold'])

# Вероятности считаются один раз и при движении слайдера не пересчитываются.
p_test = test_proba(model, test_df[FEATURES])

# Сайдбар: порог.
with st.sidebar:
    st.header('Порог решения')
    threshold = st.slider(
        'Порог вероятности "хорошее вино"',
        min_value=0.0, max_value=1.0,
        value=default_threshold, step=0.01,
        help='Вино считается хорошим, если вероятность >= порога.',
    )
    st.caption(f'Порог по умолчанию из config.json: {default_threshold:.2f}')

tab_single, tab_metrics, tab_batch, tab_coef = st.tabs(
    ['Оценка одного вина', 'Модельные метрики', 'Пакетное предсказание', 'Что выучила модель']
)

# Вкладка 4.1: ввод 12 признаков.
with tab_single:
    st.subheader('Введите характеристики вина')

    input_values = {}

    if 'is_red' in FEATURES:
        is_red = st.radio(
            'Тип вина (is_red)',
            options=[0, 1],
            format_func=lambda v: 'Красное (1)' if v == 1 else 'Белое (0)',
            horizontal=True,
        )
        input_values['is_red'] = int(is_red)

    numeric_features = [f for f in FEATURES if f != 'is_red']
    for feat in numeric_features:
        col = test_df[feat]
        lo, hi = float(col.min()), float(col.max())
        med = float(col.median())
        input_values[feat] = st.number_input(
            feat,
            min_value=lo,
            max_value=hi,
            value=med,
            step=(hi - lo) / 100.0 if hi > lo else 0.01,
            format='%.4f',
        )

    # Предсказание.
    X_input = pd.DataFrame([input_values])[FEATURES]

    proba = float(model.predict_proba(X_input)[:, 1][0])
    decision = int(proba >= threshold)

    st.subheader('Результат')
    col1, col2 = st.columns(2)
    with col1:
        st.metric('Вероятность "хорошее вино"', f'{proba:.3f}')
    with col2:
        st.metric('Решение при пороге', 'Хорошее (1)' if decision else 'Не хорошее (0)')

    if decision:
        st.success(f'Вино отобрано в подборку (p = {proba:.3f} ≥ порога {threshold:.2f}).')
    else:
        st.info(f'Вино не отобрано (p = {proba:.3f} < порога {threshold:.2f}).')

# Вкладка 4.2: Модельные метрики.
with tab_metrics:
    st.subheader(f'Метрики на тестовой выборке при пороге {threshold:.2f}')

    y_true = test_df['target'].to_numpy()
    y_pred = (p_test >= threshold).astype(int)

    # Матрица ошибок.
    st.markdown('**Матрица ошибок**')
    cm = metrics.confusion_matrix(y_true, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(4, 3.5))
    metrics.ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['не хорошее (0)', 'хорошее (1)'],
    ).plot(ax=ax_cm, cmap='Blues', colorbar=False, values_format='d')
    ax_cm.set_title(f'Confusion matrix (t={threshold:.2f})')
    st.pyplot(fig_cm)
    plt.close(fig_cm)

    # precision, recall, F1.
    precision = metrics.precision_score(y_true, y_pred, zero_division=0)
    recall = metrics.recall_score(y_true, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, zero_division=0)

    c1, c2, c3 = st.columns(3)
    c1.metric('Precision', f'{precision:.3f}')
    c2.metric('Recall', f'{recall:.3f}')
    c3.metric('F1', f'{f1:.3f}')

    # ROC-кривая с точкой текущего порога.
    fpr, tpr, thr_roc = metrics.roc_curve(y_true, p_test)
    # TPR/FPR в точке текущего порога.
    y_pred_t = (p_test >= threshold).astype(int)
    tp = int(((y_true == 1) & (y_pred_t == 1)).sum())
    fn = int(((y_true == 1) & (y_pred_t == 0)).sum())
    fp = int(((y_true == 0) & (y_pred_t == 1)).sum())
    tn = int(((y_true == 0) & (y_pred_t == 0)).sum())
    tpr_pt = tp / (tp + fn) if (tp + fn) else 0.0
    fpr_pt = fp / (fp + tn) if (fp + tn) else 0.0

    fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
    ax_roc.plot(fpr, tpr, lw=2, label=f'ROC-AUC = {metrics.roc_auc_score(y_true, p_test):.3f}')
    ax_roc.plot([0, 1], [0, 1], color='grey', ls=':')
    ax_roc.scatter([fpr_pt], [tpr_pt], s=120, marker='o', color='red', zorder=5,
                   label=f'порог {threshold:.2f}')
    ax_roc.set_xlabel('FPR')
    ax_roc.set_ylabel('TPR (recall)')
    ax_roc.set_title('ROC-кривая')
    ax_roc.legend(loc='lower right')
    st.pyplot(fig_roc)
    plt.close(fig_roc)

    # PR-кривая с точкой текущего порога.
    prec_c, rec_c, _ = metrics.precision_recall_curve(y_true, p_test)
    fig_pr, ax_pr = plt.subplots(figsize=(5, 4))
    ax_pr.plot(rec_c, prec_c, lw=2,
               label=f'PR-AUC = {metrics.average_precision_score(y_true, p_test):.3f}')
    ax_pr.axhline(y_true.mean(), color='grey', ls=':', label=f'random ({y_true.mean():.3f})')
    ax_pr.scatter([recall], [precision], s=120, marker='o', color='red', zorder=5,
                  label=f'порог {threshold:.2f}')
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('PR-кривая')
    ax_pr.legend(loc='upper right')
    st.pyplot(fig_pr)
    plt.close(fig_pr)

    # Текст для сомелье.
    st.markdown(
        '**Что произойдёт с подборкой при изменении порога**\n\n'
        '- **Порог выше** -- модель строже: в подборку попадут только вина с высокой '
        'вероятностью "хорошее". Получаем меньше посредственных вин, '
        'но часть хороших вин будет пропущена.\n'
        '- **Порог ниже** -- модель мягче: в подборку попадает больше вин. '
        'Пропускаем меньше хороших вин, но выше риск включить '
        'посредственное вино, подборка станет шире и менее надёжной.'
    )

# Вкладка 4.3: Пакетное предсказание.
with tab_batch:
    st.subheader('Пакетное предсказание по CSV')
    st.caption(f'Текущий порог из сайдбара: **{threshold:.2f}**')

    uploaded = st.file_uploader('Загрузите CSV с признаками вин (без столбца target)',
                                type=['csv'])

    if uploaded is None:
        st.info('Файл не загружен.')
    else:
        try:
            df_in = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f'Не удалось прочитать CSV: {e}')
            df_in = None

        if df_in is not None:
            missing = [c for c in FEATURES if c not in df_in.columns]
            if missing:
                st.error(
                    'В файле не хватает обязательных столбцов: '
                    + ', '.join(missing)
                    + f'. Ожидаются признаки: {", ".join(FEATURES)}.'
                )
            elif len(df_in) == 0:
                st.error('Загруженный CSV пуст - нет строк для предсказания.')
            else:
                X_in = df_in[FEATURES]
                proba = model.predict_proba(X_in)[:, 1]
                pred = (proba >= threshold).astype(int)

                out = df_in.copy()
                out['probability'] = proba
                out['prediction'] = pred
                out['decision'] = np.where(pred == 1, 'хорошее (1)', 'не хорошее (0)')

                st.markdown('**Результаты предсказания**')
                st.dataframe(out, use_container_width=True)

                csv_bytes = out.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label='Скачать результаты (CSV)',
                    data=csv_bytes,
                    file_name='wine_predictions.csv',
                    mime='text/csv',
                )

# Вкладка 4.4: Что выучила модель.
with tab_coef:
    st.subheader('Коэффициенты логистической регрессии')

    # Коэффициенты из обученного пайплайна (признаки уже стандартизованы).
    lr = model.named_steps['logisticregression']
    coefs = pd.Series(lr.coef_[0], index=FEATURES).sort_values()

    # Горизонтальная столбчатая диаграмма.
    fig_coef, ax_coef = plt.subplots(figsize=(7, 5))
    colors = ['#d62728' if v < 0 else '#2ca02c' for v in coefs.values]
    ax_coef.barh(coefs.index, coefs.values, color=colors)
    ax_coef.axvline(0, color='black', lw=0.8)
    ax_coef.set_xlabel('Коэффициент (после масштабирования)')
    ax_coef.set_title('Вклад признаков в решение «хорошее вино»')
    st.pyplot(fig_coef)
    plt.close(fig_coef)

    top_pos = coefs.sort_values(ascending=False).head(3)
    top_neg = coefs.head(3)

    st.markdown(
        '**Сильнее всего голосуют за хорошее вино (коэффициент > 0):** '
        + ', '.join(f'{name} ({val:+.3f})' for name, val in top_pos.items())
        + '.\n\n'
        '**Сильнее всего голосуют против (коэффициент < 0):** '
        + ', '.join(f'{name} ({val:+.3f})' for name, val in top_neg.items())
        + '.'
    )