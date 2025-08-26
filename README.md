# Анализ электромобилей — *Electric Vehicle Population 2025*

Этот репозиторий содержит ноутбук **`electrocars.ipynb`** с прикладным анализом открытого датасета *Electric Vehicle Population 2025* и базовыми моделями классификации. Проект ориентирован на быструю разведку данных (EDA), сравнение простых алгоритмов и получение прикладных инсайтов для устойчивого транспорта.

## Датасет
**Electric Vehicle Population 2025** — агрегированная выборка зарегистрированных электромобилей к 2025 году. Поля включают: марку/модель, год выпуска, тип EV (BEV/PHEV), запас хода, энергокомпанию, признак eligibility по чистому топливу, а также географию (штат/город/почтовый индекс). Набор полезен для анализа проникновения EV, региональной структуры, влияния политики и планирования инфраструктуры.

> Источник данных указывается пользователем (см. комментарии в ноутбуке). Файл подключается локально в ячейках загрузки данных.

## Что внутри ноутбука
- **Загрузка и очистка**: обработка пропусков (`SimpleImputer`), приведение типов, базовые преобразования категориальных признаков (`LabelEncoder`/factorize).
- **EDA**: краткие срезы `.head()/.describe()`, частоты `value_counts()` (бренды/годы/типы), несколько визуализаций (распределения, сравнения; `matplotlib/seaborn`). *(≈ 8 графиков, 5 табличных срезов)*
- **Моделирование (baseline)**: сравнение нескольких алгоритмов из `scikit-learn` и `xgboost` по метрике `accuracy`:
  - `LogisticRegression`, `RandomForestClassifier`, `KNeighborsClassifier`,
  - `SVC`, `GaussianNB`, `DecisionTreeClassifier`,
  - (опционально) `XGBClassifier` — при наличии пакета `xgboost`.
- **Оценка**: быстрый `train_test_split` без тяжёлой кросс-валидации для интерактивности; сводная таблица `Algorithm / Accuracy (%)`.

## Цели проекта
1. Получить быстрые и воспроизводимые срезы по рынку EV: структура, лидирующие бренды/типы, региональные различия.
2. Сопоставить простые алгоритмы классификации как baseline и понять предсказуемость целевой постановки.
3. Сформировать артефакты (графики/таблицы) для решения прикладных задач: где усиливать зарядную инфраструктуру, какие модели/диапазоны пробега приоритетны и т.д.

## Быстрый старт
```bash
# 1) создать среду и установить зависимости
pip install -r requirements.txt
# либо минимально:
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
# 2) открыть ноутбук
jupyter notebook electrocars.ipynb
```

> Добавьте файл датасета (CSV) и укажите путь в соответствующей ячейке загрузки.

## Минимальный конвейер (пример кода)
```python
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("Electric_Vehicle_Population_Data.csv")

target = "Electric Vehicle Type"
y = LabelEncoder().fit_transform(df[target].astype(str))
X = df.drop(columns=[target]).copy()

for c in X.columns:
    if X[c].dtype == "object":
        X[c] = pd.factorize(X[c].astype(str))[0]
    else:
        X[c] = pd.to_numeric(X[c], errors="coerce")
X = SimpleImputer(strategy="most_frequent").fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = LogisticRegression(max_iter=250)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, pred)*100, 2), "%")
```

## Советы по скорости
- Используйте **`train_test_split`** вместо тяжёлых CV для интерактива.
- Ограничьте число признаков на старте (10–15 колонок) и включайте остальные по мере необходимости.
- Для больших выборок примените даунсемплинг на этапе прототипирования.
- `RandomForest` с умеренной глубиной (`max_depth≈12–14`, `n_estimators≈80–120`) даёт хороший баланс скорость/качество.
- `xgboost` подключайте только при необходимости — он заметно тяжелее.

## Частые проблемы
- **KNN + pandas.DataFrame**: ошибка вида `AttributeError: 'Flags' object has no attribute 'c_contiguous'`.\
  Решение — передавать в модели **NumPy-массивы**:
  ```python
  import numpy as np
  X_train = np.ascontiguousarray(X_train.to_numpy(np.float32))
  X_test  = np.ascontiguousarray(X_test.to_numpy(np.float32))
  ```
- Дисбаланс классов — используйте `stratify` в `train_test_split` и проверяйте `value_counts()` по целевой переменной.

## Структура репозитория
```
.
├─ electrocars.ipynb        # основной ноутбук с EDA и baseline-моделями
├─ Electric_Vehicle_Population_Data.zip   # датасет
└─ README.md                # этот файл
```

## Лицензия и использование
Материалы предоставлены в образовательных и исследовательских целях. Проверьте лицензию исходного датасета и укажите источник при публикации результатов.

## Благодарности
Сообществу открытых данных, авторам датасета и разработчикам библиотек: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`.
