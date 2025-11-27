import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from flask import Flask, render_template_string, request
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Анализ продаж химиотерапевтических средств</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .plot-container {
            margin: 20px 0;
            text-align: center;
        }
        .metrics {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .data-input {
            margin-bottom: 10px;
        }
        .error-message {
            color: red;
            margin-bottom: 15px;
        }
        .plot-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-top: 20px;
        }
        .plot-item {
            border: 1px solid #ddd;
            padding: 10px;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="container mt-4">
        <h1 class="text-center">Анализ химиотерапевтических средств</h1>

        <form method="POST" class="mt-4">
            <input type="hidden" name="form_type" value="analysis">
            <div class="row">
                <div class="col-md-8">
                    <h3>Данные о продажах</h3>
                    <div id="sales-data">
                        {% for i in range(inputs|length) %}
                        <div class="data-input row g-2">
                            <div class="col-md-2">
                                <input type="number" class="form-control" name="price[]" placeholder="Цена (руб)" step="0.1" min="0" 
                                    value="{{ inputs[i].price if inputs else '' }}" required>
                            </div>
                            <div class="col-md-2">
                                <input type="number" class="form-control" name="sales[]" placeholder="Продажи (упаковка)" min="0" 
                                    value="{{ inputs[i].sales if inputs else '' }}" required>
                            </div>
                            <div class="col-md-2">
                                <select class="form-select" name="season[]" required>
                                    <option value="">Сезон</option>
                                    <option value="winter" {% if inputs and inputs[i].season=='winter' %}selected{% endif %}>Зима</option>
                                    <option value="spring" {% if inputs and inputs[i].season=='spring' %}selected{% endif %}>Весна</option>
                                    <option value="summer" {% if inputs and inputs[i].season=='summer' %}selected{% endif %}>Лето</option>
                                    <option value="autumn" {% if inputs and inputs[i].season=='autumn' %}selected{% endif %}>Осень</option>
                                </select>
                            </div>
                            <div class="col-md-2">
                                <select class="form-select" name="size[]" required>
                                    <option value="">Размер</option>
                                    <option value="small" {% if inputs and inputs[i].size=='small' %}selected{% endif %}>Противомикробные</option>
                                    <option value="medium" {% if inputs and inputs[i].size=='medium' %}selected{% endif %}>Антисептические</option>
                                    <option value="large" {% if inputs and inputs[i].size=='large' %}selected{% endif %}>Дезинфекционные</option>
                                </select>
                            </div>
                            <div class="col-md-2">
                                <select class="form-select" name="color[]" required>
                                    <option value="">Цвет</option>
                                    <option value="white" {% if inputs and inputs[i].color=='white' %}selected{% endif %}>Антибиотики</option>
                                    <option value="yellow" {% if inputs and inputs[i].color=='yellow' %}selected{% endif %}>Сульфаниламидные</option>
                                    <option value="red" {% if inputs and inputs[i].color=='red' %}selected{% endif %}>Противотуберкулезные</option>
                                    <option value="purple" {% if inputs and inputs[i].color=='purple' %}selected{% endif %}>Противовирусные</option>
                                    <option value="blue" {% if inputs and inputs[i].color=='blue' %}selected{% endif %}>Противогельминтные</option>
                                </select>
                            </div>
                        </div>
                        {% endfor %}
                    </div>
                    <button type="button" class="btn btn-secondary mt-2" onclick="addDataField()">Добавить данные</button>
                </div>

                <div class="col-md-4">
                    <h3>Модель регрессии</h3>
                    <div class="mb-3">
                        <label for="model_type" class="form-label">Тип модели:</label>
                        <select class="form-select" id="model_type" name="model_type">
                            <option value="linear" {% if model_type=='linear' %}selected{% endif %}>Линейная регрессия</option>
                            <option value="polynomial" {% if model_type=='polynomial' %}selected{% endif %}>Полиномиальная регрессия</option>
                            <option value="ridge" {% if model_type=='ridge' %}selected{% endif %}>Ridge регрессия</option>
                            <option value="lasso" {% if model_type=='lasso' %}selected{% endif %}>Lasso регрессия</option>
                            <option value="elasticnet" {% if model_type=='elasticnet' %}selected{% endif %}>ElasticNet</option>
                        </select>
                    </div>

                    <div id="params-container">
                        {% if model_type == 'polynomial' %}
                        <div class="mb-3">
                            <label for="degree" class="form-label">Степень полинома:</label>
                            <input type="number" class="form-control" id="degree" name="degree" value="{{ degree if degree else 2 }}" min="1" max="3">
                        </div>
                        {% elif model_type in ['ridge', 'lasso'] %}
                        <div class="mb-3">
                            <label for="alpha" class="form-label">Alpha (регуляризация):</label>
                            <input type="number" class="form-control" id="alpha" name="alpha" value="{{ alpha if alpha else ('1.0' if model_type=='ridge' else '0.1') }}" min="0" step="0.1">
                        </div>
                        {% elif model_type == 'elasticnet' %}
                        <div class="mb-3">
                            <label for="alpha" class="form-label">Alpha (регуляризация):</label>
                            <input type="number" class="form-control" id="alpha" name="alpha" value="{{ alpha if alpha else '0.1' }}" min="0" step="0.1">
                        </div>
                        <div class="mb-3">
                            <label for="l1_ratio" class="form-label">L1 ratio (соотношение L1/L2):</label>
                            <input type="number" class="form-control" id="l1_ratio" name="l1_ratio" value="{{ l1_ratio if l1_ratio else '0.5' }}" min="0" max="1" step="0.1">
                        </div>
                        {% endif %}
                    </div>
                </div>
            </div>

            <div class="text-center mt-3">
                <button type="submit" class="btn btn-primary">Анализировать данные</button>
            </div>
        </form>

        {% if show_results %}
        <div class="mt-4">
            <h2>Результаты: {{ model_name }}</h2>

            <div class="plot-container">
                <img src="data:image/png;base64,{{ plot_url }}" alt="График регрессии" class="img-fluid">
            </div>

            <div class="plot-grid">
                <div class="plot-item">
                    <img src="data:image/png;base64,{{ season_plot_url }}" alt="График по сезонам" class="img-fluid">
                </div>
                <div class="plot-item">
                    <img src="data:image/png;base64,{{ size_plot_url }}" alt="График по размерам" class="img-fluid">
                </div>
                <div class="plot-item">
                    <img src="data:image/png;base64,{{ color_plot_url }}" alt="График по цветам" class="img-fluid">
                </div>
            </div>

            <div class="metrics">
                <h4>Метрики модели:</h4>
                <p><strong>Среднеквадратичная ошибка (MSE):</strong> {{ "%.2f"|format(mse) }}</p>
                <p><strong>Коэффициент детерминации (R²):</strong> {{ "%.4f"|format(r2) }}</p>

                <h4 class="mt-3">Параметры модели:</h4>
                {% for feature, coef_value in feature_coef.items() %}
                <p><strong>{{ feature }}:</strong> {{ "%.4f"|format(coef_value) }}</p>
                {% endfor %}

                {% if prediction %}
                <h4 class="mt-3">Прогноз:</h4>
                <p>Прогнозируемые продажи: <strong>{{ "%.2f"|format(prediction) }} упк</strong></p>
                <p>При цене: {{ "%.2f"|format(prediction_price) }} руб</p>
                <p>Сезон: {{ prediction_season }}</p>
                <p>Размер: {{ prediction_size }}</p>
                <p>Цвет: {{ prediction_color }}</p>
                {% endif %}
            </div>

            <form method="POST" class="mt-3">
                <input type="hidden" name="form_type" value="prediction">
                <input type="hidden" name="model_type" value="{{ model_type }}">
                {% if model_type == 'polynomial' %}<input type="hidden" name="degree" value="{{ degree }}">{% endif %}
                {% if model_type in ['ridge', 'lasso', 'elasticnet'] %}<input type="hidden" name="alpha" value="{{ alpha }}">{% endif %}
                {% if model_type == 'elasticnet' %}<input type="hidden" name="l1_ratio" value="{{ l1_ratio }}">{% endif %}

                <!-- Передаем все исходные данные -->
                {% for input in inputs %}
                <input type="hidden" name="price[]" value="{{ input.price }}">
                <input type="hidden" name="sales[]" value="{{ input.sales }}">
                <input type="hidden" name="season[]" value="{{ input.season }}">
                <input type="hidden" name="size[]" value="{{ input.size }}">
                <input type="hidden" name="color[]" value="{{ input.color }}">
                {% endfor %}

                <h4>Сделать прогноз</h4>
                <div class="row g-3">
                    <div class="col-md-2">
                        <input type="number" class="form-control" name="prediction_price" placeholder="Цена" step="0.1" min="0" required 
                            value="{{ prediction_price if prediction_price else '' }}">
                    </div>
                    <div class="col-md-2">
                        <select class="form-select" name="prediction_season" required>
                            <option value="">Сезон</option>
                            <option value="winter" {% if prediction_season == 'winter' %}selected{% endif %}>Зима</option>
                            <option value="spring" {% if prediction_season == 'spring' %}selected{% endif %}>Весна</option>
                            <option value="summer" {% if prediction_season == 'summer' %}selected{% endif %}>Лето</option>
                            <option value="autumn" {% if prediction_season == 'autumn' %}selected{% endif %}>Осень</option>
                        </select>
                    </div>
                    <div class="col-md-2">
                        <select class="form-select" name="prediction_size" required>
                            <option value="">Размер</option>
                            <option value="small" {% if prediction_size == 'small' %}selected{% endif %}>Противомикробные</option>
                            <option value="medium" {% if prediction_size == 'medium' %}selected{% endif %}>Антисептические</option>
                            <option value="large" {% if prediction_size == 'large' %}selected{% endif %}>Дезинфекционные</option>
                        </select>
                    </div>
                    <div class="col-md-2">
                        <select class="form-select" name="prediction_color" required>
                            <option value="">Цвет</option>
                            <option value="white" {% if prediction_color == 'white' %}selected{% endif %}>Антибиотики</option>
                            <option value="yellow" {% if prediction_color == 'yellow' %}selected{% endif %}>Сульфаниламидные</option>
                            <option value="red" {% if prediction_color == 'red' %}selected{% endif %}>Противотуберкулезные</option>
                            <option value="purple" {% if prediction_color == 'purple' %}selected{% endif %}>Противовирусные</option>
                            <option value="blue" {% if prediction_color =='blue' %}selected{% endif %}>Противогельминтные</option>
                        </select>
                    </div>
                    <div class="col-md-2">
                        <button type="submit" class="btn btn-success">Прогнозировать</button>
                    </div>
                </div>
            </form>
        </div>
        {% endif %}

        {% if error %}
        <div class="alert alert-danger mt-3">
            {{ error }}
        </div>
        {% endif %}
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        function addDataField() {
            const container = document.getElementById('sales-data');
            const newField = document.createElement('div');
            newField.className = 'data-input row g-2';
            newField.innerHTML = `
                <div class="col-md-2">
                    <input type="number" class="form-control" name="price[]" placeholder="Цена (руб)" step="0.1" min="0" required>
                </div>
                <div class="col-md-2">
                    <input type="number" class="form-control" name="sales[]" placeholder="Продажи (упк)" min="0" required>
                </div>
                <div class="col-md-2">
                    <select class="form-select" name="season[]" required>
                        <option value="">Сезон</option>
                        <option value="winter">Зима</option>
                        <option value="spring">Весна</option>
                        <option value="summer">Лето</option>
                        <option value="autumn">Осень</option>
                    </select>
                </div>
                <div class="col-md-2">
                    <select class="form-select" name="size[]" required>
                        <option value="">Размер</option>
                        <option value="small">Противомикробные</option>
                        <option value="medium">Антисептические</option>
                        <option value="large">Дезинфекционные</option>
                    </select>
                </div>
                <div class="col-md-2">
                    <select class="form-select" name="color[]" required>
                        <option value="">Цвет</option>
                        <option value="white">Антибиотики</option>
                        <option value="yellow">Сульфаниламидные</option>
                        <option value="red">Противотуберкулезные</option>
                        <option value="purple">Противовирусные</option>
                        <option value="blue">Противогельминтные</option>
                    </select>
                </div>
            `;
            container.appendChild(newField);
        }

        // Динамическая загрузка параметров модели
        document.getElementById('model_type').addEventListener('change', function() {
            const modelType = this.value;
            const container = document.getElementById('params-container');

            let html = '';

            if (modelType === 'polynomial') {
                html = `
                    <div class="mb-3">
                        <label for="degree" class="form-label">Степень полинома:</label>
                        <input type="number" class="form-control" id="degree" name="degree" value="2" min="1" max="3">
                    </div>
                `;
            } else if (modelType === 'ridge' || modelType === 'lasso') {
                const defaultValue = modelType === 'ridge' ? '1.0' : '0.1';
                html = `
                    <div class="mb-3">
                        <label for="alpha" class="form-label">Alpha (регуляризация):</label>
                        <input type="number" class="form-control" id="alpha" name="alpha" value="${defaultValue}" min="0" step="0.1">
                    </div>
                `;
            } else if (modelType === 'elasticnet') {
                html = `
                    <div class="mb-3">
                        <label for="alpha" class="form-label">Alpha (регуляризация):</label>
                        <input type="number" class="form-control" id="alpha" name="alpha" value="0.1" min="0" step="0.1">
                    </div>
                    <div class="mb-3">
                        <label for="l1_ratio" class="form-label">L1 ratio (соотношение L1/L2):</label>
                        <input type="number" class="form-control" id="l1_ratio" name="l1_ratio" value="0.5" min="0" max="1" step="0.1">
                    </div>
                `;
            }

            container.innerHTML = html;
        });

        // Инициализация параметров при загрузке
        document.addEventListener('DOMContentLoaded', function() {
            document.getElementById('model_type').dispatchEvent(new Event('change'));
        });
    </script>
</body>
</html>
"""


def generate_plots(X, y, model, model_name):
    plots = []

    # 1. Основной график зависимости продаж от цены
    plt.figure(figsize=(10, 6))
    price_range = np.linspace(X['price'].min(), X['price'].max(), 100)
    most_common = X.mode().iloc[0]

    X_plot = pd.DataFrame({
        'price': price_range,
        'season': [most_common['season']] * 100,
        'size': [most_common['size']] * 100,
        'color': [most_common['color']] * 100
    })
    y_plot = model.predict(X_plot)

    plt.scatter(X['price'], y, color='blue', label='Фактические данные')
    plt.plot(price_range, y_plot, color='red', label=f'Модель {model_name}')
    plt.xlabel('Цена за упаковку (руб)')
    plt.ylabel('Продажи (упаковок)')
    plt.title('Зависимость продаж от цены')
    plt.legend()
    plt.grid(True)

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots.append(base64.b64encode(buf.read()).decode('utf-8'))
    buf.close()
    plt.close()

    # 2. График по сезонам
    plt.figure(figsize=(10, 6))
    seasons = ['winter', 'spring', 'summer', 'autumn']
    avg_price = X['price'].mean()

    for season in seasons:
        season_data = y[X['season'] == season]
        if len(season_data) > 0:
            X_plot = pd.DataFrame({
                'price': [avg_price] * 10,
                'season': [season] * 10,
                'size': [most_common['size']] * 10,
                'color': [most_common['color']] * 10
            })
            y_plot = model.predict(X_plot)
            plt.scatter([season] * len(season_data), season_data, label=f'Факт: {season}')
            plt.plot([season] * 10, y_plot, 'r-', alpha=0.5)

    plt.xlabel('Сезон')
    plt.ylabel('Продажи упаковок')
    plt.title('Влияние сезона на продажи')
    plt.legend()
    plt.grid(True)

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots.append(base64.b64encode(buf.read()).decode('utf-8'))
    buf.close()
    plt.close()

    # 3. График по размерам
    plt.figure(figsize=(10, 6))
    sizes = ['small', 'medium', 'large']

    for size in sizes:
        size_data = y[X['size'] == size]
        if len(size_data) > 0:
            X_plot = pd.DataFrame({
                'price': [avg_price] * 10,
                'season': [most_common['season']] * 10,
                'size': [size] * 10,
                'color': [most_common['color']] * 10
            })
            y_plot = model.predict(X_plot)
            plt.scatter([size] * len(size_data), size_data, label=f'Факт: {size}')
            plt.plot([size] * 10, y_plot, 'r-', alpha=0.5)

    plt.xlabel('Вид средства')
    plt.ylabel('Продажи упаковок')
    plt.title('Влияние группы на продажи')
    plt.legend()
    plt.grid(True)

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots.append(base64.b64encode(buf.read()).decode('utf-8'))
    buf.close()
    plt.close()

    # 4. График по цветам
    plt.figure(figsize=(10, 6))
    colors = ['white', 'yellow', 'red', 'purple', 'blue']

    for color in colors:
        color_data = y[X['color'] == color]
        if len(color_data) > 0:
            X_plot = pd.DataFrame({
                'price': [avg_price] * 10,
                'season': [most_common['season']] * 10,
                'size': [most_common['size']] * 10,
                'color': [color] * 10
            })
            y_plot = model.predict(X_plot)
            plt.scatter([color] * len(color_data), color_data, label=f'Факт: {color}')
            plt.plot([color] * 10, y_plot, 'r-', alpha=0.5)

    plt.xlabel('Тип действующего вещества')
    plt.ylabel('Продажи упаковок')
    plt.title('Влияние типа действующего вещества на продажи')
    plt.legend()
    plt.grid(True)

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots.append(base64.b64encode(buf.read()).decode('utf-8'))
    buf.close()
    plt.close()

    return plots


def prepare_data(request):
    prices = request.form.getlist('price[]')
    sales = request.form.getlist('sales[]')
    seasons = request.form.getlist('season[]')
    sizes = request.form.getlist('size[]')
    colors = request.form.getlist('color[]')

    data = []
    for p, s, season, size, color in zip(prices, sales, seasons, sizes, colors):
        if p and s and season and size and color:
            try:
                data.append({
                    'price': float(p),
                    'sales': float(s),
                    'season': season,
                    'size': size,
                    'color': color
                })
            except ValueError:
                continue

    if len(data) < 3:
        return None, "Недостаточно данных. Введите как минимум 3 полных набора данных."

    df = pd.DataFrame(data)

    # Определяем все возможные категории
    season_categories = ['winter', 'spring', 'summer', 'autumn']
    size_categories = ['small', 'medium', 'large']
    color_categories = ['white', 'yellow', 'red', 'purple', 'blue']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(categories=[season_categories, size_categories, color_categories]),
             ['season', 'size', 'color'])
        ],
        remainder='passthrough'
    )

    X = df[['price', 'season', 'size', 'color']]
    y = df['sales'].values

    return (X, y, preprocessor, df.to_dict('records')), None


def create_model(model_type, request, preprocessor):
    if model_type == 'linear':
        model = make_pipeline(
            preprocessor,
            LinearRegression()
        )
        model_name = "Линейная регрессия"
    elif model_type == 'polynomial':
        degree = int(request.form.get('degree', 2))
        model = make_pipeline(
            preprocessor,
            PolynomialFeatures(degree),
            LinearRegression()
        )
        model_name = f"Полиномиальная регрессия (степень {degree})"
    elif model_type == 'ridge':
        alpha = float(request.form.get('alpha', 1.0))
        model = make_pipeline(
            preprocessor,
            Ridge(alpha=alpha)
        )
        model_name = f"Ridge регрессия (alpha={alpha})"
    elif model_type == 'lasso':
        alpha = float(request.form.get('alpha', 0.1))
        model = make_pipeline(
            preprocessor,
            Lasso(alpha=alpha)
        )
        model_name = f"Lasso регрессия (alpha={alpha})"
    elif model_type == 'elasticnet':
        alpha = float(request.form.get('alpha', 0.1))
        l1_ratio = float(request.form.get('l1_ratio', 0.5))
        model = make_pipeline(
            preprocessor,
            ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        )
        model_name = f"ElasticNet (alpha={alpha}, l1_ratio={l1_ratio})"

    return model, model_name


def get_model_params(model):
    if hasattr(model.named_steps, 'linearregression'):
        regressor = model.named_steps['linearregression']
        coef = regressor.coef_
        intercept = regressor.intercept_
    elif hasattr(model.named_steps, 'ridge'):
        regressor = model.named_steps['ridge']
        coef = regressor.coef_
        intercept = regressor.intercept_
    elif hasattr(model.named_steps, 'lasso'):
        regressor = model.named_steps['lasso']
        coef = regressor.coef_
        intercept = regressor.intercept_
    elif hasattr(model.named_steps, 'elasticnet'):
        regressor = model.named_steps['elasticnet']
        coef = regressor.coef_
        intercept = regressor.intercept_
    else:
        coef = []
        intercept = 0

    feature_names = []
    if 'onehotencoder' in model.named_steps:
        feature_names = model.named_steps['onehotencoder'].get_feature_names_out(['season', 'size', 'color'])
    feature_names = list(feature_names) + ['price']

    feature_coef = {}
    for name, value in zip(feature_names, coef):
        feature_coef[name] = value
    feature_coef['intercept'] = intercept

    return feature_coef


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        form_type = request.form.get('form_type', 'analysis')

        if form_type == 'prediction':
            return handle_prediction(request)
        else:
            return handle_analysis(request)

    return render_template_string(HTML_TEMPLATE,
                                  show_results=False,
                                  inputs=[],
                                  model_type='linear')


def handle_analysis(request):
    data, error = prepare_data(request)
    if error:
        return render_template_string(HTML_TEMPLATE,
                                      show_results=False,
                                      inputs=[],
                                      error=error)

    (X, y, preprocessor, inputs) = data

    model_type = request.form.get('model_type', 'linear')
    model, model_name = create_model(model_type, request, preprocessor)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    feature_coef = get_model_params(model)
    plots = generate_plots(X, y, model, model_name)

    return render_template_string(HTML_TEMPLATE,
                                  plot_url=plots[0],
                                  season_plot_url=plots[1],
                                  size_plot_url=plots[2],
                                  color_plot_url=plots[3],
                                  model_name=model_name,
                                  mse=mse,
                                  r2=r2,
                                  feature_coef=feature_coef,
                                  inputs=inputs,
                                  model_type=model_type,
                                  degree=request.form.get('degree', 2),
                                  alpha=request.form.get('alpha', 1.0 if model_type == 'ridge' else 0.1),
                                  l1_ratio=request.form.get('l1_ratio', 0.5),
                                  show_results=True)


def handle_prediction(request):
    data, error = prepare_data(request)
    if error:
        return render_template_string(HTML_TEMPLATE,
                                      show_results=False,
                                      inputs=[],
                                      error=error)

    (X, y, preprocessor, inputs) = data

    model_type = request.form.get('model_type', 'linear')
    model, model_name = create_model(model_type, request, preprocessor)

    model.fit(X, y)

    try:
        prediction_price = float(request.form.get('prediction_price', 0))
        prediction_season = request.form.get('prediction_season')
        prediction_size = request.form.get('prediction_size')
        prediction_color = request.form.get('prediction_color')

        if not all([prediction_season, prediction_size, prediction_color]):
            raise ValueError("Заполните все поля для прогноза")

        valid_seasons = ['winter', 'spring', 'summer', 'autumn']
        valid_sizes = ['small', 'medium', 'large']
        valid_colors = ['white', 'yellow', 'red', 'purple', 'blue']

        if prediction_season not in valid_seasons:
            raise ValueError(f"Недопустимый сезон. Допустимые значения: {', '.join(valid_seasons)}")
        if prediction_size not in valid_sizes:
            raise ValueError(f"Недопустимый размер. Допустимые значения: {', '.join(valid_sizes)}")
        if prediction_color not in valid_colors:
            raise ValueError(f"Недопустимый цвет. Допустимые значения: {', '.join(valid_colors)}")

        X_pred = pd.DataFrame([{
            'price': prediction_price,
            'season': prediction_season,
            'size': prediction_size,
            'color': prediction_color
        }])

        prediction = model.predict(X_pred)[0]

        feature_coef = get_model_params(model)
        plots = generate_plots(X, y, model, model_name)

        return render_template_string(HTML_TEMPLATE,
                                      plot_url=plots[0],
                                      season_plot_url=plots[1],
                                      size_plot_url=plots[2],
                                      color_plot_url=plots[3],
                                      model_name=model_name,
                                      mse=mean_squared_error(y, model.predict(X)),
                                      r2=r2_score(y, model.predict(X)),
                                      feature_coef=feature_coef,
                                      prediction=prediction,
                                      prediction_price=prediction_price,
                                      prediction_season=prediction_season,
                                      prediction_size=prediction_size,
                                      prediction_color=prediction_color,
                                      inputs=inputs,
                                      model_type=model_type,
                                      degree=request.form.get('degree', 2),
                                      alpha=request.form.get('alpha', 1.0 if model_type == 'ridge' else 0.1),
                                      l1_ratio=request.form.get('l1_ratio', 0.5),
                                      show_results=True)

    except ValueError as e:
        return render_template_string(HTML_TEMPLATE,
                                      show_results=False,
                                      inputs=inputs,
                                      model_type=model_type,
                                      degree=request.form.get('degree', 2),
                                      alpha=request.form.get('alpha', 1.0 if model_type == 'ridge' else 0.1),
                                      l1_ratio=request.form.get('l1_ratio', 0.5),
                                      error=str(e))


if __name__ == '__main__':
    app.run(debug=True)