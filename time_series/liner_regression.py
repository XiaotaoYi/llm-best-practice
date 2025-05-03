import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 配置参数
DATA_PATH = "time_series/sales_data.csv"
STORES = 10
PRODUCTS = 50

def main():
    # ================== 数据加载与预处理 ==================
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    
    # 处理缺失值
    df['promotion_type'] = df['promotion_type'].fillna('无促销')
    df['promotion_discount'] = df['promotion_discount'].fillna(0)
    
    # ================== 特征工程 ==================
    # 添加周期性特征
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    
    # ================== 数据集划分 ==================
    # 按时间顺序划分（最后30天作为测试集）
    split_date = df['date'].max() - timedelta(days=30)
    train = df[df['date'] <= split_date]
    test = df[df['date'] > split_date]
    
    # 定义特征列
    features = [
        'store_id', 'product_id', 'temperature', 'is_holiday',
        'month_sin', 'month_cos', 'day_sin', 'day_cos',
        'weather_condition', 'season', 'is_promotion',
        'promotion_type', 'unit_price'
    ]
    
    X_train, y_train = train[features], train['quantity']
    X_test, y_test = test[features], test['quantity']
    
    # ================== 构建模型管道 ==================
    numeric_features = ['temperature', 'unit_price']
    categorical_features = ['store_id', 'product_id', 'weather_condition', 'season', 'promotion_type']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    # ================== 训练与评估 ==================
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("=== 模型评估 ===")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"R²: {r2_score(y_test, y_pred):.2f}")
    
    # ================== 全量训练 ==================
    final_model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    final_model.fit(df[features], df['quantity'])
    
    # ================== 生成未来30天数据 ==================
    # 获取最新元数据
    last_prices = df.groupby(['store_id', 'product_id'])['unit_price'].last().reset_index()
    
    # 生成未来日期
    last_date = df['date'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
    
    # 构建未来数据集
    future_data = pd.DataFrame({
        'date': np.repeat(future_dates, STORES*PRODUCTS),
        'store_id': np.tile(np.arange(1, STORES+1), len(future_dates)*PRODUCTS),
        'product_id': np.tile(np.repeat(np.arange(1, PRODUCTS+1), STORES), len(future_dates))
    })
    
    # 合并价格信息
    future_data = future_data.merge(last_prices, on=['store_id', 'product_id'])
    
    # 生成特征
    future_data['month'] = future_data['date'].dt.month
    future_data['day_of_week'] = future_data['date'].dt.day_of_week
    
    # 添加周期性特征
    future_data['month_sin'] = np.sin(2 * np.pi * future_data['month']/12)
    future_data['month_cos'] = np.cos(2 * np.pi * future_data['month']/12)
    future_data['day_sin'] = np.sin(2 * np.pi * future_data['day_of_week']/7)
    future_data['day_cos'] = np.cos(2 * np.pi * future_data['day_of_week']/7)
    
    # 填充假设值（可根据业务需求调整）
    future_data['temperature'] = 25  # 假设恒温
    future_data['is_holiday'] = 0
    future_data['weather_condition'] = '晴'
    future_data['season'] = future_data['month'].apply(lambda m: (m % 12 + 3)//3).map({1:'春',2:'夏',3:'秋',4:'冬'})
    future_data['is_promotion'] = 0
    future_data['promotion_type'] = '无促销'
    
    # ================== 执行预测 ==================
    future_pred = final_model.predict(future_data[features])
    future_data['predicted_quantity'] = np.round(future_pred).astype(int)
    
    # 保存结果
    future_data[['date', 'store_id', 'product_id', 'predicted_quantity']].to_csv(
        'future_sales_predictions.csv', index=False)
    print("\n预测结果已保存至 future_sales_predictions.csv")

if __name__ == "__main__":
    main()