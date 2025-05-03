import pandas as pd
import numpy as np
from datetime import datetime

# 配置参数
YEARS = 3
STORES = 10
PRODUCTS = 50
OUTPUT_FILE = "sales_data.csv"

def generate_sales_data():
    # 生成日期范围（近3年）
    dates = pd.date_range(end=datetime.today(), periods=365*YEARS).to_pydatetime().tolist()
    
    # 生成门店元数据
    stores = [{
        "store_id": i,
        "store_name": f"门店_{i}",
        "city": np.random.choice(["北京", "上海", "广州", "深圳", "成都"])
    } for i in range(1, STORES+1)]

    # 生成商品元数据
    products = [{
        "product_id": i,
        "product_name": f"商品_{i}",
        "category": np.random.choice(["食品", "饮料", "日用品", "电子产品", "服饰"]),
        "base_price": np.random.uniform(10, 500)
    } for i in range(1, PRODUCTS+1)]

    # 初始化数据容器
    records = []
    
    # 逐日生成数据
    for date in dates:
        # 每日环境参数（全部门店共用）
        weather = np.random.choice(["晴", "雨", "多云", "雪", "雾"], 
                                p=[0.6, 0.1, 0.15, 0.1, 0.05])
        temp = np.random.normal(loc=20, scale=10)
        is_holiday = np.random.choice([0,1], p=[0.9, 0.1])
        
        # 遍历所有门店
        for store in stores:
            # 门店促销概率（20%概率有促销）
            store_promotion = np.random.choice([0,1], p=[0.8, 0.2])
            
            # 遍历所有商品
            for product in products:
                # ====== 核心销量生成逻辑 ======
                # 基准销量（泊松分布）
                base_sales = np.random.poisson(lam=10)
                
                # 价格影响（价格越高销量越低）
                price_factor = 1 / (product["base_price"]/100 + 0.1)
                
                # 促销影响（促销时销量翻倍）
                promotion = 1
                promo_type = None
                promo_discount = 0
                if store_promotion and np.random.rand() < 0.3:  # 30%商品参与促销
                    promotion = 2 + np.random.rand()  # 2-3倍
                    promo_type = np.random.choice(["折扣", "满减", "赠品"])
                    promo_discount = np.random.uniform(0.1, 0.3) if promo_type == "折扣" else 0
                
                # 天气影响（雨雪天气销量降低）
                weather_factor = 1
                if weather in ["雨", "雪"]:
                    weather_factor = 0.6 + np.random.rand()*0.4
                
                # 最终销量计算
                quantity = int(base_sales * price_factor * promotion * weather_factor)
                
                # ====== 构造记录 ======
                record = {
                    "date": date.strftime("%Y-%m-%d"),
                    "store_id": store["store_id"],
                    "store_name": store["store_name"],
                    "product_id": product["product_id"],
                    "product_name": product["product_name"],
                    "quantity": max(0, quantity),  # 确保非负
                    "unit_price": round(product["base_price"], 2),
                    "is_promotion": int(promotion > 1),
                    "promotion_type": promo_type,
                    "promotion_discount": round(promo_discount, 2) if promo_discount else None,
                    "temperature": round(temp, 1),
                    "weather_condition": weather,
                    "is_holiday": is_holiday,
                    "day_of_week": date.isoweekday(),  # 1-7（周一到周日）
                    "month": date.month,
                    "season": (date.month%12 + 3)//3  # 季节（1-4）
                }
                
                records.append(record)
                
        # 进度提示
        if len(records) % 10000 == 0:
            print(f"已生成 {len(records)} 条记录...")
    
    # 转换为DataFrame
    df = pd.DataFrame(records)
    
    # 添加季节名称映射
    df["season"] = df["season"].map({
        1: "春", 2: "夏", 3: "秋", 4: "冬"
    })
    
    # 保存为CSV
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"数据生成完成！总计 {len(df)} 条记录，已保存至 {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_sales_data()