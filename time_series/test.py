import pandas as pd
import numpy as np
import sqlparse
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
import umap.plot as umap_plot
from umap import UMAP
from difflib import SequenceMatcher
import re
from collections import defaultdict

# ======================
# 1. 数据生成模块
# ======================
class DataGenerator:
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        self.patterns = [
            self._time_range_agg,
            self._top_n_query,
            self._status_filter_group,
            self._random_query
        ]
    
    def _time_range_agg(self):
        """时间范围聚合模式"""
        days = np.random.choice([7, 30, 90])
        tables = ["sales", "orders", "transactions"]
        metrics = ["amount", "price", "revenue"]
        return (
            f"SELECT SUM({np.random.choice(metrics)}) FROM {np.random.choice(tables)} "
            f"WHERE date BETWEEN CURRENT_DATE - INTERVAL '{days} DAY' AND CURRENT_DATE"
        )
    
    def _top_n_query(self):
        """TOP N查询模式"""
        limits = [5, 10, 20]
        dimensions = ["product_id", "item_id", "sku"]
        return (
            f"SELECT {np.random.choice(dimensions)}, SUM(price) AS revenue "
            f"FROM orders GROUP BY 1 ORDER BY 2 DESC LIMIT {np.random.choice(limits)}"
        )
    
    def _status_filter_group(self):
        """状态过滤分组模式"""
        statuses = ["'completed'", "'pending'", "'shipped'"]
        return (
            f"SELECT DATE(order_time), COUNT(*) FROM orders "
            f"WHERE status = {np.random.choice(statuses)} GROUP BY 1"
        )
    
    def _random_query(self):
        """随机噪声查询"""
        return f"SELECT {np.random.choice(['*','COUNT(*)'])} " \
               f"FROM {np.random.choice(['users','inventory'])} " \
               f"WHERE {np.random.choice(['age > 30','price < 100'])}"
    
    def generate(self):
        data = []
        for _ in range(self.num_samples):
            # 控制模式分布：高频模式占70%
            pattern = np.random.choice(self.patterns, p=[0.3, 0.25, 0.15, 0.3])
            data.append({
                "query_id": _,
                "sql": pattern(),
                "exec_count": np.random.randint(1, 100),
                "timestamp": pd.Timestamp.now() - pd.Timedelta(minutes=np.random.randint(0, 10080))
            })
        return pd.DataFrame(data)

# ======================
# 2. 特征提取模块
# ======================
class FeatureExtractor:
    @staticmethod
    def extract(sql):
        parsed = sqlparse.parse(sql)[0]
        features = {
            "has_where": int(any(tkn.value.upper() == "WHERE" for tkn in parsed.flatten())),
            "has_group_by": int(any(tkn.value.upper() == "GROUP BY" for tkn in parsed.flatten())),
            "has_order_by": int(any(tkn.value.upper() == "ORDER BY" for tkn in parsed.flatten())),
            "join_count": sum(1 for tkn in parsed.flatten() if tkn.value.upper() == "JOIN"),
            "func_sum": len(re.findall(r"SUM\s*\(", sql, re.IGNORECASE)),
            "func_count": len(re.findall(r"COUNT\s*\(", sql, re.IGNORECASE)),
            "cond_between": len(re.findall(r"BETWEEN\s+.+?\s+AND", sql, re.IGNORECASE)),
            "cond_in": len(re.findall(r"IN\s*\(.+?\)", sql, re.IGNORECASE)),
        }
        return features

# ======================
# 3. 聚类分析模块
# ======================
class ClusterAnalyzer:
    def __init__(self):
        self.vectorizer = DictVectorizer(sparse=False)
        self.scaler = StandardScaler()
        self.cluster_model = OPTICS(min_samples=20, xi=0.05)
    
    def analyze(self, features_df):
        # 特征向量化
        X = self.vectorizer.fit_transform(features_df.to_dict('records'))
        
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 聚类分析
        self.cluster_model.fit(X_scaled)
        
        # 降维可视化
        reducer = UMAP(n_components=2)
        embeddings = reducer.fit_transform(X_scaled)
        
        return self.cluster_model.labels_, embeddings

# ======================
# 4. 模板生成模块
# ======================
class TemplateGenerator:
    @staticmethod
    def generate(cluster_queries):
        base_sql = cluster_queries.iloc[0]['sql']
        params = defaultdict(list)
        
        # 多序列比对提取参数
        for _, row in cluster_queries.iterrows():
            sm = SequenceMatcher(None, base_sql, row['sql'])
            for opcode in sm.get_opcodes():
                tag, i1, i2, j1, j2 = opcode
                if tag == 'replace':
                    segment = row['sql'][j1:j2]
                    param_name = TemplateGenerator._identify_param(base_sql[i1:i2], segment)
                    params[param_name].append(segment)
                    base_sql = base_sql[:i1] + f"${param_name}" + base_sql[i2:]
        
        # 清理重复参数
        template_sql = re.sub(r"\$\w+(\$\w+)+", lambda m: m.group().split('$')[-1], base_sql)
        
        return {
            "pattern": template_sql,
            "params": {k: list(set(v)) for k, v in params.items()},
            "sample": cluster_queries.iloc[0]['sql'],
            "exec_count": cluster_queries['exec_count'].sum()
        }
    
    @staticmethod
    def _identify_param(base_seg, new_seg):
        """参数类型识别"""
        if any(kw in base_seg.upper() for kw in ["SUM", "COUNT", "AVG"]):
            return "METRIC"
        if any(kw in base_seg.upper() for kw in ["FROM", "JOIN"]):
            return "TABLE"
        if any(kw in base_seg.upper() for kw in ["WHERE", "HAVING"]):
            return "CONDITION"
        if "GROUP BY" in base_seg.upper():
            return "DIMENSION"
        return "PARAM"

# ======================
# 主流程执行
# ======================
if __name__ == "__main__":
    # 生成模拟数据
    print("生成模拟数据...")
    df = DataGenerator(num_samples=1000).generate()
    
    # 特征提取
    print("提取特征...")
    df['features'] = df['sql'].apply(FeatureExtractor.extract)
    features_df = pd.json_normalize(df['features'])
    
    # 聚类分析
    print("执行聚类分析...")
    analyzer = ClusterAnalyzer()
    labels, embeddings = analyzer.analyze(features_df)
    df['cluster'] = labels
    df[['umap_x', 'umap_y']] = embeddings
    
    # 生成模板
    print("生成预计算模板...")
    templates = []
    for cid in df['cluster'].unique():
        if cid == -1:
            continue  # 跳过噪声点
        cluster_df = df[df['cluster'] == cid]
        if len(cluster_df) > 50:  # 高频模式阈值
            template = TemplateGenerator.generate(cluster_df)
            template['cluster'] = cid
            templates.append(template)
    
    # 输出结果
    print("\n生成的预计算模板：")
    for i, template in enumerate(templates[:3]):  # 展示前3个模板
        print(f"模板{i+1}:")
        print(f"Pattern: {template['pattern']}")
        print(f"Parameters: {template['params']}")
        print(f"示例查询: {template['sample']}")
        print(f"执行次数: {template['exec_count']}\n")

    # 可视化聚类结果
    umap_plot.points(df['umap_x'], df['umap_y'], labels=df['cluster'])