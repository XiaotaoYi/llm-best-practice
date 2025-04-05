from datasets import load_from_disk

# 加载完整DatasetDict
loaded_dataset = load_from_disk("./medical_dialogue_dataset")

# 验证结构
print(loaded_dataset)
print("训练集样本:", len(loaded_dataset["train"]))
print("测试集样本:", len(loaded_dataset["test"]))

# 获取前100条（适合快速测试）
test_data = loaded_dataset["test"].select(range(100))

# 验证数量
print(f"获取的记录数：{len(test_data)}")

# 首先安装需要的库
# !pip install nltk rouge-score

import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# 假设我们有一个测试数据集，格式示例：
# test_dataset = [
#     {"input": "你的输入文本", "reference": "标准答案文本"},
#     ...
# ]

def evaluate_model(model, tokenizer, test_dataset, max_new_tokens=512):

    # 定义医疗对话的提示模板
    medical_prompt = """你是一个专业的医疗助手。请根据患者的问题提供专业、准确的回答。
    
    ### 问题：
    {}
    
    ### 回答：
    {}"""
    # 初始化评估工具
    smoothie = SmoothingFunction().method4  # BLEU平滑函数
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    bleu_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    i = 0
    for example in test_dataset:
        i = i + 1
        print("current progress i=" + str(i))
        # 生成预测文本
        inputs = tokenizer(medical_prompt.format(example["input"], ""), return_tensors="pt", padding=True, truncation=True).to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id
        )
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 获取参考文本
        reference = example["output"]
        
        # 计算BLEU
        # 使用模型tokenizer进行分词
        hyp_tokens = tokenizer.tokenize(prediction)
        ref_tokens = [tokenizer.tokenize(example["output"])]  # 注意嵌套列表
        bleu = sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=smoothie)
        bleu_scores.append(bleu)
        
        # 计算Rouge
        scores = scorer.score(reference, prediction)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    return {
        "bleu": np.mean(bleu_scores),
        "rouge1": np.mean(rouge1_scores),
        "rouge2": np.mean(rouge2_scores),
        "rougeL": np.mean(rougeL_scores),
    }

# 使用示例（在训练完成后调用）：
# 加载训练好的模型和tokenizer
# model.save_pretrained("outputs")
# model, tokenizer = FastLanguageModel.from_pretrained(...)

# 运行评估
# evaluation_results = evaluate_model(model, tokenizer, test_dataset)
# print(f"BLEU: {evaluation_results['bleu']:.4f}")
# print(f"ROUGE-1: {evaluation_results['rouge1']:.4f}")
# print(f"ROUGE-2: {evaluation_results['rouge2']:.4f}")
# print(f"ROUGE-L: {evaluation_results['rougeL']:.4f}")

# 模型推理示例
def generate_medical_response(model, tokenizer, medical_prompt, question):
    """生成医疗回答"""
    FastLanguageModel.for_inference(model)  # 启用原生2倍速推理
    inputs = tokenizer(
        [medical_prompt.format(question, "")],
        return_tensors="pt"
    ).to("cuda")
    
    from transformers import TextStreamer
    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )

    # 定义医疗对话的提示模板
medical_prompt = """你是一个专业的医疗助手。请根据患者的问题提供专业、准确的回答。

### 问题：
{}

### 回答：
{}"""

max_seq_length = 2048  # 设置最大序列长度，支持 RoPE 缩放
dtype = None  # 数据类型，None 表示自动检测。Tesla T4 使用 Float16，Ampere+ 使用 Bfloat16
load_in_4bit = True  # 使用 4bit 量化来减少内存使用

from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_model_medical",  # 训练时使用的模型
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model)  # 启用原生2倍速推理

# 测试问题
test_questions = [
    "我最近总是感觉头晕，应该怎么办？",
    "感冒发烧应该吃什么药？",
    "高血压患者需要注意什么？"
]

for question in test_questions:
    print("\n" + "="*50)
    print(f"问题：{question}")
    print("回答：")
    generate_medical_response(model,tokenizer,medical_prompt, question) 

# 导入必要的库
from unsloth import FastLanguageModel
import torch

# 设置模型参数
max_seq_length = 2048  # 设置最大序列长度，支持 RoPE 缩放
dtype = None  # 数据类型，None 表示自动检测。Tesla T4 使用 Float16，Ampere+ 使用 Bfloat16
load_in_4bit = True  # 使用 4bit 量化来减少内存使用

# 加载预训练模型和分词器
pre_model, pre_tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/root/autodl-tmp/models/qwen/Qwen2-7B-Instruct",  # 使用Qwen2.5-7B模型
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 测试问题
test_questions = [
    "我最近总是感觉头晕，应该怎么办？",
    "感冒发烧应该吃什么药？",
    "高血压患者需要注意什么？"
]

for question in test_questions:
    print("\n" + "="*50)
    print(f"问题：{question}")
    print("回答：")
    generate_medical_response(pre_model,pre_tokenizer,medical_prompt, question) 

# 运行评估
evaluation_results = evaluate_model(model, tokenizer, test_data)
print(f"BLEU: {evaluation_results['bleu']:.4f}")
print(f"ROUGE-1: {evaluation_results['rouge1']:.4f}")
print(f"ROUGE-2: {evaluation_results['rouge2']:.4f}")
print(f"ROUGE-L: {evaluation_results['rougeL']:.4f}")

evaluation_results = evaluate_model(pre_model, pre_tokenizer, test_data)
print(f"BLEU: {evaluation_results['bleu']:.4f}")
print(f"ROUGE-1: {evaluation_results['rouge1']:.4f}")
print(f"ROUGE-2: {evaluation_results['rouge2']:.4f}")
print(f"ROUGE-L: {evaluation_results['rougeL']:.4f}")