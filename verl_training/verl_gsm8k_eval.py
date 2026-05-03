"""
GSM8K 训练前后效果对比脚本
"""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import re
from tqdm import tqdm

def extract_answer(text):
    """从模型输出中提取答案"""
    # 匹配 #### 后面的数字
    match = re.search(r'####\s*([\-0-9\.]+)', text)
    if match:
        return match.group(1).strip()
    
    # 匹配 \boxed{}
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).strip()
    
    # 取最后一行数字
    lines = text.strip().split('\n')
    for line in reversed(lines):
        numbers = re.findall(r'[\-0-9\.]+', line)
        if numbers:
            return numbers[-1].strip()
    
    return ""

def load_model(model_path, device="cuda"):
    """加载模型"""
    print(f"加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    return model, tokenizer

def evaluate_model(model, tokenizer, dataset, max_samples=100, max_new_tokens=512):
    """评估模型在 GSM8K 上的准确率"""
    correct = 0
    total = 0
    results = []
    
    for i, example in enumerate(tqdm(dataset[:max_samples], desc="评估中")):
        # 获取问题
        question = example.get('question', '')
        ground_truth = str(example.get('answer', '')).strip()
        
        # 提取 ground truth 中的数字
        gt_match = re.search(r'####\s*([\-0-9\.]+)', ground_truth)
        if gt_match:
            gt_answer = gt_match.group(1).strip()
        else:
            gt_answer = ground_truth
        
        # 构建 prompt
        messages = [
            {"role": "user", "content": question + " Let's think step by step and output the final answer after \"####\"."}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 生成答案
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.6,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 解码输出
        generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        pred_answer = extract_answer(generated_text)
        
        # 判断是否正确
        is_correct = pred_answer == gt_answer
        if is_correct:
            correct += 1
        total += 1
        
        results.append({
            'question': question,
            'ground_truth': gt_answer,
            'prediction': pred_answer,
            'generated': generated_text,
            'correct': is_correct
        })
        
        # 打印前几个例子
        if i < 3:
            print(f"\n--- 示例 {i+1} ---")
            print(f"问题: {question[:100]}...")
            print(f"标准答案: {gt_answer}")
            print(f"模型答案: {pred_answer}")
            print(f"是否正确: {is_correct}")
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='模型路径')
    parser.add_argument('--output_file', default='eval_results.json', help='结果保存路径')
    parser.add_argument('--max_samples', type=int, default=100, help='评估样本数')
    parser.add_argument('--dataset', default='openai/gsm8k', help='数据集名称')
    args = parser.parse_args()
    
    # 加载数据集
    print(f"加载数据集: {args.dataset}")
    dataset = load_dataset(args.dataset, 'main')
    test_dataset = dataset['test']
    
    # 加载模型
    model, tokenizer = load_model(args.model_path)
    
    # 评估
    accuracy, results = evaluate_model(model, tokenizer, test_dataset, args.max_samples)
    
    print(f"\n{'='*50}")
    print(f"评估完成!")
    print(f"总样本数: {len(results)}")
    print(f"正确数: {sum(r['correct'] for r in results)}")
    print(f"准确率: {accuracy:.2%}")
    print(f"{'='*50}")
    
    # 保存结果
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'model_path': args.model_path,
            'accuracy': accuracy,
            'total': len(results),
            'correct': sum(r['correct'] for r in results),
            'results': results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到: {args.output_file}")

if __name__ == '__main__':
    main()
