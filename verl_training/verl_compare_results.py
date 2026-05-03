"""
对比训练前后的效果
"""
import json
import sys

def load_results(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compare_results(before_file, after_file):
    before = load_results(before_file)
    after = load_results(after_file)
    
    print("="*60)
    print("训练效果对比")
    print("="*60)
    
    print(f"\n训练前模型: {before['model_path']}")
    print(f"训练后模型: {after['model_path']}")
    
    print(f"\n{'指标':<20} {'训练前':<15} {'训练后':<15} {'提升':<15}")
    print("-"*60)
    
    before_acc = before['accuracy']
    after_acc = after['accuracy']
    improvement = after_acc - before_acc
    
    print(f"{'准确率':<20} {before_acc:.2%}          {after_acc:.2%}          {improvement:+.2%}")
    print(f"{'正确数':<20} {before['correct']:<15} {after['correct']:<15} {after['correct'] - before['correct']:+d}")
    print(f"{'总样本':<20} {before['total']:<15} {after['total']:<15}")
    
    print("\n" + "="*60)
    
    # 详细对比每个样本
    print("\n详细对比 (前10个错误改进的样本):")
    improved_count = 0
    for i, (b, a) in enumerate(zip(before['results'], after['results'])):
        if not b['correct'] and a['correct']:
            improved_count += 1
            if improved_count <= 10:
                print(f"\n样本 {i+1}:")
                print(f"  问题: {b['question'][:80]}...")
                print(f"  训练前答案: {b['prediction']} (错误)")
                print(f"  训练后答案: {a['prediction']} (正确)")
                print(f"  标准答案: {b['ground_truth']}")
    
    print(f"\n总共改进: {improved_count} 个样本")
    
    # 统计退化情况
    degraded_count = sum(1 for b, a in zip(before['results'], after['results']) if b['correct'] and not a['correct'])
    print(f"退化样本: {degraded_count} 个")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("用法: python verl_compare_results.py <训练前结果.json> <训练后结果.json>")
        sys.exit(1)
    
    compare_results(sys.argv[1], sys.argv[2])
