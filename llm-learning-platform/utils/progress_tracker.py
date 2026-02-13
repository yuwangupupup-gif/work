"""
工具函数:进度可视化与统计
"""

import json
import pandas as pd
from datetime import datetime, timedelta

def generate_progress_report(progress_file="data/progress.json"):
    """生成学习进度报告"""
    
    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            progress = json.load(f)
    except FileNotFoundError:
        return "尚无学习记录"
    
    total = len(progress)
    completed = sum(1 for task in progress.values() if task.get("completed"))
    
    report = f"""
📊 学习进度报告
{'='*50}
总任务数: {total}
已完成: {completed}
完成率: {completed/total*100:.1f}%

最近完成的任务:
"""
    
    recent_tasks = [
        (task_id, task)
        for task_id, task in progress.items()
        if task.get("completed") and task.get("completed_date")
    ]
    
    recent_tasks.sort(key=lambda x: x[1]["completed_date"], reverse=True)
    
    for task_id, task in recent_tasks[:5]:
        report += f"  ✅ {task['completed_date']}: 任务 {task_id}\n"
    
    return report

def export_notes_to_markdown(progress_file="data/progress.json", output_file="学习笔记汇总.md"):
    """导出所有笔记为 Markdown 文件"""
    
    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            progress = json.load(f)
    except FileNotFoundError:
        return "尚无学习记录"
    
    content = "# 大模型学习笔记汇总\n\n"
    content += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    for task_id, task in progress.items():
        if task.get("notes"):
            status = "✅" if task.get("completed") else "⏳"
            content += f"## {status} 任务 {task_id}\n\n"
            content += f"{task['notes']}\n\n"
            content += "---\n\n"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return f"✅ 笔记已导出到 {output_file}"

if __name__ == "__main__":
    print(generate_progress_report())
    print(export_notes_to_markdown())
