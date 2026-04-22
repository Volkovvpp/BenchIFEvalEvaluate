import pandas as pd
import ast
import os
from dotenv import load_dotenv
load_dotenv()

def parse_list(value):
    """Безопасно превращает строку '[True, False]' в список boolean"""
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except Exception:
            return []
    elif isinstance(value, list):
        return value
    return []

def calculate_ifeval_metrics(file_path):
    if not os.path.exists(file_path):
        print(f"Ошибка: Файл {file_path} не найден.")
        return

    df = pd.read_csv(file_path)

    if df['prompt_level_strict_acc'].dtype == object:
        df['prompt_level_strict_acc'] = df['prompt_level_strict_acc'].astype(str).str.lower() == 'true'
        df['prompt_level_loose_acc'] = df['prompt_level_loose_acc'].astype(str).str.lower() == 'true'

    prompt_strict = df['prompt_level_strict_acc'].mean() * 100
    prompt_loose = df['prompt_level_loose_acc'].mean() * 100

    inst_strict_series = df['inst_level_strict_acc'].apply(parse_list).explode()
    inst_loose_series = df['inst_level_loose_acc'].apply(parse_list).explode()

    inst_strict_series = inst_strict_series.dropna().astype(bool)
    inst_loose_series = inst_loose_series.dropna().astype(bool)

    inst_strict = inst_strict_series.mean() * 100
    inst_loose = inst_loose_series.mean() * 100

    inst_strict = inst_strict_series.mean() * 100
    inst_loose = inst_loose_series.mean() * 100

    overall_score = (prompt_strict + prompt_loose + inst_strict + inst_loose) / 4
    overall_strict = (prompt_strict + inst_strict) / 2

    print("=" * 55)
    print(f"ОТЧЕТ ПО МОДЕЛИ: {os.path.basename(file_path)}")
    print(f"Всего заданий (Prompts): {len(df)}")
    print(f"Всего инструкций (Instructions): {len(inst_strict_series)}")
    print("-" * 55)
    print(f"Prompt-level Strict:  {prompt_strict:>6.2f}%")
    print(f"Prompt-level Loose:   {prompt_loose:>6.2f}%")
    print(f"Inst-level Strict:    {inst_strict:>6.2f}%")
    print(f"Inst-level Loose:     {inst_loose:>6.2f}%")
    print("=" * 55)
    # Выводим итоговые баллы капсом, чтобы они выделялись
    print(f"OVERALL SCORE:      {overall_score:>6.2f}%")
    print(f"STRICT SCORE:       {overall_strict:>6.2f}%")
    print("=" * 55)

if __name__ == "__main__":
    # выбрать свой путь
    calculate_ifeval_metrics("results_plain/results-google:gemini-3.1-pro-preview.csv")