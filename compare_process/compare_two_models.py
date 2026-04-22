import csv
import os


def find_model_regressions(
        file1_path: str,
        file2_path: str,
        output_path: str,
        id_column: str = 'doc_id',
        accuracy_column: str = 'prompt_level_strict_acc'
):
    """
    Сравнивает два существующих файла и находит регрессии (где модель 1 была успешна, а модель 2 ошиблась).
    """

    if not os.path.exists(file1_path):
        print(f"Ошибка: Файл 1 не найден: {file1_path}")
        return
    if not os.path.exists(file2_path):
        print(f"Ошибка: Файл 2 не найден: {file2_path}")
        return

    print(f"Шаг 1: Анализ эталонного файла (Модель 1): {file1_path}")
    model1_successes = set()

    try:
        with open(file1_path, mode='r', encoding='utf-8') as f1:
            reader = csv.DictReader(f1)
            for row in reader:
                if row.get(accuracy_column) == 'True':
                    model1_successes.add(row.get(id_column))
    except Exception as e:
        print(f"Ошибка при чтении файла 1: {e}")
        return

    print(f"У модели 1 найдено успешно выполненных задач: {len(model1_successes)}")

    print(f"Шаг 2: Поиск регрессий во втором файле: {file2_path}")
    error_rows = []
    header = []

    try:
        with open(file2_path, mode='r', encoding='utf-8') as f2:
            reader = csv.DictReader(f2)
            header = reader.fieldnames

            for row in reader:
                doc_id = row.get(id_column)
                # Условие регрессии: первая модель смогла (True), а вторая — нет (False)
                if doc_id in model1_successes and row.get(accuracy_column) == 'False':
                    error_rows.append(row)
    except Exception as e:
        print(f"Ошибка при чтении файла 2: {e}")
        return

    print(f"Найдено регрессий: {len(error_rows)}")

    if not error_rows:
        print("Регрессий не обнаружено. Выходной файл не будет создан.")
        return

    try:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_path, mode='w', encoding='utf-8', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=header)
            writer.writeheader()
            writer.writerows(error_rows)
        print(f"Результаты успешно сохранены в: {output_path}")
    except Exception as e:
        print(f"Ошибка при записи файла: {e}")


if __name__ == "__main__":
    FILE_MODEL_1 = 'compare_process/results-google:gemini-3.1-pro-preview.csv'
    FILE_MODEL_2 = 'compare_process/results-6970986b4e761c9b775f86c6.csv'
    FILE_OUTPUT = 'compare_process/elen_errors_compare_gemini.csv'

    find_model_regressions(
        file1_path=FILE_MODEL_1,
        file2_path=FILE_MODEL_2,
        output_path=FILE_OUTPUT
    )